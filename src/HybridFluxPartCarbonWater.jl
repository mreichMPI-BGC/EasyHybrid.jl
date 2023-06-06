include("hybrid_mr.jl")
GLMakie.activate!(;float=true)
update_theme!(fontsize=2 * 14)

### Model definition for a FP of carbon and water....
struct FluxPartModel_NEE_ET2
    RUE_chain::Flux.Chain
    RUE_predictors::AbstractArray{Symbol}
    Rb_chain::Flux.Chain
    Rb_predictors::AbstractArray{Symbol}

    WUE_chain::Flux.Chain
    WUE_predictors::AbstractArray{Symbol}

    Ecoeff_chain::Flux.Chain
    Ecoeff_predictors::AbstractArray{Symbol}

    predictors::AbstractArray{Symbol}
    forcing::AbstractArray{Symbol}

    Q10
end

function FluxPartModel_NEE_ET2(RUE_predictors::AbstractArray{Symbol}, 
    Rb_predictors::AbstractArray{Symbol},
    WUE_predictors::AbstractArray{Symbol},
    Ecoeff_predictors::AbstractArray{Symbol},
     Q10=[1.5f0])

    RUE_ch = chain4RUEfun(length(RUE_predictors))
    Rb_ch = chain4Rbfun(length(Rb_predictors))
    WUE_ch = chain4Rbfun(length(WUE_predictors))
    Ecoeff_ch = chain4Rbfun(length(Ecoeff_predictors))
    FluxPartModel_NEE_ET2(RUE_ch, RUE_predictors, Rb_ch, Rb_predictors, WUE_ch, WUE_predictors, Ecoeff_ch, Ecoeff_predictors, 
            union(RUE_predictors, Rb_predictors, Ecoeff_predictors, WUE_predictors), [:SW_IN_F, :TA_F],  Q10)
end

chain4RUEfun(nInVar) = Flux.Chain(
    BatchNorm(nInVar, affine=true),
    Dense(nInVar => 15, relu),
    #GRU(5 => 5),
    #Dense(12 => 12, σ), 
    Dense(15 => 1, σ),
)
chain4Rbfun(nInVar) = Flux.Chain(
    BatchNorm(nInVar, affine=true),
    Dense(nInVar => 15, relu),
    #GRU(5 => 5),
    #Dense(9 => 5, σ), 
    Dense(15 => 1, σ),
)
  
function (m::FluxPartModel_NEE_ET2)(x)
    res = m(x, :infer)
    return vcat(res.NEE, res.ET)
end

function (m::FluxPartModel_NEE_ET2)(x, s::Symbol)
        v(sym::Symbol) = Array(x([sym]))
        RUE_input4Chain = select_predictors(x, m.RUE_predictors)    
        Rb_input4Chain = select_predictors(x, m.Rb_predictors)
        WUE_input4Chain = select_predictors(x, m.WUE_predictors)
        Ecoeff_input4Chain = select_predictors(x, m.Ecoeff_predictors)
        Rb = 100f0 * m.Rb_chain(Rb_input4Chain)
        RUE = 1f0 * m.RUE_chain(RUE_input4Chain)
        Ecoeff = 1f0 * m.Ecoeff_chain(Ecoeff_input4Chain)
        WUE = 50f0 * m.WUE_chain(WUE_input4Chain)
        #SW_IN = Matrix(x([:SW_IN]))
        #TA = Matrix(x([:TA]))
        #µmol/m2/s1 =  J/s/m2 * g/MJ / g/mol
        GPP   =    v(:SW_IN_F) .* RUE ./ 12.011f0  
        Reco = Rb .* m.Q10[1] .^ (0.1f0(v(:TA_F) .- 15.0f0))

        Tr = GPP ./ (abs.(WUE) .+ 0.001f0)
        E = Ecoeff .* v(:SW_IN_F)
        return((; RUE, Rb, Tr, E, GPP=GPP, RECO=Reco, NEE=Array(Reco) - Array(GPP),  ET=Array(Tr) + Array(E)))
end

  
  # Call @functor to allow for training the custom model
Flux.@functor FluxPartModel_NEE_ET2
##

dft= CSV.read("C:/Users\\mreichstein\\Desktop\\Fluxnet2015Data/FLX_FI-Hyy_FLUXNET2015_FULLSET_HH_plus.csv", DataFrame)  

dft2 = @chain begin
    dft
    @transform(:DateTime = DateTime(:DateTime, dateformat"yyyy-mm-dd HH:MM:SS"))
    @select(:DateTime, :year=year(:DateTime), :doy=dayofyear(:DateTime), :hour = hour(:DateTime), :TA_F, :SW_IN_F, :VPD_F, {r"^P|CUT_50|LE|^H_"})
    @transform :Rb_syn = :RECO_NT_CUT_50 / 1.5^(0.1*(:TA_F-15))
    @transform :RECO_syn = :Rb_syn * 1.5f0 ^ (0.1f0(:TA_F - 15.0f0))
    @transform :GPP_syn = :SW_IN_F * exp(-(:TA_F-25)^2/625) * 1/80.
    @transform :NEE_syn = :RECO_syn - :GPP_syn
    @transform :Tr_syn = 1f0 * :GPP_syn / sqrt(:VPD_F + 1) * 5. / 6. * 18e-6 * 3600 ##  WUE = 6 µmol/mmol
    @transform :E_syn = 1f0 * :LE_F_MDS/2.257e6*3600 - :Tr_syn
    @transform :ET_syn = :E_syn + :Tr_syn

    @subset(:year in 2003:2004, :doy in 100:300)
    transform(_, names(_, Number) .=> ByRow(Float32), renamecols=false)
end 


fpmod = FluxPartModel_NEE_ET2([:TA_F, :SW_IN_F], [:doy], [:VPD_F, :hour],[:doy])
pr=predict_all(fpmod, select(dft2, Not(:DateTime)))

dk=select(dft2, Not(:DateTime)) |> tokeyedArray
simpleLoss(model, data) = multiloss(model(data[1]), data[2]) 
simpleLoss(fpmod, (x=dk, y=dk([:NEE_syn, :ET_syn])))



latents2compare = [Symbol(v) => Symbol(v, "_syn") for v in ["GPP", "RECO", "NEE", "Tr", "E", "ET"]]

res=fit_df!(fpmod, select(dft2, Not(:DateTime)), [:NEE_syn, :ET_syn], Flux.mse, n_epoch=200, batchsize=480, opt=Adam(0.01), parameters2record=[:Q10], latents2record=latents2compare, patience=300, stateful=false)
pr=predict_all(res.bestModel, select(dft2, Not(:DateTime)))

latents2compare = ["GPP" => "GPP_NT_CUT_50", "NEE" => "NEE_CUT_50", "ET" => "ET_syn"]#Symbol(v) => Symbol(v, "_syn") for v in ["GPP", "RECO", "NEE", "Tr", "E", "ET"]]

multiloss(yhat, y) = begin
    nvar=size(yhat,1)
    return sum(Flux.mse(yhat[i,:], y[i,:])/Flux.mse(fill(mean(y[i,:]), size(y[i,:])), y[i,:]) for i in 1:nvar)

end

multiloss2(yhat, y) = begin
    nvar=size(yhat,1)
    varWeights=[1.0, 10.0]  .|> Float32
    return sum(Flux.mse(yhat[i,:], y[i,:]) .* varWeights[i] for i in 1:nvar)

end


res=fit_df!(fpmod, select(dft2, Not(:DateTime)), [:NEE_CUT_50, :ET_syn], multiloss, n_epoch=200, batchsize=480, opt=Adam(0.01), parameters2record=[:Q10], latents2record=latents2compare, patience=300, stateful=false)
pr=predict_all(res.bestModel, select(dft2, Not(:DateTime))) |> evec
