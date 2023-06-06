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
    Ecoeff_ch = chain4Rbfun(length(Ecoeff_predictors))
    FluxPartModel_NEE_ET2(RUE_ch, RUE_predictors, Rb_ch, Rb_predictors, WUE_ch, WUE_predictors, Ecoeff_ch, Ecoeff_predictors, 
            union(RUE_predictors, Rb_predictors, Ecoeff), [:SW_IN, :TA],  Q10)
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

function (m::FluxPartModel_NEE_ET2)(x, s::Symbol) # the Val thing did not work now..(?)
        v(sym::Symbol) = Array(x([sym]))
        RUE_input4Chain = select_predictors(x, m.RUE_predictors)    
        Rb_input4Chain = select_predictors(x, m.Rb_predictors)
        WUE_input4Chain = select_predictors(x, m.WUE_predictors)
        Ecoeff_input4Chain = select_predictors(x, m.Ecoeff_predictors)
        Rb = 100f0 * m.Rb_chain(Rb_input4Chain)
        RUE = 1f0 * m.RUE_chain(RUE_input4Chain)
        Ecoeff = 1f0 * m.Ecoeff_chain(Ecoeff_input4Chain)
        WUE = 1f0 * m.Ecoeff_chain(WUE_input4Chain)
        #SW_IN = Matrix(x([:SW_IN]))
        #TA = Matrix(x([:TA]))
        #µmol/m2/s1 =  J/s/m2 * g/MJ / g/mol
        GPP   =    v(:SW_IN) .* RUE ./ 12.011f0  
        Reco = Rb .* m.Q10[1] .^ (0.1f0(v(:TA) .- 15.0f0))

        Tr = GPP / WUE
        Evap = Ecoeff .* v(:SW_IN)
        return((; RUE, Rb, Tr, Evap, GPP=GPP, RECO=Reco, NEE=Array(Reco) - Array(GPP),  ET=Array(Tr) + Array(Evap)))
end

  
  # Call @functor to allow for training the custom model
Flux.@functor FluxPartModel_NEE_ET2

##


### Model definition
struct FluxPartModel_Q10
    RUE_chain::Flux.Chain
    RUE_predictors::AbstractArray{Symbol}
    Rb_chain::Flux.Chain
    Rb_predictors::AbstractArray{Symbol}

    predictors::AbstractArray{Symbol}
    forcing::AbstractArray{Symbol}

    Q10
end

function FluxPartModel_Q10(RUE_predictors::AbstractArray{Symbol}, Rb_predictors::AbstractArray{Symbol}, Q10=[1.5f0])
    RUE_ch = chain4RUEfun(length(RUE_predictors))
    Rb_ch = chain4Rbfun(length(Rb_predictors))
    FluxPartModel_Q10(RUE_ch, RUE_predictors, Rb_ch, Rb_predictors, union(RUE_predictors, Rb_predictors), [:SW_IN, :TA],  Q10)
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
  
function (m::FluxPartModel_Q10)(x)
    res = m(x, :infer)
    return res.NEE
end

function (m::FluxPartModel_Q10)(x, s::Symbol) # the Val thing did not work now..(?)
        v(sym::Symbol) = Array(x([sym]))
        RUE_input4Chain = select_predictors(x, m.RUE_predictors)    
        Rb_input4Chain = select_predictors(x, m.Rb_predictors)
        Rb = 100f0 * m.Rb_chain(Rb_input4Chain)
        RUE = 1f0 * m.RUE_chain(RUE_input4Chain)
        #SW_IN = Matrix(x([:SW_IN]))
        #TA = Matrix(x([:TA]))
        #µmol/m2/s1 =  J/s/m2 * g/MJ / g/mol
        GPP   =    v(:SW_IN) .* RUE ./ 12.011f0  
        Reco = Rb .* m.Q10[1] .^ (0.1f0(v(:TA) .- 15.0f0))
        return((; RUE, Rb, GPP=GPP, RECO=Reco, NEE=Array(Reco) - Array(GPP)))
end

  
  # Call @functor to allow for training the custom model
Flux.@functor FluxPartModel_Q10


##
df = @c "./data/AT-Neu.HH.csv" CSV.read(DataFrame, missingstring="NA") @transform(:NEE = coalesce(:NEE, :RECO_NT-:GPP_NT))
transform!(df, names(df, Number) .=> ByRow(Float32), renamecols=false)
# mapcols(df) do col
#     eltype(col) === Int ? Float32.(col) : col
# end
df = @chain df begin 
        @transform :Rb_syn = max(:Rb_syn, 0.0f0)
        @transform :RECO_syn = :Rb_syn * 1.5f0 ^ (0.1f0(:TA - 15.0f0))
        @transform :NEE_syn = :RECO_syn - :GPP_syn
end

#Create further variables
dfs =@c df begin 
    #@transform :NEE_SWin=@bycol mapwindow2(regBivar, :SW_IN, :NEE, 49)
    #unnest("NEE_SWin")
    @transform :NEE_SWin=@bycol mapwindow2((x,y)->regBivar(Float64.(x),Float64.(y), quantreg), :SW_IN, :NEE, 49)
    unnest("NEE_SWin")
    @transform {} = Float32({r"^NEE_SWin"})
    #transform(:lmres => AsTable)
    #@transform :NEE_slope=:lmres[2]
    #@transform :NEE_inter=:lmres[1]
    #select(Not(:lmres))
   end

##

# Clean df from non numeric and make a KeyedArray
df =select(df, names(df, Number)) |> disallowmissing
dk = df |> tokeyedArray

#Test forward run
NEE = fpmod(dk) |> vec;
scatter(NEE)
allinfo = fpmod(dk,:infer) |> evec
scatter(allinfo.RUE)
##
#Define model instance and fit
fpmod = FluxPartModel_Q10([:TA, :VPD], [:SW_POT_sm_diff, :SW_POT_sm],[2.0f0])
latents2compare = [Symbol(v) => Symbol(v, "_syn") for v in ["GPP", "RECO", "RUE", "Rb"]]
res=fit_df!(fpmod, df, [:NEE_syn], Flux.mse, n_epoch=100, batchsize=480, opt=Adam(0.01), parameters2record=[:Q10], latents2record=latents2compare, patience=300, stateful=false)
##

infer = res.bestModel(dk, :infer) 
scatter(infer.Rb, color=(:black, 0.05))
scatter!(dk(:Rb_syn), color=(:red, 0.05))

scatter(infer.Reco , dt.RECO_syn , color=(:black, 0.05))
ablines!(0,1, color=:red, linewidth=3, linestyle=:dash)

println(res.bestModel.Q10)

### Fit Rb directly with a FF Neural net
resRb=fit_df!(dt, [:SW_POT_sm, :SW_POT_sm_diff], :Rb_syn, (m, d) -> Flux.mse(m(d.x), d.y), model=chain4Rbfun(2), n_epoch=200, batchsize=480)




##
ok=findall(dt.SW_IN .< 0.1)
scatter(infer.Reco[ok], dt.NEE[ok])
scatter!(dt.RECO_NT[ok], dt.NEE[ok], color=(:red, 0.2))
ablines!(0,1, color=:black, linewidth=3, linestyle=:dash)
current_figure()
##

lines(dt.SW_POT_sm_diff)




ncd = NCDataset("DE-Tha_2010_gf.nc")
data = @chain begin
    mapreduce(x -> DataFrame(x => ncd[x][:]), hcat, keys(ncd)[2:end])
    @transform begin
        :Tair = :Tair - 273.15
        :Tair_min = :Tair_min - 273.15
        :PAR = 0.45 * :SW_IN * 86400e-6
        :GPP = :GPP * 86400e3
        :VPDday = :VPDday * 100
        #"{}" = {r"^Tair"} * 1000

    end
    NamedTuple.(eachrow(_))
    invert
    NamedTuple{keys(_)}(Float32.(d) for d in _)
end

##
chain4RUE = Flux.Chain(
    #BatchNorm(2, affine=true),
    Dense(2 => 5, σ),
    #GRU(5 => 5),
    Dense(5 => 5), 
    Dense(5 => 1, softplus),
)

chain4Rb = Flux.Chain(
    Dense(5 => 5, relu),
    Dense(5 => 5, relu),
    Dense(5 => 1, sigmoid)
)
##
hybridRUEmodel = CustomRUEModel([:VPDday, :Tair_min, :SW_IN])

GPPmod = hybridRUEmodel(data)
#RUEmod = hybridRUEmodel(data, :infer)

lines(GPPmod |> vec)
##
# lines(RUEmod |> vec)

# loss(m, d) = Flux.mse(m(d.x), d.y)

# ##
# opt_state = Flux.setup(Adam(0.001), hybridRUEmodel)   # explicit setup of optimiser momenta

# dloader = Flux.DataLoader((; x= data, y=reshape(data.GPP, 1, :)), batchsize=30, shuffle=true, partial=true)
# dloaderAll = Flux.DataLoader((; x= data, y=reshape(data.GPP, 1, :)), batchsize=length(data.GPP), shuffle=true, partial=true)
# ##
# epochs=100
# losses=[loss(hybridRUEmodel, dloaderAll |> first)]
# for e in 1:epochs
#     for d in dloaderAll
#         ∂L∂m = gradient(loss, hybridRUEmodel, d)[1]
#         Flux.update!(opt_state, hybridRUEmodel, ∂L∂m)
#     end
#     push!(losses, loss(hybridRUEmodel, dloaderAll |> first) )
#    # Flux.train!(loss, hybridRUEmodel, dloader, opt_state)
# end


struct CustomRUEModel
    RUEchain::Flux.Chain
    RUEpredictors::AbstractArray{Symbol}
    #Rbchain::Flux.Chain
end

function CustomRUEModel(predictors::AbstractArray{Symbol})
    ch = chain4RUEfun(length(predictors))
    CustomRUEModel(ch, predictors)
end

chain4RUEfun(nInVar) = Flux.Chain(
    #BatchNorm(2, affine=true),
    Dense(nInVar => 5, σ),
    #GRU(5 => 5),
    Dense(5 => 5), 
    Dense(5 => 1, softplus),
)
  
function (m::CustomRUEModel)(x)
    # Arbitrary code can go here, but note that everything will be differentiated.
    # Zygote does not allow some operations, like mutating arrays.
    #return x.PAR .*  x.fPAR .* vec(m.RUEchain([x.PAR x.VPDday x.Tair_min] |> transpose))

    input4Chain = mapreduce(v->x[v], hcat, m.RUEpredictors) |> transpose

    return reshape(x.PAR .*  x.fPAR .* vec(m.RUEchain(input4Chain)), 1, :)
end

function (m::CustomRUEModel)(x, stateful=true)
    # Arbitrary code can go here, but note that everything will be differentiated.
    # Zygote does not allow some operations, like mutating arrays.
    #return x.PAR .*  x.fPAR .* vec(m.RUEchain([x.PAR x.VPDday x.Tair_min] |> transpose))
    #!stateful && return x.PAR .*  x.fPAR .* vec(m.RUEchain([ x.VPDday x.Tair_min] |> transpose))
    out = reduce(hcat, [m.RUEchain(s) for s in eachslice([ x.VPDday x.Tair_min], dims=1)])
    

end

function (m::CustomRUEModel)(x, mode::Symbol)
    mode == :infer && vec(m.RUEchain([ x.VPDday x.Tair_min] |> transpose))
end
  
  # Call @functor to allow for training. Described below in more detail.
Flux.@functor CustomRUEModel

# fit_df  <- function(df=NULL, batchSize=1000L, lr=0.05, n_epoch=100L, model=NULL, startFromResult=NULL, 
#                     predictors=c("WS", "VPD","TA", "SW_IN", "SW_POT_sm", "SW_POT_sm_diff"), target="NEE", seqID=NULL, seqLen=NA, 
#                     weights=NULL, checkpoint="R_checkpoint.pt", DictBased=F,
#                     patience=50, lossFunc=lossFuncs$trimmedLoss, justKeepOnTraining=FALSE, ...) {

##