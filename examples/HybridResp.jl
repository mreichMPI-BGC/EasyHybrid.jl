using EasyHybrid
using MRTools
using Flux
##

GLMakie.activate!(;float=true)
update_theme!(fontsize=2 * 14)

### Model definition
struct RespModel_Q10_geogen
    Rb_chain::Flux.Chain
    Rb_predictors::AbstractArray{Symbol}

    predictors::AbstractArray{Symbol}
    forcing::AbstractArray{Symbol}

    Q10
    geog_flux
end

function RespModel_Q10_geogen(Rb_predictors::AbstractArray{Symbol}, Q10=[1.5f0], geog_flux=[0f0])
    Rb_ch = chain4Rbfun(length(Rb_predictors))
    RespModel_Q10_geogen(Rb_ch, Rb_predictors, Base.union(Rb_predictors), [:T],  Q10, geog_flux)
end

chain4Rbfun(nInVar) = Flux.Chain(
    BatchNorm(nInVar, affine=true),
    Dense(nInVar => 15, relu),
    #GRU(5 => 5),
    #Dense(9 => 5, σ), 
    Dense(15 => 1, σ),
)
  
function (m::RespModel_Q10_geogen)(x)
    res = m(x, :infer)
    return res.RECO
end

function (m::RespModel_Q10_geogen)(x, s::Symbol) # the Val thing did not work now..(?)
        v(sym::Symbol) = Array(x([sym]))
        Rb_input4Chain = select_predictors(x, m.Rb_predictors)
        Rb = 100f0 * m.Rb_chain(Rb_input4Chain)
        #SW_IN = Matrix(x([:SW_IN]))
        #T = Matrix(x([:T]))
        #µmol/m2/s1 =  J/s/m2 * g/MJ / g/mol
        Reco = Rb .* m.Q10[1] .^ (0.1f0(v(:T) .- 15.0f0)) .+ m.geog_flux[1]
        return((; Rb, RECO=Reco))
end

  
  # Call @functor to allow for training the custom model
Flux.@functor RespModel_Q10_geogen

##
file=string(@__DIR__) * "\\..\\data\\AT-Neu.HH.csv"
##
df = @chain file CSV.read(DataFrame, missingstring="NA") @transform(:NEE = coalesce(:NEE, :RECO_NT-:GPP_NT))
transform!(df, names(df, Number) .=> ByRow(Float32), renamecols=false)
# mapcols(df) do col
#     eltype(col) === Int ? Float32.(col) : col
# end
df = @chain df begin 
        @transform :Rb_syn = max(:Rb_syn, 0.0f0)
        @transform :RECO_syn = :Rb_syn * 1.5f0 ^ (0.1f0(:TA - 15.0f0)) + 4.0f0
        @transform :RECO_syn = @bycol :RECO_syn .+ 2f0randn32(nrow(df)) ## adding some noise
end


# Clean df from non numeric and make a KeyedArray
df =select(df, names(df, Number)) |> disallowmissing

# Instantaniate model
fpmod = RespModel_Q10_geogen([:SW_POT_sm_diff, :SW_POT_sm],[2.0f0], [0f0])

##
#Define latents to show and fit
latents2compare = [Symbol(v) => Symbol(v, "_syn") for v in ["Rb"]]

res=fit_df!(fpmod, df, [:RECO_syn], Flux.mse, n_epoch=500, batchsize=480, opt=Adam(0.01), 
    parameters2record=[:Q10, :geog_flux], latents2record=latents2compare, patience=300, stateful=false, renamer=[:TA => :T])
##

##
fig=Figure(resolution=(1920, 1080))
ax_TA = Axis(fig[1,1], ylabel = "Tair"); lines!(ax_TA, df.year .+ df.doy./365, df.TA)
ax_Rb = Axis(fig[2,1], ylabel= "Rb"); lines!(ax_Rb, df.year .+ df.doy./365, df.Rb_syn)
ax_RE = Axis(fig[3,1], ylabel ="Respiration"); lines!(ax_RE, df.year .+ df.doy./365, df.RECO_syn)
fig
##

pred_df = predict_all2df(res.bestModel, rename(df, :TA => :T))
scatter(pred_df.Rb, color=(:black, 0.05))
scatter!(pred_df.Rb_syn, color=(:red, 0.05))

scatter(pred_df.RECO , pred_df.RECO_syn , color=(:black, 0.05))
ablines!(0,1, color=:red, linewidth=3, linestyle=:dash)

println(res.bestModel.Q10)
println(res.bestModel.geog_flux)


### STOP ###

### Fit Rb directly with a FF Neural net
#resRb=fit_df!(df, [:SW_POT_sm, :SW_POT_sm_diff], :Rb_syn, (m, d) -> Flux.mse(m(d.x), d.y), model=chain4Rbfun(2), n_epoch=200, batchsize=480)




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