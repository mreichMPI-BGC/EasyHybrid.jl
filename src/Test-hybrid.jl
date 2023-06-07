### Tests the fit_df! and RNNfit_df! functions with a simple synthetic data set
##
using MRTools, EasyHybrid
using Flux
#include("LinHybMod.jl")

GLMakie.activate!(; float=true)
update_theme!(fontsize=2 * 14)


################################
## Case 1: Stateless, single output
## Create some data y = a ⋅ x1 + b, where a = f(x2,x3), b=2
df = @chain DataFrame(rand(Float32, 1000, 3), :auto) begin
    @transform :a_syn = exp(-5.0f0((:x2 - 0.7f0))^2.0f0) + :x3 / 10.0f0
    @aside b = 2.0f0
    @transform :obs = :a_syn * :x1 + b
    @transform :pred_syn = :obs
    @transform :seqID = @bycol repeat(1:100, inner=10)

end

## Illustrate data
fig = HDFigure()
draw!(fig[1,1:2], data(df) * mapping(:x1, :obs, color=:x2); axis=(;limits=(nothing, (0,nothing))) );
Colorbar(fig[1,3], current_axis().scene.plots[1], label="x2")
draw!(fig[1,4], data(df) * mapping(:x2, :a_syn))
fig
##

## Instantiate model
hymod = LinearHybridModel([:x2, :x3], [:x1], 1, 5, b=[0.0f0])

# Fit the model with the non-stateful function fit_df! to the first half of the data set
# One does not need to put predictors explicitly, if they are explicit in the model
res = fit_df!(hymod, df[1:500,:], [:obs], Flux.mse, n_epoch=500, batchsize=100, opt=Adam(0.01), parameters2record=[:b],
    latents2record=[:pred => :obs, :a => "a_syn"], patience=300, stateful=false);

test_df = df[501:1000, :]

### Make a direct evaluation
fig2=evalfit(res, test_df)

## or make explicit predictions including latents for 2nd half of the dataset (aka test set)
test_df_pred = predict_all2df(res.bestModel,test_df)
# ... and plot variation of latent a_syn in Figure above
scatter!(fig.content[3], test_df_pred.x2,test_df_pred.a_syn, color=(:red,0.5))



################################################
#### Excursion
#### Lower level access to model via KeyedArray
dk = tokeyedArray(df) 
hymod(dk)

#  Compute some loss function ...
Flux.mse(hymod(dk), dk([:obs])) # Mean sq error
Flux.mae(hymod(dk), dk([:obs])) # Mean abs error
##

### One can use fit_df! for fitting a simple (non-hybrid) NN)
#   forcing in fit_df! needs to be set to [] explicitly then (else m-dispatch would ask for model.forcing)
invars = [:x1, :x2, :x3]
nvars = invars |> length
mychain = Flux.Chain(
    BatchNorm(nvars),
    Dense(nvars => 5, σ),
    Dense(5 => 1),
)
res3 = fit_df!(mychain, df, invars, [], [:obs], Flux.mse, n_epoch=500, batchsize=100, opt=Adam(0.01), parameters2record=nothing, patience=300, stateful=false, shuffle=false)

##

######################################################
### Case 2: Latent variable is dynamically evolving to be represented with an RNN
### Physical part of the model is not stateful 
######################################################
@chain df begin
    groupby(:seqID)
    @transform! :a_dyn_syn = @bycol cumsum(:x2 .- :x3)
    @transform! :obs_dyn = :a_dyn_syn .* :x1 + 2.0f0
    @transform! :pred_syn = :obs_dyn
end

## Slope is a dynamic random walk
data(df) * mapping([:a_dyn_syn, :obs_dyn], col=dims(1)) * visual(Lines) |> draw
## Result depends on x1 and slope a_dyn_syn
data(df) * mapping(:x1, :obs_dyn, color=:a_dyn_syn)  |> draw

## Using stateless model of course doesn't really work (but manage as expected to predict the first point)
hymod = LinearHybridModel([:x2, :x3], [:x1], 1, 5, nn_chain=DenseNN, b=[0.0f0])
dynres2 = fit_df!(hymod, df, [:obs_dyn], Flux.mse, stateful=false, n_epoch=100, batchsize=10, opt=Adam(0.01), parameters2record=[:b], patience=50)

# Hence, define dynamic model, where a is predicted with a GRU
hy_dynmod = LinearHybridModel([:x2, :x3], [:x1], 1, 5, nn_chain=GRU_NN, b=[0.0f0])
## Using stateful model - with stateful = true the df is grouped into sequences according to seqID and batched along this
dynres2 = fit_df!(hy_dynmod, df, [:obs_dyn], Flux.mse, stateful=true, n_epoch=500, batchsize=10, opt=Adam(0.05),
    parameters2record=[:b], latents2record=[:a => :a_dyn_syn, :pred => :pred_syn], patience=50)

### Make a direct evaluation
fig3=evalfit(dynres2, df)

## or make explicit predictions including latents 
df_pred = predict_all2df(dynres2, df)



######################################################
### Case 3: Physical part of the model is stateful 
######################################################
df = @chain DataFrame(rand(Float32, 1000, 3), :auto) begin
    @transform :seqID = @bycol repeat(1:50, inner=20)
    @transform :a_syn = exp(-5.0f0((:x2 - 0.7f0))^2.0f0) + :x3 / 10.0f0
    @aside b = 2.0f0
    @aside obs_ini = 100.0f0
    @transform :force = :a_syn * :x1 + b
    @transform :force = 1.0f0 - exp(-:force)
    #@transform :force = @bycol :force ./ maximum(:force)
    @groupby :seqID
    @transform :obs = @bycol accumulate(*, :force, init=obs_ini)
    @transform :pred_syn = :obs
end


### Model definition

struct innerModel{S}
    state0::S
end

function (m::innerModel{S})(h, x) where {S}
    return h .* x, h .* x
end
Flux.@functor innerModel


struct LinearStateFullModel
    a_chain::Flux.Chain
    a_predictors::AbstractArray{Symbol}

    predictors::AbstractArray{Symbol}
    forcing::AbstractArray{Symbol}

    b
    stateupdater
end

stateModel(h, x) = (h .* x, h .* x)

function LinearStateFullModel(a_predictors::AbstractArray{Symbol}, b=[0.0f0])
    a_ch = chain4Rbfun(length(a_predictors))
    LinearStateFullModel(a_ch, a_predictors, Base.union(a_predictors), [:x1], b, Flux.Recur(innerModel(reshape([100.0f0], 1, 1)), reshape([100.0f0], 1, 1)))
end

chain4Rbfun(nInVar) = Flux.Chain(
    #BatchNorm(nInVar, affine=true),
    Dense(nInVar => 15, relu),
    #GRU(5 => 5),
    #Dense(9 => 5, σ), 
    Dense(15 => 1, σ),
)

function (m::LinearStateFullModel)(x)
    res = m(x, :infer)
    return res.out
end


function (m::LinearStateFullModel)(x, s::Symbol)
    v(sym::Symbol) = Array(x([sym]))
    a_input4Chain = select_predictors(x, m.a_predictors)
    force = m.a_chain(a_input4Chain) .* v(:x1) .+ m.b
    out = m.stateupdater(force)
    return ((; out))
end

# Call @functor to allow for training the custom model
Flux.@functor LinearStateFullModel

lsm = LinearStateFullModel([:x2, :x3])
dynres3 = fit_df!(lsm, df, [:pred_syn], Flux.mse, stateful=true, shuffle=false, n_epoch=1000, batchsize=10, opt=Adam(0.01), parameters2record=[:b], patience=300)





######################################################
### Case 4: Latent variable is dynamically evolving to be represented with an RNN
### Physical part of the model is not stateful 
### multi output
######################################################
@chain df begin
    groupby(:seqID)
    @transform! :a_dyn_syn = @bycol cumsum(:x2 .- :x3)
    @transform! :obs_dyn1 = :a_dyn_syn .* :x1 + 2.0f0
    @transform! :obs_dyn2 = 0.5f0 .* :a_dyn_syn .* :x2
end

## Slope is a dynamic random walk
data(df) * mapping([:a_dyn_syn, :obs_dyn1], col=dims(1)) * visual(Lines) |> draw
## Result depends on x1 and slope a_dyn_syn
data(df) * mapping([:x1, :x2], [:obs_dyn1 :obs_dyn2], color=:a_dyn_syn, row=dims(1), col=dims(2))  |> draw

## Using stateful model 
hy_dynmod = LinearHybridModel2output([:x2, :x3], [:x1], 1, 5, nn_chain=GRU_NN, a2=[0f0], b=[0f0])
## Using stateful model - with stateful = true the df is grouped into sequences according to seqID and batched along this
## Low learning rate for didactical reasons (one can see the opt better)
dynres2 = fit_df!(hy_dynmod, df, [:obs_dyn1, :obs_dyn2], Flux.mse, stateful=true, n_epoch=500, batchsize=10, opt=Adam(0.005),
    parameters2record=[:a2, :b], latents2record=[:a => :a_dyn_syn, :pred1 => :obs_dyn1, :pred2 => :obs_dyn2], patience=50)
### Make a direct evaluation
fig3=evalfit(dynres2, df)

## or make explicit predictions including latents 
df_pred = predict_all2df(dynres2, df)
