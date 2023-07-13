using MRTools # Mostly re-exporting packages and some helper functions
using EasyHybrid
using Flux

GLMakie.activate!(; float=true)
update_theme!(fontsize=2 * 14)


################################
## Case 1: Stateless, single output
## Create some data y = a ⋅ x1 + b, where a = f(x2,x3), b=2
df = @chain DataFrame(rand(Float32, 1000, 3), :auto) begin
    @transform :a_syn = exp(-5.0f0((:x2 - 0.7f0))^2.0f0) + :x3 / 10.0f0
    @aside b = 2.0f0
    @transform :obs = :a_syn * :x1 + b + 0.1f0randn(Float32)
    @transform :seqID = @bycol repeat(1:100, inner=10)

end

## Illustrate data
fig = HDFigure()
draw!(fig[1,1:2], data(df) * mapping(:x1, :obs, color=:x2); axis=(;limits=(nothing, (0,nothing))) );
Colorbar(fig[1,3], current_axis().scene.plots[1], label="x2")
draw!(fig[1,4], data(df) * mapping(:x2, :a_syn))
fig
##

## Define model struct
begin
    ########################################
    # Model definition y = ax + b, where 
    ########################################
    struct LinearHybridModel 
        DenseLayers::Flux.Chain
        predictors::AbstractArray{Symbol}
        forcing::AbstractArray{Symbol}
        b
    end

    ## Define construction functions for a FF and a recurrent NN

    function DenseNN(in_dim, out_dim, neurons)
        return Flux.Chain(
            BatchNorm(in_dim),
            Dense(in_dim => neurons ,σ),
            Dense(neurons => out_dim),
            )
    end

    function GRU_NN(in_dim, out_dim, neurons)
        return Flux.Chain(
            #BatchNorm(in_dim),
            GRU(in_dim => neurons),
            Dense(neurons => out_dim),
            )
    end


    function LinearHybridModel(predictors, forcing, out_dim, neurons; nn_chain=DenseNN,  b=[1.5f0])
        in_dim = length(predictors)
        ch = nn_chain(in_dim, out_dim, neurons)
        LinearHybridModel(ch, predictors, forcing, b)
    end


    function (lhm::LinearHybridModel)(x, ::Val{:infer})
        v(sym::Symbol) = Array(x([sym]))
        #NNinput4a = select_predictors(x, lhm.predictors)
        a = select_predictors(x, lhm.predictors) |> lhm.DenseLayers
        pred = a .* v(:x1) .+ lhm.b
        return (; a, pred)
    end

    function (lhm::LinearHybridModel)(df)
        res =  lhm(df, :infer) #??? would be _, pred = lhm(df, :infer) "better" (I would not like it, bc it is positional)
        return res.pred
    end

    function (lhm::LinearHybridModel)(df, infer::Symbol)
        return lhm(df, Val(infer))
    end

    # Call @functor to allow for training the custom model
    Flux.@functor LinearHybridModel
end

## Instantiate model
hymod = LinearHybridModel([:x2, :x3], [:x1], 1, 5, b=[0.0f0])

# Fit the model with the non-stateful function fit_df! to the first half of the data set
# One does not need to put predictors explicitly, if they are explicit in the model
res = fit_df!(hymod, df[1:500,:], [:obs], Flux.mse, n_epoch=500, batchsize=100, opt=Adam(0.01), parameters2record=[:b],
    latents2record=[:pred => :obs, :a => "a_syn"], patience=300, stateful=false);
##
test_df = df[501:1000, :]

### Make a direct evaluation
fig2=evalfit(res, test_df)

## or make explicit predictions including latents for 2nd half of the dataset (aka test set)
test_df_pred = predict_all2df(res.bestModel,test_df)

# ... and plot variation of latent a_syn in Figure above
scatter!(fig.content[3], test_df_pred.x2,test_df_pred.a_syn, color=(:red,0.5))
fig


################################################
#### Excursion

## One can use fit_df! also for fitting a simple (non-hybrid) NN)
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
#### Lower level access to model via KeyedArray
dk = tokeyedArray(df) 
hymod(dk)

#  Compute some loss function ...
Flux.mse(hymod(dk), dk([:obs])) # Mean sq error
Flux.mae(hymod(dk), dk([:obs])) # Mean abs error
##


