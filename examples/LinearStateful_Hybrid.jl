######################################################
### Case 2: Latent variable is dynamically evolving to be represented with an RNN
### Physical part of the model is not stateful 
######################################################


df = @chain DataFrame(rand(Float32, 1000, 3), :auto) begin
    @transform :seqID = @bycol repeat(1:100, inner=10)
    groupby(:seqID)
    @transform :a_dyn_syn = @bycol cumsum(:x2 .- :x3)
    @transform :obs_dyn = :a_dyn_syn .* :x1 + 2.0f0 +  0.1f0randn(Float32)
end

## Slope is a dynamic random walk
data(df) * mapping([:a_dyn_syn, :obs_dyn], col=dims(1)) * visual(Lines) |> draw
## Result depends on x1 and slope a_dyn_syn
data(df) * mapping(:x1, :obs_dyn, color=:a_dyn_syn)  |> draw

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


## Using stateless model of course doesn't really work (but manage as expected to predict the first point)
hymod = LinearHybridModel([:x2, :x3], [:x1], 1, 5, nn_chain=DenseNN, b=[0.0f0])
dynres2 = fit_df!(hymod, df, [:obs_dyn], Flux.mse, stateful=false, n_epoch=100, batchsize=10, opt=Adam(0.01), parameters2record=[:b], patience=50)

# Hence, define dynamic model, where a is predicted with a GRU
hy_dynmod = LinearHybridModel([:x2, :x3], [:x1], 1, 5, nn_chain=GRU_NN, b=[0.0f0])
## Using stateful model - with stateful = true the df is grouped into sequences according to seqID and batched along this
dynres2 = fit_df!(hy_dynmod, df, [:obs_dyn], Flux.mse, stateful=true, n_epoch=500, batchsize=10, opt=Adam(0.05),
    parameters2record=[:b], latents2record=[:a => :a_dyn_syn, :pred => :obs_dyn], patience=50)

### Make a direct evaluation
fig3=evalfit(dynres2, df)

## or make explicit predictions including latents 
df_pred = predict_all2df(dynres2, df)


