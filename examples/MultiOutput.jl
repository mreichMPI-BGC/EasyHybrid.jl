######################################################
### Case 4: Latent variable is dynamically evolving to be represented with an RNN
### Physical part of the model is not stateful 
### multi output
######################################################
df = @chain DataFrame(rand(Float32, 1000, 3), :auto) begin
    @transform :seqID = @bycol repeat(1:100, inner=10)
    groupby(:seqID)
    @transform :a_dyn_syn = @bycol cumsum(:x2 .- :x3)
    @transform :obs_dyn1 = :a_dyn_syn .* :x1 + 2.0f0
    @transform :obs_dyn2 = 0.5f0 .* :a_dyn_syn .* :x2
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
