# EasyHybrid

[![Build Status](https://github.com/mreichMPI-BGC/EasyHybrid.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mreichMPI-BGC/EasyHybrid.jl/actions/workflows/CI.yml?query=branch%3Amain)

## *|> |> |> Work in progress |> |> |> Work in progress |> |> |> Work in progress |> |> |> Work in progress |> |> |>*


## Hybrid modelling for teaching purposes

The idea of this repo to provide a relatively simple approach for hybrid modelling, i.e. creating a model which combines machine learning with domain scientific modelling. In a general sense $y = g(f(x), z, \theta)$ where $g$ is a parametric function with parameters $\theta$ to be learned, and $f$ is non-parametric and to be learned. Here $f$ is represented with a neural net in Flux.jl.  

## Quick intro example (more in the ./example/ folder)
```
using EasyHybrid
```
### Case 1: Stateless, single output
### Create some data $y = a \cdot x_1 + b$, where $a = f(x_2,x_3), b=2$
```
using DataFrames
df = @chain DataFrame(rand(Float32, 1000, 3), :auto) begin
    @transform :a_syn = exp(-5.0f0((:x2 - 0.7f0))^2.0f0) + :x3 / 10.0f0
    @aside b = 2.0f0
    @transform :obs = :a_syn * :x1 + b
    @transform :pred_syn = :obs
    @transform :seqID = @bycol repeat(1:100, inner=10)

end
```

## Illustrate data
```
using AlgebraOfGraphics
fig = Figure()
draw!(fig[1,1:2], data(df) * mapping(:x1, :obs, color=:x2); axis=(;limits=(nothing, (0,nothing))) )
Colorbar(fig[1,3], current_axis().scene.plots[1], label="x2")
draw!(fig[1,4], data(df) * mapping(:x2, :a_syn))
fig
```

### Instantiate model
``` hymod = LinearHybridModel([:x2, :x3], [:x1], 1, 5, b=[0.0f0])```

### Fit the model with the non-stateful function fit_df! to the first half of the data set
### One does not need to put predictors explicitly, if they are explicit in the model
```
res = fit_df!(hymod, df[1:500,:], [:obs], Flux.mse, n_epoch=500, batchsize=100, opt=Adam(0.01), parameters2record=[:b],
    latents2record=[:pred => :obs, :a => "a_syn"], patience=300, stateful=false)
```

### Make a direct evaluation
```
test_df = df[501:1000, :]
fig2=evalfit(res, test_df)
```

### or make explicit predictions including latents for 2nd half of the dataset (aka test set)
```
test_df_pred = predict_all2df(res.bestModel,test_df)
```
### ... and plot variation of latent a_syn in Figure above
```
scatter!(fig.content[3], test_df_pred.x2,test_df_pred.a_syn, color=(:red,0.5))
```


## *|> |> |> Work in progress |> |> |> Work in progress |> |> |> Work in progress |> |> |> Work in progress |> |> |>*
