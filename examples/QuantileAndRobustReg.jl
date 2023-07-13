using MRTools # Mostly re-exporting packages and some helper functions
using EasyHybrid
using Flux, Random

GLMakie.activate!(; float=true)
update_theme!(fontsize=2 * 14)


################################
## Example for Quantile estimation via tilted abs cost function
## 
df = @chain DataFrame(rand(Float32, 1000, 3), :auto) begin
    #@sort :x1
    @transform :a_syn = exp(-10.0f0((:x1 - 0.5f0))^2.0f0)
    #@transform :a_syn = sqrt(10f0*:x1 + 0.1f0)
    @aside b = 2.0f0
    @transform :e_obs = :a_syn * sin(8π * :x1) |> Float32
    @transform :obs = :e_obs + 0.3f0randn(Float32)
end

## Illustrate data
fig = HDFigure()
obsLay = mapping(:x1, :obs) 
e_obsLay = mapping(:x1, :e_obs) * visual(Lines; linewidth=3)
latentLay = mapping(:x1, :a_syn) * visual(Lines; linewidth=3, color=:red)
draw!(fig[1,1:2], data(@sort df :x1) * (obsLay + e_obsLay + latentLay));
#Colorbar(fig[1,3], current_axis().scene.plots[1], label="x2")
#draw!(fig[1,4], data(df) * mapping(:x2, :a_syn))
fig
##

## Define model struct
begin
    ########################################
    # Model definition y = ax + b, where 
    ########################################
    struct SinusHybridModel 
        DenseLayers::Flux.Chain
        predictors::AbstractArray{Symbol}
        forcing::AbstractArray{Symbol}
        b
    end

    ## Define construction functions for a FF and a recurrent NN

    function DenseNN(in_dim, out_dim, neurons)
        return Flux.Chain(
            BatchNorm(in_dim),
            Dense(in_dim => neurons, softplus),
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


    function SinusHybridModel(predictors, forcing, out_dim, neurons; nn_chain=DenseNN,  b=[1.5f0])
        in_dim = length(predictors)
        ch = nn_chain(in_dim, out_dim, neurons)
        SinusHybridModel(ch, predictors, forcing, b)
    end


    function (lhm::SinusHybridModel)(x, ::Val{:infer})
        v(sym::Symbol) = Array(x([sym]))
        #NNinput4a = select_predictors(x, lhm.predictors)
        a = select_predictors(x, lhm.predictors) |> lhm.DenseLayers
        pred =  a  .* sin.(#=lhm.b .*=# Float32(8f0π)*v(:x1)) .+ lhm.b
        return (; a, pred)
    end

    function (lhm::SinusHybridModel)(df)
        res =  lhm(df, :infer) #??? would be _, pred = lhm(df, :infer) "better" (I would not like it, bc it is positional)
        return res.pred
    end

    function (lhm::SinusHybridModel)(df, infer::Symbol)
        return lhm(df, Val(infer))
    end

    # Call @functor to allow for training the custom model
    Flux.@functor SinusHybridModel
end

## Instantiate model
hymod = SinusHybridModel([:x1], [:x1], 1, 15, b=[0.0f0])

# Fit the model with the non-stateful function fit_df! to the first half of the data set
# One does not need to put predictors explicitly, if they are explicit in the model
res = map(prob -> fit_df!(hymod, df[1:500,:], [:obs], qlossfun(prob), n_epoch=500, batchsize=100, opt=Adam(0.01), parameters2record=[:b],
    latents2record=[:pred => :obs, :a => "a_syn"], patience=300, stateful=false, rseed=1), [0.05, 0.5, 0.95])
##
test_df = df[501:1000, :]


## or make explicit predictions including latents for 2nd half of the dataset (aka test set)
test_df_pred = mapreduce(hcat,res, [5, 50, 95]) do r, q
    df = predict_all2df(r.bestModel,test_df)
    r !== res[1] && @select!(df, :pred)
    rename!(df, "pred" => "pred$q")
    return df
end

qplot(d, vars, vis; kw...) = data(d) * mapping(vars...) * visual(vis; kw...)

p = qplot(test_df_pred, (:x1, :obs), Scatter) + 
    mapreduce(+, "pred" .* string.([5,50,95]), [:dash, :solid, :dash]) do vari, linestyle
        qplot(@sort(test_df_pred, :x1), (:x1, vari), Lines; color=:red, linewidth=5, linestyle)      
    end
fig = HDFigure()
draw!(fig, p)
display(fig)

#### Lower level access to model via KeyedArray
dk = tokeyedArray(df) 
hymod(dk)

#  Compute some loss function ...
Flux.mse(hymod(dk), dk([:obs])) # Mean sq error
Flux.mae(hymod(dk), dk([:obs])) # Mean abs error
qlossfun(0.95)(hymod(dk), dk([:obs]))
qlossfun(0.5)(hymod(dk), dk([:obs]))
qlossfun(0.05)(hymod(dk), dk([:obs]))
##


## Add contamination
idx = sample(findall(x->x>0.8, df.x1), Int(0.3 * 200), replace=false) # 30% contamination
n = length(idx)
x_bad = df.x1[idx] .+ 0.01f0randn(n) .|> Float32
y_bad = df.e_obs[idx] .* 50.0 .+ 3.0randn(n) .|> Float32
scatter([df.x1;x_bad], [df.obs;y_bad]) |> display
df_bad = DataFrame(; x1=x_bad, obs=y_bad, a_syn=df.a_syn[idx])

df_bad = vcat(df, df_bad, cols=:union) |> shuffle
##
hymod = SinusHybridModel([:x1], [:x1], 1, 15, b=[0.0f0])
res= fit_df!(hymod, df_bad[1:500,:], [:obs], Flux.mae, n_epoch=500, batchsize=100, opt=Adam(0.01), parameters2record=[:b],
    latents2record=[:pred => :obs, :a => "a_syn"], patience=300, stateful=false, rseed=1)
##
hymod = SinusHybridModel([:x1], [:x1], 1, 15, b=[0.0f0])
res_rob= fit_df!(hymod, df_bad[1:500,:], [:obs], trimmedLossfun(0.9), n_epoch=500, batchsize=100, opt=Adam(0.01), parameters2record=[:b],
    latents2record=[:pred => :obs, :a => "a_syn"], patience=300, stateful=false, rseed=1)
##
test_df =  @chain df_bad[501:end, :] begin
    sort(:x1)
    hcat(predict_all2df(res.bestModel,_), predict_all2df(res_rob.bestModel,_), makeunique=true)
    @transform :e_obs = coalesce(:e_obs, NaN32)
end
p = qplot(test_df, (:x1, :obs), Scatter) + qplot(test_df, (:x1, :pred), Lines; color=:red, linewidth=5) +  
        qplot(test_df, (:x1, :pred_1), Lines; color=:blue, linewidth=5) + 
        qplot(test_df, (:x1, :e_obs), Lines; color=:black, linewidth=15, linestyle=:dash) +   
        qplot(test_df, (:x1, :a_syn), Lines; color=:black, linewidth=5, linestyle=:dash)  +  
        qplot(test_df, (:x1, :a), Lines; color=(:red, 0.5), linewidth=5, linestyle=:dash)  +  
        qplot(test_df, (:x1, :a_1), Lines; color=(:blue, 0.5), linewidth=5, linestyle=:dash)    
fig = HDFigure()
draw!(fig, p)
display(fig)
ylims!(-3,3)
##

test_df_pred = predict_all2df(res_rob.bestModel,test_df)

lines!(current_axis(), test_df_pred.x1,test_df_pred.pred )

p = qplot(test_df_pred, (:x1, :obs), Scatter) + qplot(@sort(test_df_pred, :x1), (:x1, :pred), Lines; color=:red, linewidth=5)      
fig = HDFigure()
draw!(fig, p)
display(fig)
ylims!(-2,2)
