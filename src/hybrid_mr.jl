using MRTools
using Flux, AxisKeys
#using AlgebraOfGraphics, GLMakie
using ProgressMeter
using MLJ: partition
import Random
import Chain: @chain
#using DataFrames, SplitApplyCombine

export fit_df!, predict_all2df, select_predictors, evalfit

#include("../JMR_Base/MRplot.jl")
##

#### Data handling

# Make vec each entry of NamedTuple (since broadcast ist reserved)
evec(nt::NamedTuple) = map(vec, nt)

# Start from Dataframe, select variables and make a Flux-compatible tensor
function select_predictors(df, predictors)
    return select(df, predictors) |> Matrix |> transpose
end
# Start from KeyedArray, selct variables and make a Flux-compatible tensor
function select_predictors(dk::KeyedArray, predictors)
    return dk(predictors) |> Array
end

# Convert a DataFrame to a Keyedarray where variables are in 1st dim (rows)
function tokeyedArray(df::DataFrame)
    d = Matrix(df) |> transpose
    return KeyedArray(d, row=Symbol.(names(df)), col =1:size(d,2))
end

# Cast a grouped dataframe into a KeyedArray, where the group is the third dimension
# Only one group dimension is currently considered 
function tokeyedArray(dfg::Union{Vector, GroupedDataFrame{DataFrame}}, vars=All())
    dkg = [select(df, vars) |> tokeyedArray for df in dfg]
    dkg = reduce((x,y)->cat(x,y, dims=3), dkg)
    newKeyNames=(AxisKeys.dimnames(dkg)[1:2]..., dfg.cols[1])
    newKeys=(axiskeys(dkg)[1:2]..., unique(dfg.groups) )
    return  (wrapdims(dkg |> Array; (; zip(newKeyNames, newKeys)...)...))
end

# Create dataloaders for training and validation
# Splits a normal dataframe into train/val and creates minibatches of x and y,
# where x is a KeyedArray and y a normal one (need to recheck why KeyedArray did not work with Zygote)
function split_data(df::DataFrame, target, xvars; f = 0.8, batchsize=32, shuffle=true, partial=true)
    d_train, d_vali = partition(df, f; shuffle)
    # wrap training data into Flux.DataLoader
    println(xvars)
    x = select(d_train, xvars) |> tokeyedArray
    y = select(d_train, target) |> Matrix |> transpose |> collect # tokeyedArray does not work bc of Zygote
    data_t = (; x, y)
    #println(size(y), size(data_t.x))
    trainloader = Flux.DataLoader(data_t; batchsize, shuffle, partial) # batches for training
    trainall = Flux.DataLoader(data_t; batchsize=size(y, 2), shuffle, partial) # whole training set for plotting
    # wrap validation data into Flux.DataLoader
    x = select(d_vali, xvars) |> tokeyedArray
    y = select(d_vali, target) |> Matrix |> transpose |> collect
    data_v = (; x, y)
    valloader = Flux.DataLoader(data_v; batchsize = size(y, 2), shuffle=false, partial=false) # whole validation for early stopping and plotting
    return trainloader, valloader, trainall
end

# As above but uses a seqID to keep same seqIDs in the same batch
# For instance needed for recurrent modelling
# Creates tensors with a third dimension, i.e. size is (nvar, seqLen, batchsize)
# Which is unfortunate since Recur in Flux wants sequence as last/3rd dimension
function split_data(df::DataFrame, target, xvars, seqID; f = 0.8, batchsize=32, shuffle=true, partial=true)
    dfg =  groupby(df, seqID)
    dkg = tokeyedArray(dfg)
    #@show axiskeys(dkg)[1]
    # Do the partitioning via indices of the 3rd dimension (e.g. seqID) because
    # partition does not allow partitioning along that dimension (or even not arrays at all)
    idx_tr, idx_vali = partition(axiskeys(dkg)[3], f; shuffle)
    # wrap training data into Flux.DataLoader
    x = dkg(row = xvars, seqID=idx_tr)
    y = dkg(row=target, seqID=idx_tr) |> Array 
    data_t = (; x, y)
    trainloader = Flux.DataLoader(data_t; batchsize, shuffle, partial)
    trainall = Flux.DataLoader(data_t; batchsize=size(x,3), shuffle=false, partial=false)
    # wrap validation data into Flux.DataLoader
    x = dkg(row = xvars, seqID=idx_vali)
    y = dkg(row = target, seqID=idx_vali) |> Array 
    data_v = (; x, y)
    valloader = Flux.DataLoader(data_v; batchsize=size(x,3), shuffle=false, partial=false)
    return trainloader, valloader, trainall
end

# Create dashboard to follow the training
# Losses, trace of parameters and current scatterplots are shown
function train_board(train_loss, vali_loss,  
    vali2scat, valicolors, train2scat, traincolors, param_trace=nothing; model="modelled") ## still need to change the "nothing" issue

    !isnothing(param_trace) && length(param_trace[]) > 2 && println("WARNING: only first two parameter traces plotted!")
    # Define figure und panels (=axes)
    fig=Figure(resolution=(1600,900))
    scat1 = GLMakie.Axis(fig[1, 1:2], title="Training -- y: observed, x: $(model)")
    scat2 = GLMakie.Axis(fig[1, 3:4], title="Validation -- y: observed, x: $(model)")
    line = GLMakie.Axis(fig[2, 1:2], ylabel="log(Loss)")
    lineZoom = GLMakie.Axis(fig[2,3:4], ylabel="log(0.01 + loss - min(losses))")
    if length(param_trace[])>0 lineTrace = GLMakie.Axis(fig[1,5], ylabel=string(param_trace[][1].first)) end
    if length(param_trace[])>1 lineTrace2 = GLMakie.Axis(fig[2,5], ylabel=string(param_trace[][2].first)) end

    foreach(autolimits!, [line]) # Don't understand why it works already with one panel, then for all not needed
    # Train and vali_loss traces (I thought I had log10 it, maybe later)
    lines!(line, train_loss, color=:black, label="Train loss")
    lines!(line, vali_loss, color=:red, label = "Vali loss")
    axislegend(line, position=:rt)
    
    # Zoom into last 200 steps and align train and vali for max visibility
    zoomedLosses = map(train_loss, vali_loss) do tl, vl # Map does similar to lift! (I like it)
        tlVals=invert(tl)[2]
        vlVals=invert(vl)[2]
         len=length(tl)
         zoomlen=min(len-1, 200)
         zoomidx = len-zoomlen:len
         return (
            train = Point2f.(zoomidx,  log10.(1e-2 .+ tlVals[zoomidx] .- minimum(tlVals[zoomidx]))), 
            vali = Point2f.(zoomidx,  log10.(1e-2 .+ vlVals[zoomidx] .- minimum(vlVals[zoomidx]))), 
         )
    end

     lines!(lineZoom, @lift($zoomedLosses.train), color=:black)
     lines!(lineZoom, @lift($zoomedLosses.vali), color=:red)

    # Plot trace of parameters
    if length(param_trace[])>0 lines!(lineTrace, #=@lift(1:length($param_trace[1].second)),=# @lift($param_trace[1].second)) end
    if length(param_trace[])>1 lines!(lineTrace2, #=@lift(1:length($param_trace[1].second)),=# @lift($param_trace[2].second)) end
    
    # Plot scatter plot obs versus mod with 1:1 line
     scatter!(scat1,train2scat, color=traincolors)
     ablines!(scat1, 0, 1, color=:red, linewidth=2, linestyle=:dash )
     scatter!(scat2, vali2scat, color=valicolors)
     ablines!(scat2, 0, 1, color=:red, linewidth=2, linestyle=:dash )
    fig
end

# Create dashboard to follow trace of latent variables during training
# only makes sense when validation for those is there
# Both parameters are Vectors of Pairs (mod => obs), Observables thereof
function latentboard(data, names)

    fig = Figure(resolution=(1600, 900))
    # axis = GLMakie.Axis(fig[1, 1], 
    # xlabel=names[][1].first |> string, 
    # ylabel=names[][1].second |> string)
    axes = []
    # Cycle through all pairs and arrange plot in 3 colums
    for (i, _) in enumerate(data[])
        col = (i - 1) % 3 + 1
        row = (i - 1) ÷ 3 + 1
        push!(axes, GLMakie.Axis(fig[row, col],
            xlabel=@lift($names[i].first |> string), # unneccsary to lift names since they don't change...
            ylabel=@lift($names[i].second |> string))
        )
        scatter!(axes[end], @lift($data[i].first |> vec), @lift($data[i].second |> vec), color=(:black, 0.4))
        ablines!(0, 1, color=:red, linestyle=:dash)
    end
    fig
end

# Function to run a (stateful) model sequentially 
# the seqeunce is along dim 2 ! (because in Flux last dim is obs dim, first dim is variable)
# !! need to do it also for the :infer mode with latents (need to give back a Keyedarray or so) !!
function run2!(m, x)
    # x is Array(var, seqElement, seqID)
    #print("Running...")
    Flux.reset!(m)
    #println(x |> size)
    res = [m(x[:,i,:]) for i in axes(x,2)]
    #print(size(res))
    res = @. reshape(res, size(res,1),1, size(res, 2) ) # make sure that res also has [nout, 1, batchsize] size
    return reduce(hcat, res) # concat along dim 2 also works for arrays
end

function run!(m, x)
    # x is Array(var, seqElement, seqID)
    #print("Running...")
    Flux.reset!(m)
    #println(size(x))
    res = m(permutedims(x, (1,3,2))) # [m(x[:,i,:]) for i in axes(x,2)]
    #print(size(res))
    #res = @. reshape(res, size(res,1),1, size(res, 2) ) # make sure that res also has [nout, 1, batchsize] size
    return permutedims(res, (1,3,2)) # concat along dim 2 also works for arrays
end

function run!(m, x, s::Symbol)
    # x is Array(var, seqElement, seqID)
    #print("Running...")
    perm32(x) = permutedims(x, (1,3,2))
    Flux.reset!(m)
    res = m(perm32(x), :infer)
    return map(perm32, res)
    #res = [m(x[:,i,:], :infer) for i in axes(x,2)]
    #res = @. reshape(res, size(res,1),1, size(res, 2) ) # make sure that res also has [nout, 1, batchsize] size
    #return reduce(hcat, res) # concat along dim 2 also works for arrays
end

"""
fit_df!(data, predictors, target, lossfun; modeltype=nothing, model=nothing, lr=0.001, opt=Adam(lr), opt_state=Flux.setup(opt, model),
n_epoch=1000, patience=100, batchsize=30, latents2record=nothing, printevery=10, plotevery=50)

Fit a semi-parametric model to tabular data using Flux.jl.

Argument:
- `model`: the model to fit
- `data`: a dataframe of input data
- `predictors`: an array of column names to use as predictors in the model.
- `forcing:  column names which are "physical" forcing of the model
- `target`: the name of the column to use as the target variable.
- `lossfun`: the basic loss function to optimize: this is a function supposed to take two arrays of same dim (like Flux.mse)

Optional keyword arguments:
- `modeltype`: a function that creates an untrained model, which is passed to the `model` argument if provided.
- `model`: a pre-trained model to use. If not provided, `modeltype` is used to create one.
- `lr`: the learning rate to use for the optimizer.
- `opt`: the optimizer to use.
- `opt_state`: the initial state of the optimizer.
- `n_epoch`: the number of epochs to train for.
- `patience`: the number of epochs to wait for improvement on the validation set before stopping training.
- `batchsize`: the size of each mini-batch during training.
- `latents2record`: an optional dictionary of model parameters to record during training.
- `printevery`: how often to print out loss values during training.
- `plotevery`: how often to plot the loss curves and other diagnostics during training.

Returns a named tuple containing the training and validation losses, the best model found during training, the final optimizer state, and any recorded model parameters.
"""
# Multi dispatch: if forcing is not provided, it is taken from model.forcing
# better would be to provide a mapping from dataset to model variables (should be easy)
function fit_df!(model, data, predictors, target::Vector{Symbol}, lossfun; kwargs...)
    fit_df!(model, data, predictors, model.forcing, target, lossfun; 
    kwargs...) 
end

# This one is to work with normal Chain, not hybrid ==> no forcing
# Forcing is [] then, fortunately []... is ok in array construction (while nothing... is not)
# could make it Symbol[]
function fit_df!(model::Flux.Chain, data, predictors, target::Vector{Symbol}, lossfun; kwargs...)
    println("Chain")
    fit_df!(model, data, predictors, [], target, lossfun; 
    kwargs...) 
end

# Neither preditors not forcing given: both taken from model
# Model.predictors can be seen at default
function fit_df!(model, data, target::Vector{Symbol}, lossfun; kwargs...)
    fit_df!(model, data, model.predictors, model.forcing, target, lossfun; 
    kwargs...) 
end

## Finally the full function
"""
    fit_df!(model, data, predictors, forcing, target, lossfun, renamer;
        lossfun2 = m->0f0, lr = 0.001, opt = Adam(lr), opt_state = Flux.setup(opt, model),
        stateful = true, n_epoch = 1000, patience = 100, batchsize = 30, parameters2record = nothing,
        latents2record = [[] => []], printevery = 10, plotevery = 50, rseed = 123,
        shuffle = true, showboards = true, showprogress = true)

Fit a model to data.

# Arguments
- `model`: a Flux model to be trained.
- `data`: a `DataFrame` containing the data to be used for training.
- `predictors`: predictor feature(s) as `Symbol`s.
- `forcing`: forcing feature(s) as `Symbol`s.
- `target`: target variable as a `Vector` of `Symbol`(s).
- `lossfun`: loss function to use during training.
- `renamer`: a Vector of Pairs used to rename columns in dataframes to ensure they match the model.
- `lossfun2`: a part of the loss function only dependent on the model (for regularization).
- `lr`: learning rate.
- `opt`: optimizer to use.
- `opt_state`: initial optimizer state.
- `stateful`: whether or not the model is stateful
- `n_epoch`: number of epochs to train the model.
- `patience`: number of epochs to wait before ending early.
- `batchsize`: mini-batch size.
- `parameters2record`: specific model parameters to record during training.
- `latents2record`: specific latents to record during training.
- `printevery`: number of epochs between print statements.
- `plotevery`: number of epochs between data visualizations.
- `rseed`: random seed to use during training.
- `shuffle`: whether or not to shuffle the data between epochs.
- `showboards`: whether or not to plot board output.
- `showprogress`: whether or not to print progress bar output.

# Returns
- a NamedTuple including the best Model after training with the updated weights.
"""
function fit_df!(model, data, predictors, forcing, target::Vector{Symbol}, lossfun;
    renamer = map(f -> f => f, forcing),
    lossfun2=m->0f0, # weighted L1 penaltly would be 0.01f0 * sum(abs,  Flux.destructure(m.Rb_chain)[1])
    lr=0.001, opt=Adam(lr), opt_state=Flux.setup(opt, model), stateful=true,
    n_epoch=1000, patience=100, batchsize=30, parameters2record=nothing,  latents2record=[[] => []],
    printevery=10, plotevery=50,
    rseed=123, shuffle = true, showboards=true, showprogress=true) 

    Random.seed!(rseed)
    # Define the stateful and stateless loss function instances
    # based on the given general lossfun
    # i.e. the two loss functions only differ in how the model is run
    # [since now they only differ but Flux.reset! this might be simplified]
    statefulloss(m,d) = lossfun(run!(m, d[1]), d[2]) + lossfun2(m)
    statelessloss(m,d) = lossfun(m(d[1]), d[2])  + lossfun2(m) #+ 0.01f0 * sum(abs,  Flux.destructure(m.Rb_chain)[1])#+ sum(x->sum(abs, x),Flux.params(m.Rb_chain))

    # Define variables which are extracted from df
    vbls=Base.union(predictors, forcing, last.(latents2record))
    vbls = filter(x->isa(x, Union{Symbol, String}), vbls) .|> Symbol ## filter out Any[] and convert to Symbol if needed
    @info "Predictors for NN: $predictors"
    @info "Forcing for system model: $forcing"
    @info "Target variable(s): $target"



    # Define dataloaders for stateful (nvar, seqLen, batchsize) or stateless (nvar,batchsize)
    # uniloss = unified loss
    # !!! seqID as identifier of sequence still hardcoded!!
    if stateful 
        println("Using stateful approach...")
        # Partition data into training and validation sets
        trainloader, valiloader, trainall = split_data(DataFrames.rename(data, renamer...), target, vbls, :seqID; batchsize, shuffle = shuffle)
        uniloss = statefulloss
    else
        println("Using stateless approach...")
        trainloader, valiloader, trainall = split_data(DataFrames.rename(data, renamer...), target, vbls; batchsize, shuffle = shuffle)
        uniloss = statelessloss
    end

    @info "Dims of one batch: $(map(size,trainloader|>first))"

    ## These "loaders" contain simply the full dataset, i.e. first takes all
    valiloader = valiloader |> first
    trainall = trainall |> first

    # Save a copy of the initial model and initialize loss values to memorize what the best model is
    bestModel=Base.deepcopy(model)
    vali_losses=[uniloss(model, valiloader)]
    best_vali_loss = vali_losses[1]
    train_losses=[uniloss(model, trainall)]

    patience_cnt = 0 


    # Record parameters if specified
    # trace_param is a Vector of Pairs Parameter => trace of values
    if isnothing(parameters2record) 
        trace_param=[]
    else 
        trace_param = map(parameters2record) do x x => copy(getfield(model, x)) end
    end

    if showboards
         #Initialize plotting windows
        GLMakie.set_window_config!(float=true)
        trainScreen=GLMakie.Screen()
        latentScreen=GLMakie.Screen()
    end


        # Define entities to plot on trainboard as Points2f Vectors wrapped as Observable
        train_loss2plot = [Point2f(0, train_losses[1] |> log10)] |> Observable
        vali_loss2plot = [Point2f(0, vali_losses[1] |> log10)] |> Observable
        trace_param2plot = trace_param |> Observable
        vali2plot=Point2f.(rand(100), rand(100)) |> Observable # just too lazy to run model
        valicolors = fill((:blue,0.4), 100) |> Observable
        train2plot=Point2f.(rand(100), rand(100)) |> Observable
        traincolors = fill((:blue,0.4), 100) |> Observable

        fig1 = train_board(train_loss2plot, vali_loss2plot, 
            vali2plot, valicolors,  train2plot, traincolors,  trace_param2plot)

        # Define entities to plot on latentboard

        # Run model with latent output
        fig2=nothing
        if latents2record[1].first != [] 
            pred = stateful ? predict_all(model, valiloader.x, Val(:stateful)) : predict_all(model, valiloader.x)
            pred = evec(pred)
            # Extract latent variable from model output (namedtuple) and observation (KeyedArray)
            # latents2record was a Vector of Pairs encoding Model and observed 
            alllatentdata = map(latents2record) do ll
                pred[ll.first |> Symbol] => valiloader.x(row = ll.second |> Symbol)
            end |> Observable
            alllatentnames = latents2record |> Observable
            fig2=latentboard(alllatentdata, alllatentnames)
        end
    #else # if showboards
    #    fig1 = fig2 = nothing
    #end

    if showprogress
        p = Progress(n_epoch, barglyphs=BarGlyphs("[=> ]"), color = :yellow)
    end

    # Train the model for n_epoch epochs
    for e in 1:n_epoch
        #print(first(trainloader)[2] |> size)
        for d in trainloader
            ∂L∂m = Flux.gradient(uniloss, model, d)[1] 
            Flux.update!(opt_state, model, ∂L∂m)
        end

        # Record loss values and latent variables (if specified) after each epoch      
        push!(vali_losses, uniloss(model, valiloader))
        push!(train_losses, uniloss(model, trainall))

        if !isnothing(parameters2record) 
            # Push each current model parameter into trace vector 
            foreach(trace_param) do x push!(x.second, copy(getfield(model, x.first)[1])) end 
        end
        
        # Check if model has improved on validation and update bestmodel, and count patience
        if last(vali_losses) < best_vali_loss
            patience_cnt = 0
            best_vali_loss = last(vali_losses)
            bestModel=Base.deepcopy(model)
        else
            patience_cnt += 1
        end

        #if showboards
            # Record plotting variables for trainboard
            push!(vali_loss2plot[], Point2f(e,vali_losses[end] |> log10))
            push!(train_loss2plot[], Point2f(e, train_losses[end] |> log10))
            trace_param2plot[]=trace_param

            predTrain = stateful ? run!(model, trainall.x) : model(trainall.x)
            predVali = stateful ? run!(model, valiloader.x) : model(valiloader.x)
            obstrain = trainall.y[:] 
            obsvali = valiloader.y[:] 
        
            # Define scatterplot data training set
            len = length(predTrain)
            plotidx = range(1, len, step=1 + len ÷ 2000) # Choose step so that max ~2000 points are plotted
            train2plot[]=Point2f.(predTrain[plotidx], obstrain[plotidx])
            varinum=CartesianIndices(predTrain)[plotidx] .|> Tuple .|> first
            traincolors[] = map(x->(x, 0.4), [:blue, :green, :red, :brown][varinum])
            # Define scatterplot data validation set
            len = length(predVali)
            plotidx = range(1, len, step=1 + len ÷ 2000) 
            vali2plot[]=Point2f.(predVali[plotidx], obsvali[plotidx])
            varinum=CartesianIndices(predVali)[plotidx] .|> Tuple .|> first
            valicolors[] = map(x->(x, 0.4), [:blue, :green, :red, :brown][varinum])

            # Notify Observables that very only pushed (not newly assigned)
            notify(vali_loss2plot); notify(train_loss2plot); notify(trace_param2plot)

            # Update the latent variable plot
            if latents2record[1].first != [] 

                pred = stateful ? predict_all(model, valiloader.x, Val(:stateful)) : predict_all(model, valiloader.x)
                pred = evec(pred)
                # Extract latent variable from model output (namedtuple) and observation (KeyedArray)
                # latents2record was a Vector of Pairs encoding Model and observed 
                alllatentdata[] = map(latents2record) do ll
                    pred[ll.first |> Symbol] => valiloader.x(row = ll.second |> Symbol)
                end 
                showboards && display(latentScreen, fig2)
            end
        showboards && display(trainScreen, fig1)

        #end


        showprogress && next!(p; showvalues = [
            (:epoch, e),
            (:Patience, patience_cnt), 
            #(:Parameters, length(trace_param) > 0 ? trace_param[1].second[end] : " none"),
            (:Parameters, length(trace_param) > 0 ? map(f->f => getproperty(model, f)[1], parameters2record) |> NamedTuple : " none"),
            (:Best_parameters, length(trace_param) > 0 ? map(f->f => getproperty(bestModel, f)[1], parameters2record) |> NamedTuple : " none"),
            (:validation, (Current = vali_losses[end], Best = best_vali_loss)),
            (:training, train_losses[end])
            ]
        )

        patience_cnt > patience && break

        # Print and plot progress at specified intervals
        # not needed anymore because Observables update is so fast?
       
    end #epoch

    # Return named tuple storing losses, bestmodel, state of optimizer and the trace of the (physical) parameters
    return (; train_losses, vali_losses, bestModel, stateful, opt_state, trace_param, latents2record, trainScreen=fig1, latentScreen=fig2)
end
##

## Prediction for stateless model
predict_all(model, data::KeyedArray, ::Val{:stateless}) = model(data, :infer)
predict_all(model, data::KeyedArray) = predict_all(model, data::KeyedArray, Val(:stateless))
predict_all(model, data::DataFrame,  ::Val{:stateless}) = predict_all(model, data)
predict_all(model, data::DataFrame) = predict_all(model, @chain data select(_, names(_, Number))  tokeyedArray)
pred2df(pred) =  map(vec, pred) |> DataFrame

function predict_all2df(model, data::DataFrame,::Val{:stateless})
    pred = predict_all(model, data)
    hcat(data, pred |> pred2df, makeunique=true)
end
predict_all2df(model, data::DataFrame) = predict_all2df(model, data::DataFrame, Val(:stateless))
predict_all2df(fit::NamedTuple, data::DataFrame, ::Val{:stateless}) = predict_all2df(fit.bestModel, data, Val(:stateless))
predict_all2df(fit::NamedTuple, data::DataFrame) = predict_all2df(fit.bestModel, data, fitmode(fit))

## Prediction for stateful model: main difference: run! function employed to reset model and permutedims
predict_all(model, data::KeyedArray, ::Val{:stateful}) = run!(model, data, :infer)
predict_all(model, data::DataFrame, ::Val{:stateful}) = predict_all(model, groupby(data, :seqID) |> tokeyedArray, Val(:stateful))
function predict_all2df(model, data::DataFrame, ::Val{:stateful})
    pred = predict_all(model, data, Val(:stateful))
    hcat(data, pred |> pred2df, makeunique=true)
end
predict_all2df(fit::NamedTuple, data::DataFrame, ::Val{:stateful}) = predict_all2df(fit.bestModel, data, Val(:stateful))

fitmode(fit::NamedTuple) = fit.stateful ? Val(:stateful) : Val(:stateless)

function evalfit(fit::NamedTuple, df::DataFrame, fig=Figure(resolution=(1920,1080)); kw...)
    df = predict_all2df(fit, df)
    plots=map(fit.latents2record) do latent
        evalplot(df, latent.first |> Symbol, latent.second |> Symbol; kw...)
    end
    d = Dict(1 => (1,1), 2 => (1,2), 3 => (2,2), 4 => (2,2), 5 => (2,3), 6 => (2,3))
    nplots=length(plots)
    layout= nplots in keys(d) ? d[length(plots)] : (3,4) 

    for (i, p) in enumerate(plots)
        col = ((i-1) % layout[2]) + 1
        row = (i-1) ÷ layout[2] + 1
        draw!(fig[row, col], p)
    end
    fig
end
