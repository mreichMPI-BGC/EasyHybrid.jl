using Flux
export LinearHybridModel
########################################
# Model definition y = ax + b, where 
########################################
struct LinearHybridModel # lhm
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


# function LinearHybridModel(predictors, forcing, out_dim, neurons, b=[1.5f0])
#     in_dim = length(predictors)
#     ch = DenseNN(in_dim, out_dim, neurons)
#     LinearHybridModel(ch, predictors, forcing, b)
# end

function LinearHybridModel(predictors, forcing, out_dim, neurons; nn_chain=DenseNN,  b=[1.5f0])
    in_dim = length(predictors)
    ch = nn_chain(in_dim, out_dim, neurons)
    LinearHybridModel(ch, predictors, forcing, b)
end


function (lhm::LinearHybridModel)(dk, ::Val{:infer})
    #dk = permutedims(dk, (1,3,2))
    x_matrix = select_predictors(dk, lhm.predictors)
    α = lhm.DenseLayers(x_matrix)
    #x = lhm.forcing
    ŷ = α .* Array(dk([:x1])) .+ lhm.b
    return (; a=α, pred=ŷ)
end

function (lhm::LinearHybridModel)(df)
    _, ŷ =  lhm(df, :infer)
    return ŷ
end

function (lhm::LinearHybridModel)(df, infer::Symbol)
    return lhm(df, Val(infer))
end

# Call @functor to allow for training the custom model
Flux.@functor LinearHybridModel


struct LinearHybridModel2output # lhm
    DenseLayers::Flux.Chain
    predictors::AbstractArray{Symbol}
    forcing::AbstractArray{Symbol}
    a2
    b
end

## Define construction functions for a FF and a recurrent NN


function LinearHybridModel2output(predictors, forcing, out_dim, neurons; nn_chain=DenseNN,  a2=[1f0], b=[1.5f0])
    in_dim = length(predictors)
    ch = nn_chain(in_dim, out_dim, neurons)
    LinearHybridModel2output(ch, predictors, forcing, a2, b)
end


function (lhm::LinearHybridModel2output)(dk, ::Val{:infer})
    x_matrix = select_predictors(dk, lhm.predictors)
    v(sym::Symbol) = Array(dk([sym]))

    α = lhm.DenseLayers(x_matrix)
    #x = lhm.forcing
    ŷ1 = α .* v(:x1) .+ lhm.b
    ŷ2 = α  .* lhm.a2 .* v(:x2)
    return (; a=α, pred1=ŷ1, pred2 = ŷ2)
end

function (lhm::LinearHybridModel2output)(df)
    res =  lhm(df, :infer)
    return vcat(res.pred1, res.pred2)
end

function (lhm::LinearHybridModel2output)(df, infer::Symbol)
    return lhm(df, Val(infer))
end

# Call @functor to allow for training the custom model
Flux.@functor LinearHybridModel2output