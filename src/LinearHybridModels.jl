export LinearHybridModel, LinearHybridModel_2outputs, GRU_NN, DenseNN
########################################
# Model definition y = ax + b, where 
########################################

struct LinearHybridModel <: EasyHybridModels # lhm
    DenseLayers::Flux.Chain
    predictors::AbstractArray{Symbol}
    forcing::AbstractArray{Symbol}
    b
end

## Define construction functions for a FF and a recurrent NN
"""
DenseNN(`in_dim`, `out_dim`, neurons)
"""
function DenseNN(in_dim, out_dim, neurons)
    return Flux.Chain(
        BatchNorm(in_dim),
        Dense(in_dim => neurons, σ),
        Dense(neurons => out_dim),
    )
end

"""
GRU_NN(`in_dim`, `out_dim`, neurons)
"""
function GRU_NN(in_dim, out_dim, neurons)
    return Flux.Chain(
        #BatchNorm(in_dim),
        GRU(in_dim => neurons),
        Dense(neurons => out_dim),
    )
end

"""
LinearHybridModel(predictors::AbstractArray{Symbol}, forcing::AbstractArray{Symbol},
    `out_dim`, neurons, `nn_chain`; b=[1.5f0])

    - nn_chain :: DenseNN, a Dense neural network
"""
function LinearHybridModel(predictors::AbstractArray{Symbol}, forcing::AbstractArray{Symbol},
    out_dim::Int, neurons::Int, nn_chain; b=[1.5f0])

    in_dim = length(predictors)
    ch = nn_chain(in_dim, out_dim, neurons)
    LinearHybridModel(ch, predictors, forcing, b)
end


function (lhm::LinearHybridModel)(dk, ::Val{:infer})
    #dk = permutedims(dk, (1,3,2))
    x_matrix = select_predictors(dk, lhm.predictors)
    α = lhm.DenseLayers(x_matrix)
    x = select_variable(dk, lhm.forcing[1])
    ŷ = α .* x .+ lhm.b # :x1 was hard-coded, this should come from forcing
    return (; a=α, pred=ŷ)
end

function (lhm::LinearHybridModel)(df)
    _, ŷ = lhm(df, :infer)
    return ŷ
end

"""
(lhm::LinearHybridModel)(df, infer::Symbol)
"""
function (lhm::LinearHybridModel)(df, infer::Symbol)
    return lhm(df, Val(infer))
end
# Call @functor to allow for training the custom model
Flux.@functor LinearHybridModel


struct LinearHybridModel_2outputs <: EasyHybridModels # lhm
    DenseLayers::Flux.Chain
    predictors::AbstractArray{Symbol}
    forcing::AbstractArray{Symbol}
    a
    b
end

## Define construction functions for a FF and a recurrent NN

"""
`LinearHybridModel_2outputs`(predictors::AbstractArray{Symbol}, forcing::AbstractArray{Symbol},
    `out_dim::Int`, neurons::Int, `nn_chain`; a=[1.0f0], b=[1.5f0])

    - nn_chain :: DenseNN, a Dense neural network
"""
function LinearHybridModel_2outputs(predictors::AbstractArray{Symbol}, forcing::AbstractArray{Symbol},
    out_dim::Int, neurons::Int, nn_chain; a=[1.0f0], b=[1.5f0])

    in_dim = length(predictors)
    ch = nn_chain(in_dim, out_dim, neurons)
    LinearHybridModel_2outputs(ch, predictors, forcing, a, b)
end


function (lhm::LinearHybridModel_2outputs)(dk, ::Val{:infer})
    x_matrix = select_predictors(dk, lhm.predictors)
    α = lhm.DenseLayers(x_matrix)
    x1 = select_variable(dk, lhm.forcing[1])
    x2 = select_variable(dk, lhm.forcing[2])

    ŷ1 = α .* x1 .+ lhm.b
    ŷ2 = α .* lhm.a .* x2

    return (; a=α, pred1=ŷ1, pred2=ŷ2)
end

function (lhm::LinearHybridModel_2outputs)(df)
    res = lhm(df, :infer)
    return vcat(res.pred1, res.pred2)
end

"""
LinearHybridModel_2outputs(df, infer::Symbol)
"""
function (lhm::LinearHybridModel_2outputs)(df, infer::Symbol)
    return lhm(df, Val(infer))
end
# Call @functor to allow for training the custom model
Flux.@functor LinearHybridModel_2outputs