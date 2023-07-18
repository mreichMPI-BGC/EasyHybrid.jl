using EasyHybrid
using Test
using Chain: @chain
using DataFrames
using DataFrameMacros
using Flux
using Random

include("synthetic_test_data.jl")

dk = gen_dk()
dk_twos = gen_dk_2outputs()
#lhm_twos = LinearHybridModel_2outputs([:x2, :x3], [:x1, :x2], 1, 5, DenseNN; b=[4.0f0])
#out_twos = lhm_twos(dk_twos, :infer)

@testset "EasyHybrid.jl" begin
    # test model instantiation
    lhm = LinearHybridModel([:x2, :x3], [:x1], 1, 5, DenseNN; b=[2.0f0])
    @test lhm.forcing == [:x1]
    @test lhm.b == [2.0f0]
    @test lhm.predictors == [:x2, :x3]
    @test typeof(lhm.DenseLayers) <: Flux.Chain

    # test model coupling
    out_lhm = lhm(dk, :infer)
    @test size(out_lhm.a) == (1, 1000)
    @test size(out_lhm.pred) == (1000, 1000)

    # test two_outputs
    lhm_twos = LinearHybridModel_2outputs([:x2, :x3], [:x1, :x2], 1, 5, DenseNN; b=[4.0f0])
    @test lhm_twos.a == [1.0f0]
    @test lhm_twos.b == [4.0f0]
    out_twos = lhm_twos(dk_twos, :infer)
    @test length(out_twos) == 3
    #test model output
end
