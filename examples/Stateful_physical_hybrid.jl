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
begin
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
end

lsm = LinearStateFullModel([:x2, :x3])
dynres3 = fit_df!(lsm, df, [:pred_syn], Flux.mse, stateful=true, shuffle=false, n_epoch=1000, batchsize=10, opt=Adam(0.01), parameters2record=[:b], patience=300)


