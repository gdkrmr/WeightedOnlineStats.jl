@testset "WeightedVariance fit!" begin
    test_fit(WeightedVariance{Float64},
             x, w,
             x -> (var(x), mean(x)),
             (x, w) -> (var(x, weights(w), corrected = false), mean(x, weights(w))))
    test_fit(WeightedVariance{Float32},
             x, w,
             x -> (var(x), mean(x)),
             (x, w) -> (var(x, weights(w), corrected = false), mean(x, weights(w))))

    test_fit(WeightedVariance{Float64},
             xmis, wmis,
             x -> (var(x, corrected = false), mean(x)),
             ## I dont know why this version has numerical issues, in my opinion
             ## this should be the "more" correct one:
             # (x, w) -> begin x = # missing_to_nan(x[1:end-2])
             # w = missing_to_nan(w[1:end-2])
             # (var(x[1:end-2], weights(w[1:end-2]), corrected = false),
             # mean(x[1:end-2], weights(w[1:end-2]))) end
             (x, w) -> begin
             m = sum(skipmissing(x .* w)) / sum(skipmissing(w))
             v = sum(skipmissing(w .* ((x .- m) .^ 2))) / sum(skipmissing(w))
             (v, m)
             end)

    s, m, v = sum(x .* w), mean(x, weights(w)), var(x, weights(w), corrected = false)
    sa, ma, va = sum(x .* w), mean(x, weights(w)), var(x, aweights(w), corrected = true)
    sf, mf, vf = sum(x .* w), mean(x, weights(w)), var(x, fweights(w), corrected = true)
    sp, mp, vp = sum(x .* w), mean(x, weights(w)), var(x, pweights(w), corrected = true)

    sval, mval, vval = fit!(WeightedVariance(), x, w)      |> x -> (sum(x), mean(x), var(x))
    szip, mzip, vzip = fit!(WeightedVariance(), zip(x, w)) |> x -> (sum(x), mean(x), var(x))

    svala, mvala, vvala = fit!(WeightedVariance(), x, w) |>
        x -> (sum(x), mean(x), var(x, corrected = true, weight_type = :analytic))
    svalf, mvalf, vvalf = fit!(WeightedVariance(), x, w) |>
        x -> (sum(x), mean(x), var(x, corrected = true, weight_type = :frequency))
    @test_throws ArgumentError fit!(WeightedVariance(), x, w) |>
        x -> (sum(x), mean(x), var(x, corrected = true, weight_type = :something))

    @test v ≈ vzip
    @test v ≈ vval
    @test m ≈ mzip
    @test m ≈ mval
    @test va ≈ vvala
    @test vf ≈ vvalf
    @test ma ≈ mvala
    @test mf ≈ mvalf

    @test s ≈ sval
    @test s ≈ szip
    @test s ≈ svala
    @test s ≈ svalf

    # After implementing :probability, these should pass/not throw any more:
    @test_throws ErrorException mvalp, vvalp = fit!(WeightedVariance(), x, w) |>
        x -> (mean(x), var(x, corrected = true, weight_type = :probability))
    @test_broken vp ≈ vvalp
    @test_broken mp ≈ mvalp
end

@testset "WeighedVariance merge!" begin
    v = var(x, weights(w), corrected = false)

    wv = fit!(WeightedVariance(), x, w)
    ov = map(x, w) do xi, wi
        fit!(WeightedVariance(), xi, wi)
    end;
    rv = reduce(merge!, deepcopy(ov))
    rv2 = merge!(
        fit!(WeightedVariance(), x[1:end ÷ 2],       w[1:end ÷ 2]),
        fit!(WeightedVariance(), x[end ÷ 2 + 1:end], w[end ÷ 2 + 1:end]))

    rv_32 = reduce(merge!, deepcopy(ov), init = WeightedVariance(Float32))
    rv2_32 = merge!(
        fit!(WeightedVariance(Float32), x[1:end ÷ 2],       w[1:end ÷ 2]),
        fit!(WeightedVariance(), x[end ÷ 2 + 1:end], w[end ÷ 2 + 1:end]))

    @test rv.μ ≈ wv.μ
    @test rv.σ2 ≈ wv.σ2
    @test rv.W ≈ wv.W
    @test rv.W2 ≈ wv.W2

    @test rv2.μ ≈ wv.μ
    @test rv2.σ2 ≈ wv.σ2
    @test rv2.W ≈ wv.W
    @test rv2.W2 ≈ wv.W2

    @test var(rv) ≈ v
    @test var(rv2) ≈ v

    @test var(rv_32) ≈ v
    @test var(rv2_32) ≈ v

    @test mean(rv_32) ≈ wv.μ
    @test mean(rv2_32) ≈ wv.μ

    @test eltype(rv) == Float64
    @test eltype(rv2) == Float64
    @test eltype(rv_32) == Float32
    @test eltype(rv2_32) == Float32
end

@testset "WeightedVariance copy" begin
    o = fit!(WeightedVariance(), x2, w)

    o2 = copy(o)

    @test Ref(o.μ) != Ref(o.μ)
    @test Ref(o.σ2) != Ref(o.σ2)
    @test Ref(o.W) != Ref(o2.W)
    @test Ref(o.W2) != Ref(o2.W2)
    @test Ref(o.n) != Ref(o2.n)

    @test o.μ == o2.μ
    @test o.σ2 == o2.σ2
    @test o.W == o2.W
    @test o.W2 == o2.W2
    @test o.n == o2.n
end

@testset "WeightedVariance constructor" begin
    @test WeightedVariance{Float64}() == WeightedVariance()
    @test WeightedVariance{Float32}() == WeightedVariance(Float32)
    @test WeightedVariance() == WeightedVariance(0.0, 0.0, 0.0, 0.0, 0)
    @test WeightedVariance() == WeightedVariance{Float64}(0, 0, 0, 0, 0)
end
