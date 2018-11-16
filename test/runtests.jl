# using Revise
# using Pkg
# cd("..")
# Pkg.activate(".")

using Test
using WeightedOnlineStats
import OnlineStatsBase: eachcol, eachrow
using StatsBase
using Statistics
using Random


Random.seed!(123)
l = 1000
x = rand(l);
x2 = rand(l, 5)
w = rand(l);

Base.isapprox(x::Tuple, y::Tuple) =
    reduce(&, map((a, b) -> a ≈ b, x, y))

function test_fit(T::Type{<: WeightedOnlineStats.WeightedOnlineStat{S}},
                  data, weights, unpack_fun, jfun) where S
    l = length(data)
    @assert l == length(weights)

    o = T()
    for (xi, wi) in zip(data, weights)
        fit!(o, xi, wi)
    end

    @test unpack_fun(o) ≈ jfun(data, weights)
    @test eltype(unpack_fun(o)) == eltype(o)
end
# test_fit(WeightedCovMatrix{Float64},
#          eachrow(x2), w,
#          o -> (cov(o), cor(o), mean(o), var(o)),
#          (x3, w2) -> begin
#          (cov(x3.thing, weights(w2), corrected = false),
#           Statistics.cor(x3.thing, weights(w2)),
#           map((x4) -> mean(x4, weights(w2)), eachcol(x3.thing)),
#           map((x4) -> var(x4, weights(w2), corrected = false), eachcol(x3.thing)))
#          end)

@testset "WeightedSum fit!" begin
    test_fit(WeightedSum{Float64}, x, w, sum, (x, w) -> sum(x .* w))
    test_fit(WeightedSum{Float32}, x, w, sum, (x, w) -> sum(x .* w))


    s = sum(broadcast(*, x, w))

    szip = sum(fit!(WeightedSum(), zip(x, w)))
    sval = sum(fit!(WeightedSum(), x, w))

    s2val = sum(fit!(WeightedSum(Float32), x, w))
    s2zip = sum(fit!(WeightedSum(Float32), zip(x, w)))

    @test s ≈ szip
    @test s ≈ sval

    @test s ≈ s2zip
    @test s ≈ s2val
    @test typeof(s2val) == Float32
    @test typeof(s2zip) == Float32
end

@testset "WeightedSum merge!" begin
    s = sum(broadcast(*, x, w))

    smap = map(x, w) do xi, wi
        fit!(WeightedSum(), xi, wi)
    end;

    rs = reduce(merge!, smap) |> sum
    rs2 = merge!(
        fit!(WeightedSum(), x[1:end ÷ 2], w[1:end ÷ 2]),
        fit!(WeightedSum(), x[end ÷ 2 + 1:end], w[end ÷ 2 + 1:end])
    ) |> sum

    # Float32
    smap2 = map(x, w) do xi, wi
        fit!(WeightedSum(), xi, wi)
    end;

    rs_32 = reduce(merge!, smap2, init = WeightedSum(Float32)) |> sum
    rs2_32 = merge!(
        fit!(WeightedSum(Float32), x[1:end ÷ 2], w[1:end ÷ 2]),
        fit!(WeightedSum(), x[end ÷ 2 + 1:end], w[end ÷ 2 + 1:end])
    ) |> sum

    @test rs ≈ s
    @test rs2 ≈ s

    @test rs_32 ≈ s
    @test rs2_32 ≈ s

    @test typeof(rs) == Float64
    @test typeof(rs2) == Float64
    @test typeof(rs_32) == Float32
    @test typeof(rs2_32) == Float32
end

@testset "WeighedMean fit!" begin
    test_fit(WeightedMean{Float64}, x, w, mean, (x, w) -> mean(x, weights(w)))
    test_fit(WeightedMean{Float32}, x, w, mean, (x, w) -> mean(x, weights(w)))

    m = mean(x, weights(w))

    mval = mean(fit!(WeightedMean(), x, w))
    mzip = mean(fit!(WeightedMean(), zip(x, w)))

    @test m ≈ mzip
    @test m ≈ mval

    m2val = mean(fit!(WeightedMean(Float32), x, w))
    m2zip = mean(fit!(WeightedMean(Float32), zip(x, w)))

    @test m ≈ m2zip
    @test m ≈ m2val
    @test typeof(m2val) == Float32
    @test typeof(m2zip) == Float32
end

@testset "WeighedMean merge!" begin
    m = mean(x, weights(w))

    om = map(x, w) do xi, wi
        fit!(WeightedMean(), xi, wi)
    end;

    rm = reduce(merge!, om) |> mean
    rm2 = merge!(
        fit!(WeightedMean(), x[1:end ÷ 2], w[1:end ÷ 2]),
        fit!(WeightedMean(), x[end ÷ 2 + 1:end], w[end ÷ 2 + 1:end])
    ) |> mean

    rm_32 = reduce(merge!, om, init = WeightedMean(Float32)) |> mean
    rm2_32 = merge!(
        fit!(WeightedMean(Float32), x[1:end ÷ 2], w[1:end ÷ 2]),
        fit!(WeightedMean(), x[end ÷ 2 + 1:end], w[end ÷ 2 + 1:end])
    ) |> mean

    @test rm ≈ m
    @test rm2 ≈ m
    @test rm_32 ≈ m
    @test rm2_32 ≈ m

    @test typeof(rm) == Float64
    @test typeof(rm2) == Float64
    @test typeof(rm_32) == Float32
    @test typeof(rm2_32) == Float32
end

@testset "WeightedVariance fit!" begin
    test_fit(WeightedVariance{Float64},
             x, w,
             x -> (var(x), mean(x)),
             (x, w) -> (var(x, weights(w), corrected = false), mean(x, weights(w))))
    test_fit(WeightedVariance{Float32},
             x, w,
             x -> (var(x), mean(x)),
             (x, w) -> (var(x, weights(w), corrected = false), mean(x, weights(w))))

    m, v = mean(x, weights(w)), var(x, weights(w), corrected = false)
    ma, va = mean(x, weights(w)), var(x, aweights(w), corrected = true)
    mf, vf = mean(x, weights(w)), var(x, fweights(w), corrected = true)
    mp, vp = mean(x, weights(w)), var(x, pweights(w), corrected = true)

    mval, vval = fit!(WeightedVariance(), x, w) |> x -> (mean(x), var(x))
    mzip, vzip = fit!(WeightedVariance(), zip(x, w)) |> x -> (mean(x), var(x))

    mvala, vvala = fit!(WeightedVariance(), x, w) |>
        x -> (mean(x), var(x, corrected = true, weight_type = :analytic))
    mvalf, vvalf = fit!(WeightedVariance(), x, w) |>
        x -> (mean(x), var(x, corrected = true, weight_type = :frequency))
    @test_throws ArgumentError fit!(WeightedVariance(), x, w) |>
        x -> (mean(x), var(x, corrected = true, weight_type = :something))

    @test v ≈ vzip
    @test v ≈ vval
    @test m ≈ mzip
    @test m ≈ mval
    @test va ≈ vvala
    @test vf ≈ vvalf
    @test ma ≈ mvala
    @test mf ≈ mvalf

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

@testset "WeightedCovMatrix fit!" begin
    m, c = map(x -> mean(x, weights(w)), eachcol(x2)), cov(x2, weights(w), corrected = false)
    ma, ca = map(x -> mean(x, weights(w)), eachcol(x2)), cov(x2, aweights(w), corrected = true)
    mf, cf = map(x -> mean(x, weights(w)), eachcol(x2)), cov(x2, fweights(w), corrected = true)
    mp, cp = map(x -> mean(x, weights(w)), eachcol(x2)), cov(x2, pweights(w), corrected = true)

    o = WeightedCovMatrix()
    for i in 1:l
        fit!(o, x2[i,:], w[i])
    end
    o
    mfor, cfor = mean(o), cov(o)

    o_32 = WeightedCovMatrix(Float32)
    for i in 1:l
        fit!(o_32, x2[i,:], w[i])
    end
    o_32
    mfor_32, cfor_32 = mean(o_32), cov(o_32)

    mval, cval = fit!(WeightedCovMatrix(), x2, w) |> x -> (mean(x), cov(x))
    mzip, czip = fit!(WeightedCovMatrix(), zip(eachrow(x2), w)) |> x -> (mean(x), cov(x))

    mvala, cvala = fit!(WeightedCovMatrix(), x2, w) |>
        x -> (mean(x), cov(x, corrected = true, weight_type = :analytic))
    mvalf, cvalf = fit!(WeightedCovMatrix(), x2, w) |>
        x -> (mean(x), cov(x, corrected = true, weight_type = :frequency))
    @test_throws ArgumentError fit!(WeightedCovMatrix(), x2, w) |>
        x -> (mean(x), cov(x, corrected = true, weight_type = :something))

    @test c ≈ czip
    @test c ≈ cfor
    @test c ≈ cfor_32
    @test c ≈ cval
    @test m ≈ mzip
    @test m ≈ mfor
    @test m ≈ mfor_32
    @test m ≈ mval
    @test ca ≈ cvala
    @test cf ≈ cvalf
    @test ma ≈ mvala
    @test mf ≈ mvalf

    # After implementing :probability, these should pass/not throw any more:
    @test_throws ErrorException mvalp, vvalp = fit!(WeightedCovMatrix(), x2, w) |>
        x -> (mean(x), cov(x, corrected = true, weight_type = :probability))
    @test_broken vp ≈ vvalp
    @test_broken mp ≈ mvalp

    @test eltype(o) == Float64
    @test eltype(o_32) == Float32
end

@testset "WeightedCovMatrix merge!" begin
    c = cov(x2, weights(w), corrected = false)

    wc = fit!(WeightedCovMatrix(), x2, w)
    oc = map(eachrow(x2), w) do xi, wi
        fit!(WeightedCovMatrix(), xi, wi)
    end;
    rc = reduce(merge!, deepcopy(oc))
    rc2 = merge!(
        fit!(WeightedCovMatrix(), x2[1:end ÷ 2, :],       w[1:end ÷ 2]),
        fit!(WeightedCovMatrix(), x2[end ÷ 2 + 1:end, :], w[end ÷ 2 + 1:end]))

    rc_32 = reduce(merge!, deepcopy(oc), init = WeightedCovMatrix(Float32))
    rc2_32 = merge!(
        fit!(WeightedCovMatrix(Float32), x2[1:end ÷ 2, :],       w[1:end ÷ 2]),
        fit!(WeightedCovMatrix(),        x2[end ÷ 2 + 1:end, :], w[end ÷ 2 + 1:end]))

    @test rc.b ≈ wc.b
    @test rc.C ≈ wc.C
    @test rc.W ≈ wc.W
    @test rc.W2 ≈ wc.W2

    @test rc2.b ≈ wc.b
    @test rc2.C ≈ wc.C
    @test rc2.W ≈ wc.W
    @test rc2.W2 ≈ wc.W2

    @test cov(rc) ≈ c
    @test cov(rc2) ≈ c

    @test rc_32.b ≈ wc.b
    @test rc_32.C ≈ wc.C
    @test rc_32.W ≈ wc.W
    @test rc_32.W2 ≈ wc.W2

    @test rc2_32.b ≈ wc.b
    @test rc2_32.C ≈ wc.C
    @test rc2_32.W ≈ wc.W
    @test rc2_32.W2 ≈ wc.W2

    @test cov(rc_32) ≈ c
    @test cov(rc2_32) ≈ c

    @test eltype(rc) == Float64
    @test eltype(rc2) == Float64
    @test eltype(rc_32) == Float32
    @test eltype(rc2_32) == Float32
end

@testset "Constructors" begin
    @test WeightedSum{Float64}() == WeightedSum()
    @test WeightedSum{Float32}() == WeightedSum(Float32)
    @test WeightedSum() == WeightedSum(0.0, 0.0)

    @test WeightedMean{Float64}() == WeightedMean()
    @test WeightedMean{Float32}() == WeightedMean(Float32)
    @test WeightedMean() == WeightedMean(0.0, 0.0)

    @test WeightedVariance{Float64}() == WeightedVariance()
    @test WeightedVariance{Float32}() == WeightedVariance(Float32)
    @test WeightedVariance() == WeightedVariance(0.0, 0.0, 0.0, 0.0)

    @test WeightedCovMatrix{Float64}() == WeightedCovMatrix()
    @test WeightedCovMatrix{Float32}() == WeightedCovMatrix(Float32)
    @test WeightedCovMatrix() == WeightedCovMatrix(zeros(Float64, 0, 0),
                                                   zeros(Float64, 0, 0),
                                                   zeros(Float64, 0),
                                                   0.0, 0.0, 0)
end
