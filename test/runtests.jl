# using Revise
# using Pkg
# cd("..")
# Pkg.activate(".")
using Test
using WeightedOnlineStats
using StatsBase
using Statistics

l = 1000
x = rand(l);
w = rand(l);

@testset "WeightedSum fit!" begin
    s = sum(broadcast(*, x, w))

    szip = sum(fit!(WeightedSum(), zip(x, w)))
    o = WeightedSum()
    for i in 1:l
        fit!(o, x[i], w[i])
    end
    sfor = sum(o)
    sval = sum(fit!(WeightedSum(), x, w))

    @test s ≈ szip
    @test s ≈ sfor
    @test s ≈ sval
end

@testset "WeightedSum merge!" begin
    s = sum(broadcast(*, x, w))

    smap = map(x, w) do xi, wi
        WeightedSum(xi, wi)
    end;

    rs = reduce(merge!, smap) |> sum
    rs2 = merge!(
        fit!(WeightedSum(), x[1:end ÷ 2], w[1:end ÷ 2]),
        fit!(WeightedSum(), x[end ÷ 2 + 1:end], w[end ÷ 2 + 1:end])
    ) |> sum

    @test rs ≈ s
    @test rs2 ≈ s
end

@testset "WeighedMean fit!" begin
    m = mean(x, weights(w))

    o = WeightedMean()
    for i in 1:l
        fit!(o, x[i], w[i])
    end
    mfor = mean(o)
    mval = mean(fit!(WeightedMean(), x, w))
    mzip = mean(fit!(WeightedMean(), zip(x, w)))

    @test m ≈ mzip
    @test m ≈ mfor
    @test m ≈ mval

    o2 = WeightedMean(Float32)
    for i in 1:l
        fit!(o2, x[i], w[i])
    end
    m2for = mean(o2)
    m2val = mean(fit!(WeightedMean(Float32), x, w))
    m2zip = mean(fit!(WeightedMean(Float32), zip(x, w)))

    @test m ≈ m2zip
    @test m ≈ m2for
    @test m ≈ m2val
    @test typeof(m2for) == Float32
    @test typeof(m2val) == Float32
    @test typeof(m2zip) == Float32
end

@testset "WeighedMean merge!" begin
    m = mean(x, weights(w))

    om = map(x, w) do xi, wi
        WeightedMean(xi, wi)
    end;

    rm = reduce(merge!, om) |> mean
    rm2 = merge!(
        fit!(WeightedMean(), x[1:end ÷ 2], w[1:end ÷ 2]),
        fit!(WeightedMean(), x[end ÷ 2 + 1:end], w[end ÷ 2 + 1:end])
    ) |> mean

    @test rm ≈ m
    @test rm2 ≈ m
end

@testset "WeightedVariance fit!" begin
    m, v = mean(x, weights(w)), var(x, weights(w), corrected = false)
    ma, va = mean(x, weights(w)), var(x, aweights(w), corrected = true)
    mf, vf = mean(x, weights(w)), var(x, fweights(w), corrected = true)
    mp, vp = mean(x, weights(w)), var(x, pweights(w), corrected = true)

    o = WeightedVariance()
    for i in 1:l
        fit!(o, x[i], w[i])
    end
    o
    mfor, vfor = mean(o), var(o)

    mval, vval = fit!(WeightedVariance(), x, w) |> x -> (mean(x), var(x))
    mzip, vzip = fit!(WeightedVariance(), zip(x, w)) |> x -> (mean(x), var(x))

    mvala, vvala = fit!(WeightedVariance(), x, w) |>
        x -> (mean(x), var(x, corrected = true, weight_type = :analytic))
    mvalf, vvalf = fit!(WeightedVariance(), x, w) |>
        x -> (mean(x), var(x, corrected = true, weight_type = :frequency))
    @test_throws ArgumentError fit!(WeightedVariance(), x, w) |>
        x -> (mean(x), var(x, corrected = true, weight_type = :something))

    @test v ≈ vzip
    @test v ≈ vfor
    @test v ≈ vval
    @test m ≈ mzip
    @test m ≈ mfor
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
end

@testset "Constructors" begin

    @test WeightedMean{Float64}() == WeightedMean()
    @test WeightedMean{Float32}() == WeightedMean(Float32)
    @test WeightedMean() == WeightedMean(0.0, 0.0)

    @test WeightedVariance{Float64}() == WeightedVariance()
    @test WeightedVariance{Float32}() == WeightedVariance(Float32)
    @test WeightedVariance() == WeightedVariance(0.0, 0.0, 0.0, 0.0)
end
