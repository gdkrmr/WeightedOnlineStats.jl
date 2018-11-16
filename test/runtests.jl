# using Revise
# using Pkg
# cd("..")
# Pkg.activate(".")
using Test
using WeightedOnlineStats
using StatsBase
import OnlineStatsBase: eachcol, eachrow
using Statistics

l = 1000
x = rand(l);
x2 = rand(l, 5)
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
        fit!(WeightedSum(), xi, wi)
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
        fit!(WeightedMean(), xi, wi)
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

@testset "WeightedCovarianceMatrix fit!" begin
    m, c = map(x -> mean(x, weights(w)), eachcol(x2)), cov(x2, weights(w), corrected = false)
    ma, ca = map(x -> mean(x, weights(w)), eachcol(x2)), cov(x2, aweights(w), corrected = true)
    mf, cf = map(x -> mean(x, weights(w)), eachcol(x2)), cov(x2, fweights(w), corrected = true)
    mp, cp = map(x -> mean(x, weights(w)), eachcol(x2)), cov(x2, pweights(w), corrected = true)

    o = WeightedCovarianceMatrix()
    for i in 1:l
        fit!(o, x2[i,:], w[i])
    end
    o
    mfor, cfor = mean(o), cov(o)

    mval, cval = fit!(WeightedCovarianceMatrix(), x2, w) |> x -> (mean(x), cov(x))
    mzip, czip = fit!(WeightedCovarianceMatrix(), zip(eachrow(x2), w)) |> x -> (mean(x), cov(x))

    mvala, cvala = fit!(WeightedCovarianceMatrix(), x2, w) |>
        x -> (mean(x), cov(x, corrected = true, weight_type = :analytic))
    mvalf, cvalf = fit!(WeightedCovarianceMatrix(), x2, w) |>
        x -> (mean(x), cov(x, corrected = true, weight_type = :frequency))
    @test_throws ArgumentError fit!(WeightedCovarianceMatrix(), x2, w) |>
        x -> (mean(x), cov(x, corrected = true, weight_type = :something))

    @test c ≈ czip
    @test c ≈ cfor
    @test c ≈ cval
    @test m ≈ mzip
    @test m ≈ mfor
    @test m ≈ mval
    @test ca ≈ cvala
    @test cf ≈ cvalf
    @test ma ≈ mvala
    @test mf ≈ mvalf

    # After implementing :probability, these should pass/not throw any more:
    @test_throws ErrorException mvalp, vvalp = fit!(WeightedCovarianceMatrix(), x2, w) |>
        x -> (mean(x), cov(x, corrected = true, weight_type = :probability))
    @test_broken vp ≈ vvalp
    @test_broken mp ≈ mvalp
end

@testset "WeightedCovarianceMatrix merge!" begin
    c = cov(x2, weights(w), corrected = false)

    wc = fit!(WeightedCovarianceMatrix(), x2, w)
    oc = map(eachrow(x2), w) do xi, wi
        fit!(WeightedCovarianceMatrix(), xi, wi)
    end;
    rc = reduce(merge!, deepcopy(oc))
    rc2 = merge!(
        fit!(WeightedCovarianceMatrix(), x2[1:end ÷ 2, :],       w[1:end ÷ 2]),
        fit!(WeightedCovarianceMatrix(), x2[end ÷ 2 + 1:end, :], w[end ÷ 2 + 1:end]))

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
end

d1, w1 = fill(1, 40), fill(4, 40)
d2, w2 = fill(2, 30), fill(3, 30)
d3, w3 = fill(3, 20), fill(2, 20)
d4, w4 = fill(4, 10), fill(1, 10)

d, wh = vcat(d1, d2, d3, d4), vcat(w1, w2, w3, w4)

@testset "WeightedHist fit!" begin
    h = (unique(d),
            map(*,
                sort(collect(keys(countmap(wh))), rev = true),
                sort(collect(values(countmap(wh))), rev = true)
            )
        )
    ws = sum(wh)

    o = WeightedHist(4)
    for i in 1:length(d)
        fit!(o, d[i], wh[i])
    end

    hfor, vsfor = value(o), weightsum(o)
    hval, vsval = value(fit!(WeightedHist(4), d, wh)),
                    weightsum(fit!(WeightedHist(4), d, wh))
    hzip, vszip = value(fit!(WeightedHist(4), zip(d, wh))),
                    weightsum(fit!(WeightedHist(4), zip(d, wh)))

    @test (h, ws) == (hfor, vsfor)
    @test (h, ws) == (hval, vsval)
    @test (h, ws) == (hzip, vszip)
end

@testset "WeightedHist merge" begin
    h = (unique(d),
            map(*,
                sort(collect(keys(countmap(wh))), rev = true),
                sort(collect(values(countmap(wh))), rev = true)
            )
        )
    ws = sum(wh)

    oh = map(d, wh) do di, whi
        fit!(WeightedHist(4), di, whi)
    end;

    r = reduce(merge!, oh)
    r2 = merge!(
        fit!(WeightedHist(4), d[1:end ÷ 2], wh[1:end ÷ 2]),
        fit!(WeightedHist(4), d[end ÷ 2 + 1:end], wh[end ÷ 2 + 1:end])
    )

    rh, rws = value(r), weightsum(r)
    rh2, rws2 = value(r2), weightsum(r2)

    @test (h, ws) == (rh, rws)
    @test (h, ws) == (rh2, rws2)
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

    @test WeightedCovarianceMatrix{Float64}() == WeightedCovarianceMatrix()
    @test WeightedCovarianceMatrix{Float32}() == WeightedCovarianceMatrix(Float32)
    @test WeightedCovarianceMatrix() == WeightedCovarianceMatrix(zeros(Float64, 0, 0), zeros(Float64, 0, 0), zeros(Float64, 0), 0.0, 0.0, 0)
end
