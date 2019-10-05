@testset "WeightedCovMatrix fit!" begin
    s, m, c = (map(x -> sum(x .* w), eachcol(x2)),
               map(x -> mean(x, weights(w)), eachcol(x2)),
               cov(x2, weights(w), corrected = false))
    ma, ca = map(x -> mean(x, weights(w)), eachcol(x2)), cov(x2, aweights(w), corrected = true)
    mf, cf = map(x -> mean(x, weights(w)), eachcol(x2)), cov(x2, fweights(w), corrected = true)
    mp, cp = map(x -> mean(x, weights(w)), eachcol(x2)), cov(x2, pweights(w), corrected = true)

    o = WeightedCovMatrix()
    for i in 1:l
        fit!(o, x2[i,:], w[i])
    end
    o
    sfor, mfor, cfor = sum(o), mean(o), cov(o)

    o_32 = WeightedCovMatrix(Float32)
    for i in 1:l
        fit!(o_32, x2[i,:], w[i])
    end
    o_32
    sfor_32, mfor_32, cfor_32 = sum(o_32), mean(o_32), cov(o_32)

    sval, mval, cval = fit!(WeightedCovMatrix(), x2, w) |> x -> (sum(x), mean(x), cov(x))
    szip, mzip, czip = fit!(WeightedCovMatrix(), zip(eachrow(x2), w)) |> x -> (sum(x), mean(x), cov(x))

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

    @test s ≈ sfor
    @test s ≈ sfor_32
    @test s ≈ sval
    @test s ≈ szip

    # After implementing :probability, these should pass/not throw any more:
    @test_throws ErrorException mvalp, vvalp = fit!(WeightedCovMatrix(), x2, w) |>
        x -> (mean(x), cov(x, corrected = true, weight_type = :probability))
    @test_broken vp ≈ vvalp
    @test_broken mp ≈ mvalp

    @test eltype(o) == Float64
    @test eltype(o_32) == Float32

    oold = copy(o)
    fit!(o, [missing, 1, 2, 3, 4], 1)
    @test o == oold
    fit!(o, [1, 2, 3, 4, 5], missing)
    @test o == oold

    # issue #29
    fit!(o, eachrow([missing 1 2 3 4;
                     missing 2 3 4 5]),
         [1.0, 1.0])
    @test o == oold
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

@testset "WeightedCovMatrix copy" begin
    o = fit!(WeightedCovMatrix(), x2, w)

    o2 = copy(o)

    @test Ref(o.C) != Ref(o2.C)
    @test Ref(o.A) != Ref(o2.A)
    @test Ref(o.b) != Ref(o2.b)
    @test Ref(o.W) != Ref(o2.W)
    @test Ref(o.W2) != Ref(o2.W2)
    @test Ref(o.n) != Ref(o2.n)

    @test o.C == o2.C
    @test o.A == o2.A
    @test o.b == o2.b
    @test o.W == o2.W
    @test o.W2 == o2.W2
    @test o.n == o2.n
end

@testset "WeightedCovMatrix constructor" begin
    @test WeightedCovMatrix{Float64}() == WeightedCovMatrix()
    @test WeightedCovMatrix{Float32}() == WeightedCovMatrix(Float32)
    @test WeightedCovMatrix() == WeightedCovMatrix(zeros(Float64, 0, 0),
                                                   zeros(Float64, 0, 0),
                                                   zeros(Float64, 0),
                                                   0.0, 0.0, 0)
    @test convert(WeightedCovMatrix{Float32}, WeightedCovMatrix()) ==
        WeightedCovMatrix{Float32}()
end
