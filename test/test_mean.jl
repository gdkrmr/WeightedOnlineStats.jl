@testset "WeighedMean fit!" begin
    test_fit(WeightedMean{Float64}, x, w, mean, (x, w) -> mean(x, weights(w)))
    test_fit(WeightedMean{Float32}, x, w, mean, (x, w) -> mean(x, weights(w)))

    test_fit(WeightedMean{Float64}, xmis, wmis, mean,
        (x, w) -> sum(skipmissing(x .* w)) / sum(skipmissing(w)))

    m = mean(x, weights(w))
    s = sum(broadcast(*, w, x))

    mval = mean(fit!(WeightedMean(), x, w))
    mzip = mean(fit!(WeightedMean(), zip(x, w)))
    sval = sum(fit!(WeightedMean(), x, w))
    szip = sum(fit!(WeightedMean(), zip(x, w)))

    @test m ≈ mzip
    @test m ≈ mval
    @test s ≈ szip
    @test s ≈ sval

    m2val = mean(fit!(WeightedMean(Float32), x, w))
    m2zip = mean(fit!(WeightedMean(Float32), zip(x, w)))

    @test m ≈ m2zip
    @test m ≈ m2val
    @test typeof(m2val) == Float32
    @test typeof(m2zip) == Float32
end

@testset "WeighedMean merge!" begin
    m = mean(x, weights(w))
    wm = fit!(WeightedMean(), x, w) |> mean

    om = map(x, w) do xi, wi
        fit!(WeightedMean(), xi, wi)
    end
    rm = reduce(merge!, om) |> mean
    rm_32 = reduce(merge!, om, init = WeightedMean(Float32)) |> mean

    rm2 = merge!(
        fit!(WeightedMean(), x[1:end÷2], w[1:end÷2]),
        fit!(WeightedMean(), x[end÷2+1:end], w[end÷2+1:end])
    ) |> mean
    rm2_32 = merge!(
        fit!(WeightedMean(Float32), x[1:end÷2], w[1:end÷2]),
        fit!(WeightedMean(), x[end÷2+1:end], w[end÷2+1:end])
    ) |> mean

    @test wm ≈ m
    @test rm ≈ m
    @test rm2 ≈ m
    @test rm_32 ≈ m
    @test rm2_32 ≈ m

    @test typeof(rm) == Float64
    @test typeof(rm2) == Float64
    @test typeof(rm_32) == Float32
    @test typeof(rm2_32) == Float32
end

@testset "WeightedMean constructor" begin
    @test WeightedMean{Float64}() == WeightedMean()
    @test WeightedMean{Float32}() == WeightedMean(Float32)
    @test WeightedMean() == WeightedMean(0.0, 0.0, 0)
    @test WeightedMean() == WeightedMean{Float64}(0, 0, 0)
end

@testset "issue #42" begin
    # https://github.com/gdkrmr/WeightedOnlineStats.jl/issues/42

    x = ones(3)
    w = [0.0, 0.5, 1.0]

    m1 = WeightedMean()
    m2 = WeightedMean()

    # value NaN because the first weight is zero
    fit!(m1, x, w)
    # but if we reverse the sequences everything works as expected:
    fit!(m2, reverse(x), reverse(w))
    # WeightedMean: ∑wᵢ=1.5 | value=1.0
    @test m1 == m2

    # zero weight mean later in the sequence should be recoverable
    x = [1.0,  1.0, 2.0,  2.0]
    w = [1.0, -1.0, 1.0, -1.0]

    o = WeightedMean()
    # Weighted mean is empty
    fit!(o, x[1], w[1])
    @test mean(o) === 1.0
    # mean is 1
    fit!(o, x[2], w[2])
    @test mean(o) === NaN
    # mean is NaN because the weight is negative. The negative weight is
    # "deleting" another weight. So here we should have the same state as a new
    # WeightedMean() with no data. The mean is undefined.
    fit!(o, x[3], w[3])
    @test  mean(o) === 2.0
    # the mean is 2 because we "deleted" the first value with the second
    fit!(o, x[4], w[4])
    @test mean(o) === NaN
    # the mean is NaN because we "deleted" the third value with the fourth

    o1 = WeightedMean()
    o2 = WeightedMean()

    fit!(o1, x[1:2], w[1:2])
    @test mean(o1) === NaN
    fit!(o2, 1.0, 1.0)
    @test mean(o2) === 1.0
    merge!(o1, o2)
    @test mean(o1) === 1.0

    o1 = WeightedMean()
    o2 = WeightedMean()

    fit!(o1, x[1:2], w[1:2])
    @test mean(o1) === NaN
    fit!(o2, 1.0, 1.0)
    @test mean(o2) === 1.0
    merge!(o2, o1)
    @test mean(o2) === 1.0

end
