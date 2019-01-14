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
    @test rm ≈ wm
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
