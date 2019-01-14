@testset "WeightedSum fit!" begin
    test_fit(WeightedSum{Float64}, x, w, sum, (x, w) -> sum(x .* w))
    test_fit(WeightedSum{Float32}, x, w, sum, (x, w) -> sum(x .* w))

    test_fit(WeightedSum{Float64}, xmis, wmis, sum, (x, w) -> sum(skipmissing(x .* w)))

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

@testset "WeightedSum constructor" begin
    @test WeightedSum{Float64}() == WeightedSum()
    @test WeightedSum{Float32}() == WeightedSum(Float32)
    @test WeightedSum() == WeightedSum(0.0, 0.0, 0)
end
