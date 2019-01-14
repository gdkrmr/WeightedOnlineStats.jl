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

@testset "WeightedHist copy" begin
    o1 = WeightedHist(20)
    o2 = copy(o1)

    @test Ref(o2.alg.value, 1) != Ref(o1.alg.value, 1)
    @test Ref(o2.alg.b)        != Ref(o1.alg.b)
    @test Ref(o2.alg.ex)       != Ref(o1.alg.ex)
end

@testset "WeightedHist constructor" begin
    @test WeightedHist{WeightedAdaptiveBins{Float64}}(
        WeightedAdaptiveBins{Float64}(
            Pair{Float64, Float64}[], 20, Extrema(Float64)
        )
    ) == WeightedHist(20)
    @test WeightedHist{WeightedAdaptiveBins{Float32}}(
        WeightedAdaptiveBins{Float32}(
            Pair{Float32, Float32}[], 20, Extrema(Float32)
        )
    ) == WeightedHist(Float32, 20)
    @test WeightedHist(20) == WeightedHist(Float64, 20)
end
