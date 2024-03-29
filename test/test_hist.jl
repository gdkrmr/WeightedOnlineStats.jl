using Statistics
import OnlineStats
d1, w1 = fill(1, 40), fill(4, 40)
d2, w2 = fill(2, 30), fill(3, 30)
d3, w3 = fill(3, 20), fill(2, 20)
d4, w4 = fill(4, 10), fill(1, 10)

d, wh = vcat(d1, d2, d3, d4), vcat(w1, w2, w3, w4)

@testset "WeightedAdaptiveHist fit!" begin
    h = (unique(d),
            map(*,
                sort(collect(keys(countmap(wh))), rev = true),
                sort(collect(values(countmap(wh))), rev = true)
            )
        )
    ws = sum(wh)

    o = WeightedAdaptiveHist(4)
    for i in 1:length(d)
        fit!(o, d[i], wh[i])
    end

    hfor, vsfor = value(o), weightsum(o)
    hval, vsval = value(fit!(WeightedAdaptiveHist(4), d, wh)),
                    weightsum(fit!(WeightedAdaptiveHist(4), d, wh))
    hzip, vszip = value(fit!(WeightedAdaptiveHist(4), zip(d, wh))),
                    weightsum(fit!(WeightedAdaptiveHist(4), zip(d, wh)))

    @test (h, ws) == (hfor, vsfor)
    @test (h, ws) == (hval, vsval)
    @test (h, ws) == (hzip, vszip)
end

@testset "WeightedAdaptiveHist merge" begin
    h = (unique(d),
            map(*,
                sort(collect(keys(countmap(wh))), rev = true),
                sort(collect(values(countmap(wh))), rev = true)
            )
        )
    ws = sum(wh)

    oh = map(d, wh) do di, whi
        fit!(WeightedAdaptiveHist(4), di, whi)
    end;

    r = reduce(merge!, oh)
    r2 = merge!(
        fit!(WeightedAdaptiveHist(4), d[1:end ÷ 2], wh[1:end ÷ 2]),
        fit!(WeightedAdaptiveHist(4), d[end ÷ 2 + 1:end], wh[end ÷ 2 + 1:end])
    )

    rh, rws = value(r), weightsum(r)
    rh2, rws2 = value(r2), weightsum(r2)

    @test (h, ws) == (rh, rws)
    @test (h, ws) == (rh2, rws2)
end

@testset "WeightedAdaptiveHist copy" begin
    o1 = WeightedAdaptiveHist(20)
    o2 = copy(o1)

    @test o1 !== o2
    @test o2.alg.value !== o1.alg.value
    # @test o2.alg.b     !== o1.alg.b
    @test o2.alg.ex    !== o1.alg.ex
end

@testset "WeightedAdaptiveHist constructor" begin
    @test WeightedAdaptiveHist{WeightedAdaptiveBins{Float64}}(
        WeightedAdaptiveBins{Float64}(
            Pair{Float64, Float64}[], 20, Extrema(Float64)
        )
    ) == WeightedAdaptiveHist(20)
    @test WeightedAdaptiveHist{WeightedAdaptiveBins{Float32}}(
        WeightedAdaptiveBins{Float32}(
            Pair{Float32, Float32}[], 20, Extrema(Float32)
        )
    ) == WeightedAdaptiveHist(Float32, 20)
    @test WeightedAdaptiveHist(20) == WeightedAdaptiveHist(Float64, 20)
end

@testset "WeightedHist" begin
    @testset "fit!" begin
        h = WeightedHist(-3:1:1)

        fit!(h, -2.5,1.3)
        @test h.counts == [1,0,0,0]
        @test h.meanw  == [1.3,0,0,0]

        fit!(h, (-2.1, 1.0))
        @test value(h).y == [2.3, 0,0,0]
        @test h.counts == [2,0,0,0]
        @test h.meanw == [1.15,0,0,0]

        fit!(h, (-20, 2.0))
        @test value(h).y == [2.3, 0,0,0]
        @test h.outcount == [1, 0, 0]
        @test h.meanwout == [2.0,0,0]

        fit!(h, 20, 1.7)
        @test value(h).y == [2.3, 0,0,0]
        @test h.meanwout == [2.0, 0.0, 1.7]
        @test h.outcount == [1, 0, 1]


        fit!(h, -0.1, 1.1)
        @test value(h).y == [2.3, 0,1.1,0]
        @test h.edges === -3:1:1
    end

    @testset "merge!" begin
        h1 = WeightedHist(-2:2:2)
        fit!(h1, (-1, 5))
        fit!(h1, (1, 10))
        h1_copy = deepcopy(h1)
        @test merge!(h1, WeightedHist([-2,0,2])) == h1_copy
        @test value(merge!(h1_copy, h1)).y == [10, 20.]
    end

    @testset "stats" begin
        h = WeightedHist(-2:2:2)
        ho = OnlineStats.Hist(-2.:2:2)
        for _ in 1:rand(1:20)
            x = randn()
            fit!(h, x, 1.0)
            fit!(ho, x)
        end
        @test mean(h) ≈ mean(ho)
        @test std(h) ≈ std(ho)
        @test median(h) ≈ median(ho)
        @test nobs(h) ≈ nobs(ho)
        @test var(h) ≈ var(ho)
        @test all(extrema(h) .≈ extrema(ho))
    end

    @testset "N-dimensional Hist" begin
      h = WeightedHist((-2:2:2,0:3:6))
      fit!(h,(-1.5,1.5),1.5)
      @test h.counts == [1 0; 0 0]
      @test h.meanw  == [1.5 0; 0 0]

      fit!(h,(-0.5,0.2),1.1)
      @test h.counts == [2 0; 0 0]
      @test h.meanw  == [1.3 0;0 0]

      fit!(h,(-3.0,0.0),1.5)
      @test h.counts == [2 0; 0 0]
      @test h.meanw  == [1.3 0;0 0]
      @test h.outcount == [0 1 0; 0 0 0; 0 0 0]

      fit!(h,(-10,-10),1.1)
      @test h.counts == [2 0; 0 0]
      @test h.meanw  == [1.3 0;0 0]
      @test h.outcount == [1 1 0; 0 0 0; 0 0 0]

      fit!(h,(1.5,4.5),2.6)
      @test h.counts == [2 0; 0 1]
      @test h.meanw  == [1.3 0; 0 2.6]
      @test value(h) == (x=(-2:2:2, 0:3:6),y=[2.6 0;0 2.6])

      @test mean(h) == (0.0,3.0)
      @test var(h)  == (1.2380952380952381, 2.785714285714286)
      @test std(h)  == (1.1126972805283737, 1.6690459207925605)
      @test median(h) == (-1.0, 1.5)
      @test nobs(h) == 5
      @test extrema(h) == ((-1,1),(1.5,4.5))
    end


end
