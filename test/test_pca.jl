using StatsBase

"""
the eigenvectors have arbitrary directions.
"""
function test_proj(v1, v2; atol = 0.0)
    all(isapprox(v1[:, i], v2[:, i], atol = atol) ||
        isapprox(v1[:, i], -v2[:, i], atol = atol) for i in 1:size(v1, 2))
end

@testset "PCA" begin

    x = rand(4, 100)
    w = ones(100)
    zsm = fit(ZScoreTransform, x, dims = 2)
    zm = fit(ZScoreTransform, x, scale = false, dims = 2)

    c = fit!(WeightedCovMatrix(), x', w)

    t, p = pca(c, cov_pca = true)
    p6 = fit(
        PCA,
        StatsBase.transform(zm, x),
        method = :cov,
        maxoutdim = 4,
        pratio = 1.0,
        mean = nothing
    )

    @test t.mean ≈ zm.mean
    @test t.scale ≈ zm.scale
    @test isapprox(p.prinvars, p6.prinvars, atol = 1.0e-2)
    @test isapprox(p.tprinvar, p6.tprinvar, atol = 1.0e-2)
    @test isapprox(p.tvar, p6.tvar, atol = 1.0e-2)
    @test test_proj(p.proj, p6.proj)

    # @test x === StatsBase.transform((), x)
    # @test x === StatsBase.reconstruct((), x)
    # @test x === StatsBase.transform([], x)
    # @test x === StatsBase.reconstruct([], x)

    y = StatsBase.transform(t, x)
    x2 = StatsBase.reconstruct(t, y)
    @test isapprox(x2, x)

    x2 = StatsBase.transform(t, x)
    y = StatsBase.predict(p, x2)
    x3 = MultivariateStats.reconstruct(p, y)
    x4 = StatsBase.reconstruct(t, x3)
    @test isapprox(x4, x)


    t, p = pca(c, cov_pca = false)
    p4 = fit(
        PCA,
        StatsBase.transform(zsm, x),
        method = :cov,
        maxoutdim = 4,
        pratio = 1.0,
        mean = 0
    )

    @test t.mean ≈ zsm.mean
    @test isapprox(t.scale, zsm.scale, atol = 1.0e-2)
    @test isapprox(p.prinvars, p4.prinvars, atol = 1.0e-2)
    @test isapprox(p.tprinvar, p4.tprinvar, atol = 1.0e-2)
    @test isapprox(p.tvar, p4.tvar, atol = 1.0e-2)
    @test test_proj(p.proj, p4.proj)

    y = StatsBase.transform(t, x)
    x2 = StatsBase.reconstruct(t, y)
    @test isapprox(x2, x)

    x2 = StatsBase.transform(t, x)
    y = StatsBase.predict(p, x2)
    x3 = MultivariateStats.reconstruct(p, y)
    x4 = StatsBase.reconstruct(t, x3)
    @test isapprox(x4, x)
end
