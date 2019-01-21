using StatsBase

"""
the eigenvectors have arbitrary directions.
"""
function test_proj(v1, v2; atol = 0.0)
    all(isapprox(v1[:, i],  v2[:, i], atol = atol) ||
        isapprox(v1[:, i], -v2[:, i], atol = atol) for i in 1:size(v1, 2))
end

@testset "PCA" begin

    x = rand(100, 4)
    w = ones(100)
    zsm = fit(ZScoreTransform, x')
    zm = fit(ZScoreTransform, x', scale = false)

    c = fit!(WeightedCovMatrix(), x, w)

    t, p = pca(c, cov_pca = true)
    p6 = fit(
        PCA,
        StatsBase.transform(zm, permutedims(x)),
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

    t, p = pca(c, cov_pca = false)
    p4 = fit(
        PCA,
        StatsBase.transform(zsm, x'),
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

end
