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
    c = fit!(WeightedCovMatrix(), x, w)

    p1 = PCA(c, cov_pca = true)
    p6 = fit(PCA, x', method = :cov, maxoutdim = 4, pratio = 1.0, mean = nothing, std = 1.0)

    @test p1.mean ≈ p6.mean
    @test p1.std ≈ p6.std
    @test isapprox(p1.prinvars, p6.prinvars, atol = 1.0e-2)
    @test isapprox(p1.tprinvar, p6.tprinvar, atol = 1.0e-2)
    @test isapprox(p1.tvar, p6.tvar, atol = 1.0e-2)
    @test test_proj(p1.proj, p6.proj)

    p2 = PCA(c, cov_pca = false)
    p4 = fit(PCA, x', method = :cov, maxoutdim = 4, pratio = 1.0, mean = nothing, std = nothing)

    @test p2.mean ≈ p4.mean
    @test isapprox(p2.std, p4.std, atol = 1.0e-2)
    @test isapprox(p2.prinvars, p4.prinvars, atol = 1.0e-2)
    @test isapprox(p2.tprinvar, p4.tprinvar, atol = 1.0e-2)
    @test isapprox(p2.tvar, p4.tvar, atol = 1.0e-2)
    @test test_proj(p2.proj, p4.proj)

end
