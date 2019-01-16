import MultivariateStats
import MultivariateStats:
    PCA, pcacov, indim, outdim, projection, principalvar, principalvars,
    tprincipalvar, tresidualvar, tvar, principalratio, transform, reconstruct

export
    PCA, pcacov, indim, outdim, projection, principalvar, principalvars,
    tprincipalvar, tresidualvar, tvar, principalratio, transform, reconstruct


"""
    PCA(x::WeightedCovMatrix)

Constructs a `MultivariateStats.PCA` object from a `WeightedCovMatrix`. For
usage of the result, see the documentation of `MultivariateStats.jl`

# Parameters
- `cov_pca::Bool = false`: Do a covariance or correlation pca.
- `maxoutdim::Int = size(x, 1)`: Limits the maximum number of output dimensions.
- `pratio::AbstractFloat = 1.0`: Limits the number of output dimensions to a ratio of explained variance.
- `correct::Bool = false`: Correct the cov(x)/cor(x)/std(x), see the documentation of weights in `StatsBase.jl` for details.
- `weight_type::Symbol`: The type of weight for the correction, see the documentation of weights in `StatsBase.jl` for details. Ignored if `correct` is `false`.

# Example
    c = fit!(WeightedCovMatrix(), rand(100, 4), rand(100))
    p = PCA(c)
    # note that for prediction/reconstruction with the `PCA` object, the observations are in columns!
    transform(p, rand(4, 100))
    reconstruct(p, rand(4, 100))
"""
function MultivariateStats.PCA(
        x           :: WeightedCovMatrix;
        cov_pca     :: Bool          = false,
        maxoutdim   :: Int           = size(x, 1),
        pratio      :: AbstractFloat = 1.0,
        correct     :: Bool          = false,
        weight_type :: Symbol        = :analytic
    )
    c = cov_pca ?
        cov(x, corrected = correct, weight_type = weight_type) :
        cor(x, corrected = correct, weight_type = weight_type)
    m = mean(x)
    s = cov_pca ?
        eltype(m)[] :
        std(x, corrected = correct, weight_type = weight_type)
    return MultivariateStats.pcacov(c, m, s, maxoutdim = maxoutdim, pratio = pratio)
end
