import StatsBase
import StatsBase: ZScoreTransform
export ZScoreTransform

import MultivariateStats
import MultivariateStats:
    PCA, pcacov, indim, outdim, projection, principalvar, principalvars,
    tprincipalvar, tresidualvar, tvar, principalratio, transform, reconstruct

export
    pca, PCA, pcacov, indim, outdim, projection, principalvar, principalvars,
    tprincipalvar, tresidualvar, tvar, principalratio, transform, reconstruct

"""
    pca(x::WeightedCovMatrix)::Tuple{StatsBase.ZScoreTransform, MultivariateStats.PCA}

Constructs a `StatsBase.ZScoreTransform`, and a `MultivariateStats.PCA` object
from a `WeightedCovMatrix`.

# Parameters
- `cov_pca::Bool = false`: Do a covariance or correlation pca.
- `maxoutdim::Int = size(x, 1)`: Limits the maximum number of output dimensions.
- `pratio::AbstractFloat = 1.0`: Limits the number of output dimensions to a ratio of explained variance.
- `correct::Bool = false`: Correct the cov(x)/cor(x)/std(x), see the documentation of weights in `StatsBase.jl` for details.
- `weight_type::Symbol`: The type of weight for the correction, see the documentation of weights in `StatsBase.jl` for details. Ignored if `correct` is `false`.

# Example
    c = fit!(WeightedCovMatrix(), rand(100, 4), rand(100))
    t, p = pca(c)
    # note that for prediction/reconstruction with the `PCA` object, the observations are in columns!
    transform((t, p), rand(4, 100))
    reconstruct(p, t), rand(4, 100))
"""
function pca(
        x           :: WeightedCovMatrix{T};
        cov_pca     :: Bool          = false,
        maxoutdim   :: Int           = size(x, 1),
        pratio      :: AbstractFloat = 1.0,
        correct     :: Bool          = false,
        weight_type :: Symbol        = :analytic
    ) where T

    d = size(x, 1)
    c = cov_pca ?
        cov(x, corrected = correct, weight_type = weight_type) :
        cor(x, corrected = correct, weight_type = weight_type)
    m = mean(x)
    s = cov_pca ?
        T[] :
        std(x, corrected = correct, weight_type = weight_type)

    t = ZScoreTransform(d, m, s)
    p = MultivariateStats.pcacov(c, T[], maxoutdim = maxoutdim, pratio = pratio)

    return t, p
end

function transform(t::Tuple{ZScoreTransform, PCA}, x)
    xz = StatsBase.transform(t[1], x)
    transform(t[2], xz)
end

function reconstruct(t::Tuple{PCA, ZScoreTransform}, x)
    xz = reconstruct(t[1], x)
    StatsBase.reconstruct(t[2], xz)
end

transform(t::Tuple{}, x) = x
transform(t::Tuple{T}, x) where T = transform(t, x)
function transform(t::Tuple, x)
    xt = transform(x[1])
    for i in 2:length(t)
        xt = transform(t[i], xt)
    end
    xt
end

reconstruct(t::Tuple{}, x) = x
reconstruct(t::Tuple{T}, x) where T = reconstruct(t, x)
function reconstruct(t::Tuple, x)
    xt = reconstruct(x[1])
    for i in 2:length(t)
        xt = reconstruct(t[i], xt)
    end
    xt
end
