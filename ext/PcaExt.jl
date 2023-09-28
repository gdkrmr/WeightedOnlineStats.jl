module PcaExt

using WeightedOnlineStats

import StatsBase
import StatsBase: ZScoreTransform
export ZScoreTransform

import MultivariateStats
import MultivariateStats:
    PCA, pcacov, indim, outdim, projection, principalvar, principalvars,
    tprincipalvar, tresidualvar, tvar, principalratio, transform, reconstruct,
    predict

export
    PCA, pcacov, indim, outdim, projection, principalvar, principalvars,
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
    c = fit!(WeightedCovMatrix(), rand(4, 100), rand(100))
    t, p = pca(c)
"""
function WeightedOnlineStats.pca(
    x::WeightedCovMatrix{T};
    cov_pca::Bool = false,
    maxoutdim::Int = size(x, 1),
    pratio::AbstractFloat = 1.0,
    correct::Bool = false,
    weight_type::Symbol = :analytic
) where {T}

    d = size(x, 1)
    c = cov_pca ?
        cov(x, corrected = correct, weight_type = weight_type) :
        cor(x, corrected = correct, weight_type = weight_type)
    m = mean(x)
    s = cov_pca ?
        T[] :
        std(x, corrected = correct, weight_type = weight_type)

    t = ZScoreTransform(d, 2, m, s)
    p = MultivariateStats.pcacov(c, T[], maxoutdim = maxoutdim, pratio = pratio)

    return t, p
end

# This is type piracy trying to resolve this here: https://github.com/JuliaStats/StatsBase.jl/issues/781

# function StatsBase.transform(t::Tuple{ZScoreTransform, PCA}, x)
#     xz = StatsBase.transform(t[1], x)
#     StatsBase.transform(t[2], xz)
# end

# function StatsBase.reconstruct(t::Tuple{PCA, ZScoreTransform}, x)
#     xz = StatsBase.reconstruct(t[1], x)
#     StatsBase.reconstruct(t[2], xz)
# end
# StatsBase.transform(t::MultivariateStats.PCA, x) = MultivariateStats.predict(t, x)
# StatsBase.reconstruct(t::MultivariateStats.PCA, x) = MultivariateStats.reconstruct(t, x)

# function StatsBase.transform(t::Union{Tuple, Array}, x)
#     # y is typeunstable, probably doesn't matter
#     y = x
#     for tt in t
#         y = StatsBase.transform(tt, y)
#     end
#     return y
# end

# function StatsBase.reconstruct(t::Union{Tuple,Array}, y)
#     # x is typeunstable, probably doesn't matter
#     x = y
#     for tt in t
#         x = StatsBase.reconstruct(tt, x)
#     end
#     return x
# end


end # module
