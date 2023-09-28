module WeightedOnlineStats

export WeightedSum, WeightedMean,
    WeightedVariance, WeightedCovMatrix,
    WeightedHist, WeightedAdaptiveHist, WeightedAdaptiveBins,
    fit!, merge!, weightsum, value,
    mean, std, cov, cor, median, quantile

import OnlineStats
import OnlineStats: Tup, VectorOb,
    TwoThings,
    Algorithm, Extrema,
    smooth, smooth!, smooth_syr!
import OnlineStatsBase
import OnlineStatsBase:
    OnlineStat, name,
    fit!, merge!,
    _fit!, _merge!,
    eachrow, eachcol,
    nobs, value
import Statistics
import Statistics: mean, var, std, cov, cor, median, quantile
import LinearAlgebra
import LinearAlgebra: Hermitian, lmul!, rmul!, Diagonal, diag
import StatsBase: midpoints

include("interface.jl")
include("sum.jl")
include("mean.jl")
include("var.jl")
include("covmatrix.jl")

export pca
"""
    `pca` creates a PCA object from a `WeightedCovMatrix` object. Requires MultivariateStats to be loaded!
"""
function pca end
if !isdefined(Base, :get_extension)
    include("../ext/PcaExt.jl")
end

include("histogram.jl")

end # module WeightedOnlineStats
