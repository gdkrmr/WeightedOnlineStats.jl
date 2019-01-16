module WeightedOnlineStats

export WeightedSum, WeightedMean,
    WeightedVariance, WeightedCovMatrix,
    WeightedHist, WeightedAdaptiveBins,
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

include("interface.jl")
include("sum.jl")
include("mean.jl")
include("var.jl")
include("covmatrix.jl")
include("pca.jl")
include("histogram.jl")

end # module WeightedOnlineStats
