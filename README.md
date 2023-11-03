# `WeightedOnlineStats.jl`
[![DOI](https://zenodo.org/badge/156201284.svg)](https://zenodo.org/badge/latestdoi/156201284)
[![CI](https://github.com/gdkrmr/WeightedOnlineStats.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/gdkrmr/WeightedOnlineStats.jl/actions/workflows/CI.yml)
[![codecov.io](http://codecov.io/github/gdkrmr/WeightedOnlineStats.jl/coverage.svg?branch=master)](http://codecov.io/github/gdkrmr/WeightedOnlineStats.jl?branch=master)

An extension of `OnlineStatsBase.jl` that supports proper statistical weighting
and arbitrary numerical precision.

# Usage
```julia
using WeightedOnlineStats

values = rand(100)
weights = rand(100)

# fit using arrays:
o1 = fit!(WeightedMean(), values, weights)

# fit using an iterator that returns a tuple (value, weight):
o2 = fit!(WeightedMean(), zip(values, weights))

# fit single values at a time:
o3 = WeightedMean()
for i in 1:length(values)
    fit!(o3, values[i], weights[i])
end

mean(o1)
mean(o2)
mean(o3)
```

# Statistics

`WeightedOnlineStats.jl` currently implements the following Statistics:

- `WeightedSum`
- `WeightedMean`
- `WeightedVariance`
- `WeightedCovMatrix`
- `WeightedHist`
- `WeightedAdaptiveHist`
