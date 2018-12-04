using Revise
using StatsBase
using Statistics
using OnlineStats
using BenchmarkTools
using WeightedOnlineStats

x = (collect(reverse((1:10_000_000)*0.0000001)))
w = (collect((1:10_000_000)*0.0000001))
x1 = x[1:5_000_000]
x2 = (x[5_000_001:end])
w1 = w[1:5_000_000]
w2 = (w[5_000_001:end])

vw = WeightedVariance()
vw1 = WeightedVariance()
vw2 = WeightedVariance()

fit!(vw, x, w)
fit!(vw1, x1, w1)
fit!(vw2, x2, w2)

@code_warntype merge!(vw1, vw2)

@benchmark merge($vw1, $vw2)
#=
BenchmarkTools.Trial.1.1:
ll. 133-145 (w/o 135-138)
  memory estimate:  544 bytes
  allocs estimate:  7
  --------------
  minimum time:     3.140 μs (0.00% GC)
  median time:      3.257 μs (0.00% GC)
  mean time:        3.476 μs (1.52% GC)
  maximum time:     272.601 μs (97.37% GC)
  --------------
  samples:          10000
  evals/sample:     8

BenchmarkTools.Trial.1.2:
ll. 133-145 (w/o 139-141)
  memory estimate:  544 bytes
  allocs estimate:  7
  --------------
  minimum time:     3.141 μs (0.00% GC)
  median time:      3.206 μs (0.00% GC)
  mean time:        3.380 μs (1.54% GC)
  maximum time:     268.521 μs (97.16% GC)
  --------------
  samples:          10000
  evals/sample:     8

BenchmarkTools.Trial.2:
ll. 151-161
  memory estimate:  544 bytes
  allocs estimate:  7
  --------------
  minimum time:     3.136 μs (0.00% GC)
  median time:      3.229 μs (0.00% GC)
  mean time:        3.347 μs (1.53% GC)
  maximum time:     262.741 μs (97.26% GC)
  --------------
  samples:          10000
  evals/sample:     8
=#
x = (collect(reverse((1:10_000_000)*0.0000001)))
w = (collect((1:10_000_000)*0.0000001))
x1 = x[1:5_000_000]
x2 = BigFloat.(x[5_000_001:end])
w1 = w[1:5_000_000]
w2 = BigFloat.(w[5_000_001:end])

vw = WeightedVariance()
vw1 = WeightedVariance()
vw2 = WeightedVariance(BigFloat)

fit!(vw, x, w)
fit!(vw1, x1, w1)
fit!(vw2, x2, w2)

@code_warntype merge(vw1, vw2)
merge(vw1, vw2)
merge(vw2, vw1)
@benchmark merge($vw1, $vw2)
@benchmark merge($vw2, $vw1)
#=
BenchmarkTools.Trial.1.1:
ll. 133-145 (w/o 135-138)
  memory estimate:  2.28 KiB
  allocs estimate:  39
  --------------
  minimum time:     4.955 μs (0.00% GC)
  median time:      5.060 μs (0.00% GC)
  mean time:        5.603 μs (3.95% GC)
  maximum time:     346.025 μs (94.78% GC)
  --------------
  samples:          10000
  evals/sample:     7

BenchmarkTools.Trial.1.2:
ll. 133-145 (w/o 139-141)
  memory estimate:  2.06 KiB
  allocs estimate:  35
  --------------
  minimum time:     4.779 μs (0.00% GC)
  median time:      4.861 μs (0.00% GC)
  mean time:        5.366 μs (3.63% GC)
  maximum time:     348.297 μs (96.43% GC)
  --------------
  samples:          10000
  evals/sample:     7

BenchmarkTools.Trial.2:
ll. 151-161
  memory estimate:  2.50 KiB
  allocs estimate:  43
  --------------
  minimum time:     5.398 μs (0.00% GC)
  median time:      5.515 μs (0.00% GC)
  mean time:        6.042 μs (3.80% GC)
  maximum time:     426.032 μs (96.56% GC)
  --------------
  samples:          10000
  evals/sample:     6
=#

using Traceur

x = rand(100)
x2 = rand(100, 2)
w = rand(100)

@trace fit!(WeightedSum(), x, w)
@trace fit!(WeightedSum(Float32), x, w)

@trace fit!(WeightedMean(), x, w)
@trace fit!(WeightedMean(Float32), x, w)

@trace fit!(WeightedVariance(), x, w)
@trace fit!(WeightedVariance(Float32), x, w)

@trace fit!(WeightedCovMatrix(), x2, w)
@trace fit!(WeightedCovMatrix(Float32), x2, w)
