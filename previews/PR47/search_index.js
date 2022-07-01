var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"","category":"page"},{"location":"api/","page":"API","title":"API","text":"Modules = [WeightedOnlineStats]","category":"page"},{"location":"api/#WeightedOnlineStats.WeightedAdaptiveHist","page":"API","title":"WeightedOnlineStats.WeightedAdaptiveHist","text":"Weighted Histogram\n\nCalculate a histogram of weighted data.\n\nExample\n\n# A weighted histogram with 4 bins:\no = fit!(WeightedAdaptiveHist(4), rand(1000), rand(1000))\n\nmean(o)\nvar(o)\nstd(o)\nmedian(o)\nquantile(o, [0, 0.25, 0.5, 0.25, 1.0])\nextrema(o)\n\n\n\n\n\n","category":"type"},{"location":"api/#WeightedOnlineStats.WeightedCovMatrix","page":"API","title":"WeightedOnlineStats.WeightedCovMatrix","text":"WeightedCovMatrix(T = Float64)\n\nWeighted covariance matrix, tracked as a matrix of type T.\n\nAfter a call to cov the covariance matrix is stored in o.C.\n\nExample:\n\no = fit!(WeightedCovMatrix(), rand(100, 4) |> eachrow, rand(100))\no = fit!(WeightedCovMatrix(), rand(4, 100) |> eachcol, rand(100))\nsum(o)\nmean(o)\nvar(o)\nstd(o)\ncov(o)\ncor(o)\n\n\n\n\n\n","category":"type"},{"location":"api/#WeightedOnlineStats.WeightedHist1D","page":"API","title":"WeightedOnlineStats.WeightedHist1D","text":"WeightedHist(edges; left = true, closed = true)\n\nCreate a histogram with bin partition defined by edges.\n\nIf left, the bins will be left-closed.\nIf closed, the bin on the end will be closed.\nE.g. for a two bin histogram a b) b c) vs. a b) b c\n\nIf edges is a tuple instead of an array, a multidimensional histogram will be generated that behaves like a WeightedOnlineStat{VectorOb}.\n\nExamples\n\no = fit!(WeightedHist(-5:.1:5), randn(10^6))\n\n# approximate statistics\nusing Statistics\nmean(o)\nvar(o)\nstd(o)\nquantile(o)\nmedian(o)\nextrema(o)\narea(o)\npdf(o)\n\n2d Histogram\n\nhist2d = fit!(WeightedHist((-5:1:5, -5:1:5) ), randn(10000,2), rand(10000))\nvalue(hist2d).y\n\n\n\n\n\n","category":"type"},{"location":"api/#WeightedOnlineStats.WeightedMean","page":"API","title":"WeightedOnlineStats.WeightedMean","text":"WeightedMean(T = Float64)\n\nSimple weighted mean, tracked as type T.\n\nExample:\n\no = fit!(WeightedMean(), rand(100), rand(100))\nsum(o)\nmean(o)\n\n\n\n\n\n","category":"type"},{"location":"api/#WeightedOnlineStats.WeightedSum","page":"API","title":"WeightedOnlineStats.WeightedSum","text":"WeightedSum(T = Float64)\n\nSimple weighted sum, tracked as type T.\n\nExample:\n\no = fit!(WeightedSum(), rand(100), rand(100))\nsum(o)\n\n\n\n\n\n","category":"type"},{"location":"api/#WeightedOnlineStats.WeightedVariance","page":"API","title":"WeightedOnlineStats.WeightedVariance","text":"WeightedVariance(T = Float64)\n\nSimple weighted variance, tracked as type T.\n\nExample:\n\no = fit!(WeightedVariance(), rand(100), rand(100))\nsum(o)\nmean(o)\nvar(o)\nstd(o)\n\n\n\n\n\n","category":"type"},{"location":"api/#WeightedOnlineStats.pca-Union{Tuple{WeightedCovMatrix{T}}, Tuple{T}} where T","page":"API","title":"WeightedOnlineStats.pca","text":"pca(x::WeightedCovMatrix)::Tuple{StatsBase.ZScoreTransform, MultivariateStats.PCA}\n\nConstructs a StatsBase.ZScoreTransform, and a MultivariateStats.PCA object from a WeightedCovMatrix.\n\nParameters\n\ncov_pca::Bool = false: Do a covariance or correlation pca.\nmaxoutdim::Int = size(x, 1): Limits the maximum number of output dimensions.\npratio::AbstractFloat = 1.0: Limits the number of output dimensions to a ratio of explained variance.\ncorrect::Bool = false: Correct the cov(x)/cor(x)/std(x), see the documentation of weights in StatsBase.jl for details.\nweight_type::Symbol: The type of weight for the correction, see the documentation of weights in StatsBase.jl for details. Ignored if correct is false.\n\nExample\n\nc = fit!(WeightedCovMatrix(), rand(4, 100), rand(100))\nt, p = pca(c)\n\n\n\n\n\n","category":"method"},{"location":"#WeightedOnlineStats.jl","page":"WeightedOnlineStats.jl","title":"WeightedOnlineStats.jl","text":"","category":"section"},{"location":"","page":"WeightedOnlineStats.jl","title":"WeightedOnlineStats.jl","text":"A version onf OnlineStats.jl allowing observations with custom weights.","category":"page"},{"location":"#Basics","page":"WeightedOnlineStats.jl","title":"Basics","text":"","category":"section"},{"location":"#Creating","page":"WeightedOnlineStats.jl","title":"Creating","text":"","category":"section"},{"location":"","page":"WeightedOnlineStats.jl","title":"WeightedOnlineStats.jl","text":"using WeightedOnlineStats\nm = WeightedMean()\nx = randn(100);\nw = randn(100);","category":"page"},{"location":"#Updating","page":"WeightedOnlineStats.jl","title":"Updating","text":"","category":"section"},{"location":"","page":"WeightedOnlineStats.jl","title":"WeightedOnlineStats.jl","text":"fit!(m, x, w)","category":"page"},{"location":"#Merging","page":"WeightedOnlineStats.jl","title":"Merging","text":"","category":"section"},{"location":"","page":"WeightedOnlineStats.jl","title":"WeightedOnlineStats.jl","text":"m2 = WeightedMean()\nx2 = rand(100);\nw2 = rand(100);\nfit!(m2, x2, w2)\nmerge!(m, m2)","category":"page"}]
}
