##############################################################
# Using the code from  OnlineStats.jl/src/stats/hist.jl
# Modifying it to work with WeightedOnlineStats
##############################################################

abstract type WeightedHistogramStat{T} <: WeightedOnlineStat{T} end
split_candidates(o::WeightedHistogramStat) = midpoints(o)
Statistics.mean(o::WeightedHistogramStat) = mean(midpoints(o), fweights(counts(o)))
Statistics.var(o::WeightedHistogramStat) = var(midpoints(o), fweights(counts(o)); corrected=true)
Statistics.std(o::WeightedHistogramStat) = sqrt(var(o))
Statistics.median(o::WeightedHistogramStat) = quantile(o, .5)

function Base.show(io::IO, o::WeightedHistogramStat)
    print(io, name(o, false, false), ": ")
    print(io, "∑wᵢ=", nobs(o))
    print(io, " | value=")
    show(IOContext(io, :compact => true), value(o))
end

#-----------------------------------------------------------------------# WeightedHist
"""
    WeightedHist(edges; left = true, closed = true)

Create a histogram with bin partition defined by `edges`.
- If `left`, the bins will be left-closed.
- If `closed`, the bin on the end will be closed.
    - E.g. for a two bin histogram ``[a, b), [b, c)`` vs. ``[a, b), [b, c]``
# Example
    o = fit!(WeightedHist(-5:.1:5), randn(10^6))

    # approximate statistics
    using Statistics
    mean(o)
    var(o)
    std(o)
    quantile(o)
    median(o)
    extrema(o)
    area(o)
    pdf(o)
"""
struct WeightedHist{T, R} <: WeightedHistogramStat{T}
    edges::R
    counts::Vector{Float64}
    out::Vector{Float64}
    left::Bool
    closed::Bool

    function WeightedHist(edges::R, T::Type = eltype(edges); left::Bool=true, closed::Bool = true) where {R<:AbstractVector}
        new{T,R}(edges, zeros(Int, length(edges) - 1), [0,0], left, closed)
    end
end
nobs(o::WeightedHist) = sum(o.counts) + sum(o.out)
weightsum(o::WeightedHist) = nobs(o)
value(o::WeightedHist) = (x=o.edges, y=o.counts)

midpoints(o::WeightedHist) = midpoints(o.edges)
counts(o::WeightedHist) = o.counts
edges(o::WeightedHist) = o.edges

function Base.extrema(o::WeightedHist)
    x, y = midpoints(o), counts(o)
    x[findfirst(x -> x > 0, y)], x[findlast(x -> x > 0, y)]
end
function Statistics.quantile(o::WeightedHist, p = [0, .25, .5, .75, 1])
    x, y = midpoints(o), counts(o)
    inds = findall(x -> x != 0, y)
    quantile(x[inds], fweights(y[inds]), p)
end

function area(o::WeightedHist)
    c = o.counts
    e = o.edges
    if isa(e, AbstractRange)
        return step(e) * sum(c)
    else
        return sum((e[i+1] - e[i]) * c[i] for i in 1:length(c))
    end
end

function pdf(o::WeightedHist, y)
    i = OnlineStats.binindex(o.edges, y, o.left, o.closed)
    if i < 1 || i > length(o.counts)
        return 0.0
    else
        return o.counts[i] / area(o)
    end
end

function _fit!(o::WeightedHist, x, wt)
    i = OnlineStats.binindex(o.edges, x, o.left, o.closed)
    if 1 ≤ i < length(o.edges)
        o.counts[i] += wt
    else
        o.out[1 + (i > 0)] += wt
    end
end

function _merge!(o::WeightedHist, o2::WeightedHist)
    if o.edges == o2.edges
        for j in eachindex(o.counts)
            o.counts[j] += o2.counts[j]
        end
    else
        @warn("WeightedHistogram edges do not align.  Merging is approximate.")
        for (yi, wi) in zip(midpoints(o2.edges), o2.counts)
            for k in 1:wi
                _fit!(o, yi)
            end
        end
    end
end

#-----------------------------------------------------------------------# Adaptive Hist
abstract type WeightedHistAlgorithm{N} <: Algorithm end
Base.show(io::IO, o::WeightedHistAlgorithm) = print(io, name(o, false, false))
make_alg(o::WeightedHistAlgorithm) = o

"""
    Weighted Histogram

Calculate a histogram of weighted data.

# Example
    # A weighted histogram with 4 bins:
    o = fit!(WeightedAdaptiveHist(4), rand(1000), rand(1000))

    mean(o)
    var(o)
    std(o)
    median(o)
    quantile(o, [0, 0.25, 0.5, 0.25, 1.0])
    extrema(o)
"""
struct WeightedAdaptiveHist{N, H <: WeightedHistAlgorithm{N}} <: WeightedHistogramStat{N}
    alg::H
    WeightedAdaptiveHist{H}(alg::H) where {N, H<:WeightedHistAlgorithm{N}} = new{N, H}(alg)
end

WeightedAdaptiveHist(args...; kw...) = (alg = make_alg(args...; kw...); WeightedAdaptiveHist{typeof(alg)}(alg))

for f in [:nobs, :counts, :midpoints, :edges, :area]
    @eval $f(o::WeightedAdaptiveHist) = $f(o.alg)
end
for f in [:(_fit!), :pdf, :cdf, :(Base.getindex)]
    @eval $f(o::WeightedAdaptiveHist, y, w) = $f(o.alg, y, w)
end

Base.copy(o::WeightedAdaptiveHist) = WeightedAdaptiveHist(copy(o.alg))

# Base.show(io::IO, o::Hist) = print(io, "Hist: ", o.alg)
OnlineStatsBase._merge!(o::WeightedAdaptiveHist, o2::WeightedAdaptiveHist) = _merge!(o.alg, o2.alg)
function OnlineStatsBase.value(o::WeightedAdaptiveHist)
    (midpoints(o), counts(o))
end

function Base.extrema(o::WeightedAdaptiveHist)
    mids, counts = value(o)
    inds = findall(x->x!=0, counts)  # filter out zero weights
    mids[inds[1]], mids[inds[end]]
end
function Statistics.quantile(o::WeightedAdaptiveHist, p = [0, .25, .5, .75, 1])
    mids, counts = value(o)
    inds = findall(x->x!=0, counts)  # filter out zero weights
    quantile(mids[inds], fweights(counts[inds]), p)
end

function weightsum(o::WeightedAdaptiveHist)
    nobs(o)
end

#-----------------------------------------------------------------------# WeightedAdaptiveBins
struct WeightedAdaptiveBins{T} <: WeightedHistAlgorithm{T}
    value::Vector{Pair{T, T}}
    b::Int
    ex::Extrema{T}
    function WeightedAdaptiveBins{T}(value = Pair{T, T}[], b = 10, ex = Extrema(T)) where T
        new{T}(value, b, ex)
    end
end

Base.copy(o::T) where T <: WeightedAdaptiveBins = T(copy(o.value), copy(o.b), copy(o.ex))

make_alg(T::Type, b::Int) = WeightedAdaptiveBins{T}(Pair{T, T}[], b, Extrema(T))
make_alg(b::Int) = WeightedAdaptiveBins{Float64}(Pair{Float64, Float64}[], b, Extrema(Float64))
midpoints(o::WeightedAdaptiveBins) = first.(o.value)
counts(o::WeightedAdaptiveBins) = last.(o.value)
OnlineStatsBase.nobs(o::WeightedAdaptiveBins) =
    isempty(o.value) ? 0 : sum(last, o.value)
function Base.:(==)(a::T, b::T) where {T<:WeightedAdaptiveBins}
    (a.value == b.value) && (a.b == b.b) && (a.ex == b.ex)
end
Base.extrema(o::WeightedAdaptiveHist{<:Any, <:WeightedAdaptiveBins}) = extrema(o.alg.ex)

# Doesn't happen with weighted stats
OnlineStatsBase._fit!(o::WeightedAdaptiveBins, y::Number, w::Number) =
    _fit!(o, Pair(y, w))

function OnlineStatsBase._fit!(o::WeightedAdaptiveBins{T}, y::Pair) where T
    y2 = convert(Pair{T, T}, y)

    fit!(o.ex, first(y2))
    v = o.value
    i = searchsortedfirst(v, y2)
    insert!(v, i, y2)
    if length(v) > o.b
        # find minimum difference
        i = 0
        mindiff = T(Inf)
        for k in 1:(length(v) - 1)
            @inbounds diff = first(v[k + 1]) - first(v[k])
            if diff < mindiff
                mindiff = diff
                i = k
            end
        end
        # merge bins i, i+1
        q2, k2 = v[i + 1]
        if k2 > 0
            q1, k1 = v[i]
            k3 = k1 + k2
            v[i] = Pair(smooth(q1, q2, k2 / k3), k3)
        end
        deleteat!(o.value, i + 1)
    end
end

function OnlineStatsBase._merge!(o::T, o2::T) where {T <: WeightedAdaptiveBins}
    for v in o2.value
        _fit!(o, v)
    end
    fit!(o.ex, extrema(o2.ex))
end

function Base.getindex(o::WeightedAdaptiveBins, i)
    if i == 0
        return Pair(minimum(o.ex), 0)
    elseif i == (length(o.value) + 1)
        return Pair(maximum(o.ex), 0)
    else
        return o.value[i]
    end
end

# based on linear interpolation
function pdf(o::WeightedAdaptiveBins, x::Number)
    v = o.value
    if x ≤ minimum(o.ex)
        return 0.0
    elseif x ≥ maximum(o.ex)
        return 0.0
    else
        i = searchsortedfirst(v, Pair(x, 0.0))
        x1, y1 = o[i - 1]
        x2, y2 = o[i]
        return smooth(y1, y2, (x - x1) / (x2 - x1)) / area(o)
    end
end

function cdf(o::WeightedAdaptiveBins, x::Number)
    if x ≤ minimum(o.ex)
        return 0.0
    elseif x ≥ maximum(o.ex)
        return 1.0
    else
        i = searchsortedfirst(o.value, Pair(x, 0.0))
        x1, y1 = o[i - 1]
        x2, y2 = o[i]
        w = x - x1
        h = smooth(y1, y2, (x2 - x) / (x2 - x1))
        return (area(o, i-2) + w * h) / area(o)
    end
end

function area(o::WeightedAdaptiveBins, ind = length(o.value))
    out = 0.0
    for i in 1:ind
        w = first(o[i+1]) - first(o[i])
        h = (last(o[i+1]) + last(o[i])) / 2
        out += h * w
    end
    out
end
