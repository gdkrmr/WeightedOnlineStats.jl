##############################################################
# Using the code from  OnlineStats.jl/src/stats/hist.jl
# Modifying it to work with WeightedOnlineStats
##############################################################

import LinearAlgebra
abstract type WeightedHistogramStat{T} <: WeightedOnlineStat{T} end
abstract type WeightedHist{T} <: WeightedHistogramStat{T} end
split_candidates(o::WeightedHistogramStat) = midpoints(o)
Statistics.mean(o::WeightedHistogramStat) = mean(midpoints(o), fweights(counts(o)))
Statistics.var(o::WeightedHistogramStat) = var(midpoints(o), fweights(counts(o)); corrected=true)
Statistics.std(o::WeightedHistogramStat) = sqrt.(var(o))
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
struct WeightedHist1D{R} <: WeightedHist{Float64}
    edges::R
    counts::Vector{Int}
    meanw::Vector{Float64}
    outcount::Vector{Int}
    meanwout::Vector{Float64}
    left::Bool
    closed::Bool
end
struct WeightedHistND{R, N} <: WeightedHist{OnlineStats.VectorOb}
    edges::R
    counts::Array{Int,N}
    meanw::Array{Float64,N}
    outcount::Array{Int,N}
    meanwout::Array{Float64,N}
    left::Bool
    closed::Bool
end

function WeightedHist(edges; left::Bool=true, closed::Bool = true)
    edges = isa(edges,Tuple) ? edges : (edges,)
    counts = zeros(Int, map(i->length(i)-1, edges))
    meanw = zeros(Float64, map(i->length(i)-1, edges))
    outcount = zeros(Int,ntuple(_->3,length(edges)))
    meanwout = zeros(Float64,ntuple(_->3,length(edges)))
    if length(edges) == 1
      WeightedHist1D(edges[1],counts,meanw,outcount,meanwout,left,closed)
    else
      WeightedHistND{typeof(edges),length(edges)}(edges, counts, meanw,outcount,meanwout, left, closed)
    end
end
# Special case for 1D Histogram
nobs(o::WeightedHist) = sum(o.counts) + sum(o.outcount)
weightsum(o::WeightedHist) = LinearAlgebra.dot(o.counts, o.meanw) + LinearAlgebra.dot(o.outcount,o.meanwout)
value(o::WeightedHist) = (x=edges(o), y=o.counts .* o.meanw)
binindices(o::WeightedHist1D,x) = OnlineStats.binindex(o.edges, x, o.left, o.closed)
binindices(o::WeightedHistND,x) = CartesianIndex(map((e,ix)->OnlineStats.binindex(e, ix, o.left, o.closed), o.edges, x))
midpoints(o::WeightedHistND) = Iterators.product(map(midpoints,o.edges)...)
midpoints(o::WeightedHist1D) = midpoints(edges(o))
counts(o::WeightedHist) = o.counts
edges(o::WeightedHist) = o.edges
function Statistics.mean(o::WeightedHist)
  weights = value(o).y
  N = ndims(o.counts)
  r = ntuple(N) do idim
    a = map(i->i[idim],midpoints(o))
    mean(a,fweights(weights))
  end
  N==1 ? r[1] : r
end
function Statistics.var(o::WeightedHist)
  weights = value(o).y
  N = ndims(o.counts)
  r = ntuple(N) do idim
    a = map(i->i[idim],midpoints(o))
    var(a,fweights(weights),corrected=true)
  end
  N==1 ? r[1] : r
end
Statistics.std(o::WeightedHist) = sqrt.(var(o))
Statistics.median(o::WeightedHist) = quantile(o, .5)

function Base.extrema(o::WeightedHist1D)
    x, y = midpoints(o), counts(o)
    x[findfirst(!iszero,y)],x[findlast(!iszero,y)]
end
function Base.extrema(o::WeightedHistND{<:Any,N}) where N
    x, y = midpoints(o), counts(o)
    ntuple(N) do idim
      avalue = any(!iszero, y, dims = setdiff(1:N,idim))[:]
      x.iterators[idim][findfirst(avalue)],x.iterators[idim][findlast(avalue)]
    end
end

function Statistics.quantile(o::WeightedHist, p = [0, .25, .5, .75, 1])
    x, y = midpoints(o), counts(o)
    N = ndims(y)
    inds = findall(!iszero, y)
    yweights = fweights(y[inds])
    subset = collect(x)[inds]
    r = ntuple(N) do idim
      data = map(i->i[idim],subset)
      quantile(data, fweights(y[inds]), p)
    end
    if N==1
      return r[1]
    else
      return r
    end
end

function area(o::WeightedHist)
    c = o.counts
    e = o.edges
    return mapreduce(+, CartesianIndices(c)) do I
      ar = prod(map((ed,i)->ed[i+1]-ed[i],e,I.I))
      c[I]*ar
    end
end

outindex(o, ci::CartesianIndex) = CartesianIndex(map((i,l)->i < 1 ? 1 : i > l ? 3 : 2, ci.I, size(o.counts)))
outindex(o, ci::Int) = CartesianIndex(ci < 1 ? 1 : ci > length(o.counts) ? 3 : 2)
function pdf(o::WeightedHist, y)
    ci = binindices(o, y)
    if all(isequal(2),outindex(o,ci).I)
        return o.counts[ci]*o.meanw[ci] / area(o) / weightsum(o)
    else
        return 0.0
    end
end

function _fit!(o::WeightedHist, x, wt)
    #length(x) == N || error("You must provide $(N) values for the histogram")
    ci = binindices(o, x)
    oi = outindex(o,ci)
    if all(isequal(2),oi.I)
        o.counts[ci] += 1
        o.meanw[ci] = smooth(o.meanw[ci], wt, 1.0 / o.counts[ci])
    else
        o.outcount[oi] += 1
        o.meanwout[oi] = smooth(o.meanwout[oi], wt, 1.0 / o.outcount[oi])
    end
end

function _merge!(o::WeightedHist, o2::WeightedHist)
    if o.edges == o2.edges
        for j in eachindex(o.counts)
            newcount = o.counts[j] + o2.counts[j]
            if newcount > 0
              o.meanw[j] = (o.meanw[j]*o.counts[j] + o2.meanw[j]*o2.counts[j])/newcount
            end
            o.counts[j] = newcount
        end
        for j in eachindex(o.outcount)
          newcount = o.outcount[j] + o2.outcount[j]
          if newcount > 0
            o.meanwout[j] = (o.meanwout[j]*o.outcount[j] + o2.meanwout[j]*o2.outcount[j])/newcount
          end
          o.outcount[j] = newcount
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
