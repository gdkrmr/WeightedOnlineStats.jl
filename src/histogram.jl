import OnlineStats: Algorithm, Extrema, fit!
import Statistics: median, quantile
export weightsum, value

##############################################################
# Using the code from  OnlineStats.jl/src/stats/hist.jl
# Modifying it to work with WeightedOnlineStats
##############################################################

abstract type WeightedHistAlgorithm{N} <: Algorithm end
Base.show(io::IO, o::WeightedHistAlgorithm) = print(io, name(o, false, false))
make_alg(o::WeightedHistAlgorithm) = o

#-----------------------------------------------------------------------# Hist
struct WeightedHist{N, H <: WeightedHistAlgorithm{N}} <: WeightedOnlineStat{N}
    alg::H
    W::Number
    WeightedHist{H}(alg::H) where {N, H<:WeightedHistAlgorithm{N}} = new{N, H}(alg)
end
WeightedHist(args...; kw...) = (alg = make_alg(args...; kw...); WeightedHist{typeof(alg)}(alg))

for f in [:nobs, :counts, :midpoints, :edges, :area]
    @eval $f(o::WeightedHist) = $f(o.alg)
end
for f in [:(_fit!), :pdf, :cdf, :(Base.getindex)]
    @eval $f(o::WeightedHist, y, w) = $f(o.alg, y, w)
end

# Base.show(io::IO, o::Hist) = print(io, "Hist: ", o.alg)
_merge!(o::WeightedHist, o2::WeightedHist) = _merge!(o.alg, o2.alg)
function value(o::WeightedHist)
    (midpoints(o), counts(o))
end

split_candidates(o::WeightedHist) = midpoints(o)
mean(o::WeightedHist) = mean(midpoints(o), fweights(counts(o)))
var(o::WeightedHist) = var(midpoints(o), fweights(counts(o)); corrected=true)
std(o::WeightedHist) = sqrt(var(o))
median(o::WeightedHist) = quantile(o, .5)
function Base.extrema(o::WeightedHist)
    mids, counts = value(o)
    inds = findall(x->x!=0, counts)  # filter out zero weights
    mids[inds[1]], mids[inds[end]]
end
function quantile(o::WeightedHist, p = [0, .25, .5, .75, 1])
    mids, counts = value(o)
    inds = findall(x->x!=0, counts)  # filter out zero weights
    quantile(mids[inds], fweights(counts[inds]), p)
end

function Base.show(io::IO, o::WeightedHist)
    print(io, name(o, false, false), ": ")
    print(io, "∑wᵢ=", nobs(o))
    print(io, " | value=")
    show(IOContext(io, :compact => true), value(o))
end

function weightsum(o::WeightedHist)
    nobs(o)
end

#-----------------------------------------------------------------------# WeightedAdaptiveBins
struct WeightedAdaptiveBins <: WeightedHistAlgorithm{Number}
    value::Vector{Pair{Float64, Number}}
    b::Int
    ex::Extrema{Float64}
    W::Number
end
make_alg(b::Int) = WeightedAdaptiveBins(Pair{Float64, Number}[], b, Extrema(Float64), 0.0)
midpoints(o::WeightedAdaptiveBins) = first.(o.value)
counts(o::WeightedAdaptiveBins) = last.(o.value)
nobs(o::WeightedAdaptiveBins) = isempty(o.value) ? 0 : sum(last, o.value)
function Base.:(==)(a::T, b::T) where {T<:WeightedAdaptiveBins}
    (a.value == b.value) && (a.b == b.b) && (a.ex == b.ex)
end
Base.extrema(o::WeightedHist{<:Any, <:WeightedAdaptiveBins}) = extrema(o.alg.ex)

# Doesn't happen with weighted stats
_fit!(o::WeightedAdaptiveBins, y::Number, w::Number) = _fit!(o, Pair(y, w))

function _fit!(o::WeightedAdaptiveBins, y::Pair)
    # o.W += last(y)
    fit!(o.ex, first(y))
    v = o.value
    i = searchsortedfirst(v, y)
    insert!(v, i, y)
    if length(v) > o.b
        # find minimum difference
        i = 0
        mindiff = Inf
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

function _merge!(o::T, o2::T) where {T <: WeightedAdaptiveBins}
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
