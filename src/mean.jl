"""
    WeightedMean(T = Float64)

Simple weighted mean, tracked as type `T`.

# Example:
    o = fit!(WeightedMean(), rand(100), rand(100))
    sum(o)
    mean(o)
"""
mutable struct WeightedMean{T} <: WeightedOnlineStat{T}
    μ::T
    W::T
    n::Int
    function WeightedMean{T}(μ = T(0), W = T(0), n = 0) where T
        new{T}(T(μ), T(W), Int(n))
    end
end

WeightedMean(μ::T, W::T, n::Int) where T = WeightedMean{T}(μ, W, n)
WeightedMean(::Type{T}) where T = WeightedMean(T(0), T(0), 0)
WeightedMean() = WeightedMean(Float64)

function OnlineStatsBase._fit!(o::WeightedMean{T}, x, w) where T
    xx = convert(T, x)
    ww = convert(T, w)

    o.n += 1
    o.W = smooth(o.W, ww, T(1) / o.n)
    o.μ = smooth(o.μ, xx, ww / (o.W * o.n))

    o
end

function OnlineStatsBase._merge!(o::WeightedMean{T}, o2::WeightedMean) where T
    o2_W = convert(T, o2.W)
    o2_μ = convert(T, o2.μ)

    o.n += o2.n
    o.W = smooth(o.W, o2_W, o2.n / o.n)
    o.μ = smooth(o.μ, o2_μ, (o2_W * o2.n) / (o.W * o.n))

    o
end

OnlineStatsBase.value(o::WeightedMean) = o.μ
Statistics.mean(o::WeightedMean) = value(o)
Base.sum(o::WeightedMean) = mean(o) * weightsum(o)
Base.copy(o::WeightedMean) = WeightedMean(o.μ, o.W, o.n)
