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
    function WeightedMean{T}(μ = T(0), W = T(0)) where T
        new{T}(T(μ), T(W))
    end
end

WeightedMean(μ::T, W::T) where T = WeightedMean{T}(μ, W)
WeightedMean(::Type{T}) where T = WeightedMean(T(0), T(0))
WeightedMean() = WeightedMean(Float64)

function _fit!(o::WeightedMean{T}, x, w) where T
    ww = convert(T, w)
    xx = convert(T, x)

    o.W += ww
    o.μ = smooth(o.μ, xx, ww / o.W)

    o
end

function _merge!(o::WeightedMean{T}, o2::WeightedMean) where T
    o2_W = convert(T, o2.W)
    o2_μ = convert(T, o2.μ)

    o.W += o2_W
    o.μ = smooth(o.μ, o2_μ, o2_W / o.W)

    o
end
value(o::WeightedMean) = o.μ
mean(o::WeightedMean) = value(o)
Base.sum(o::WeightedMean) = mean(o) * weightsum(o)
Base.copy(o::WeightedMean) = WeightedMean(o.μ, o.W)
