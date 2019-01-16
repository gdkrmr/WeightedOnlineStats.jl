"""
    WeightedSum(T = Float64)

Simple weighted sum, tracked as type `T`.

# Example:
    o = fit!(WeightedSum(), rand(100), rand(100))
    sum(o)
"""
mutable struct WeightedSum{T} <: WeightedOnlineStat{T}
    ∑::T
    W::T
    n::Int
    function WeightedSum{T}(∑ = T(0), W = T(0), n = 0) where T
        new{T}(T(∑), T(W), Int(n))
    end
end

WeightedSum(∑::T, W::T, n::Int) where T = WeightedSum{T}(∑, W, n)
WeightedSum(::Type{T}) where T = WeightedSum(T(0), T(0), 0)
WeightedSum() = WeightedSum(Float64)

function WeightedOnlineStats._fit!(o::WeightedSum{T}, x, w) where T
    ww = convert(T, w)
    xx = convert(T, x)

    o.n += 1
    o.W += smooth(o.W, ww, T(1) / o.n)
    o.∑ += xx * ww
    o
end

function WeightedOnlineStats._merge!(o::WeightedSum{T}, o2::WeightedSum) where T
    o.n += o2.n
    o.W = smooth(o.W, convert(T, o2.W), convert(T, o2.n / o.n))
    o.∑ += convert(T, o2.∑)
    o
end

OnlineStatsBase.value(o::WeightedSum) = o.∑
Base.sum(o::WeightedSum) = value(o)
Base.copy(o::WeightedSum) = WeightedSum(o.∑, o.W)
