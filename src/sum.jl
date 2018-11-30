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
    function WeightedSum{T}(∑ = T(0), W = T(0)) where T
        new{T}(∑, W)
    end
end

WeightedSum(∑::T, W::T) where T = WeightedSum{T}(∑, W)
WeightedSum(::Type{T}) where T = WeightedSum(T(0), T(0))
WeightedSum() = WeightedSum(Float64)

function _fit!(o::WeightedSum{T}, x, w) where T
    ww = convert(T, w)
    xx = convert(T, x)

    o.W += ww
    o.∑ += xx * ww
    o
end

function _merge!(o::WeightedSum{T}, o2::WeightedSum) where T
    o.W += convert(T, o2.W)
    o.∑ += convert(T, o2.∑)
    o
end

value(o::WeightedSum) = o.∑
Base.sum(o::WeightedSum) = value(o)
Base.copy(o::WeightedSum) = WeightedSum(o.∑, o.W)
