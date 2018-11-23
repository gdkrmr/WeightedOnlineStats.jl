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
    w2 = convert(T, w)
    o.W += w2
    o.μ = smooth(o.μ, x, w2 / o.W)
    o
end

function _merge!(o::WeightedMean{T}, o2::WeightedMean) where T
    o2_W = convert(T, o2.W)
    o.W += o2_W
    o.μ = smooth(o.μ, convert(T, o2.μ), o2_W / o.W)
    o
end
value(o::WeightedMean) = o.μ
mean(o::WeightedMean) = value(o)
Base.copy(o::WeightedMean) = WeightedMean(o.μ, o.W)
