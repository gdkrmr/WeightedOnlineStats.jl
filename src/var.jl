"""
WeightedVariance(T = Float64)

Simple weighted variance, tracked as type `T`.

# Example:
    o = fit!(WeightedVariance(), rand(100), rand(100))
    sum(o)
    mean(o)
    var(o)
    std(o)
"""
mutable struct WeightedVariance{T} <: WeightedOnlineStat{T}
    μ::T
    σ2::T
    W::T
    W2::T
    function WeightedVariance{T}(μ = T(0), σ2 = T(0), W = T(0), W2 = T(0)) where T
        new{T}(T(μ), T(σ2), T(W), T(W2))
    end
end

WeightedVariance(μ::T, σ2::T, W::T, W2::T) where T =
    WeightedVariance{T}(μ, σ2, W, W2)
WeightedVariance(::Type{T}) where T = WeightedVariance(T(0), T(0), T(0), T(0))
WeightedVariance() = WeightedVariance(Float64)

function _fit!(o::WeightedVariance{T}, x, w) where T
    xx = convert(T, x)
    ww = convert(T, w)

    o.W += ww
    o.W2 += ww * ww
    γ = ww / o.W
    μ = o.μ

    o.μ = smooth(o.μ, xx, γ)
    o.σ2 = smooth(o.σ2, (xx - o.μ) * (xx - μ), γ)

    return o
end

function _merge!(o::WeightedVariance{T}, o2::WeightedVariance) where T

    o2_μ = convert(T, o2.μ)
    o2_σ2 = convert(T, o2.σ2)
    o2_W = convert(T, o2.W)
    o2_W2 = convert(T, o2.W2)

    W = o.W + o2_W
    γ1 = o.W / W
    γ2 = o2_W / W

    μ = smooth(o.μ, o2_μ, γ2)

    # o.σ2 =
    #     γ1 * ( o.σ2 +  o.μ ^ 2) +
    #     γ2 * (o2_σ2 + o2_μ ^ 2) -
    #     μ ^ 2
    o.σ2 =
        γ1 * ( o.σ2  + (o.μ - μ) ^ 2) +
        γ2 * (o2_σ2 + (o2_μ - μ) ^ 2)

    o.μ = μ
    o.W = W
    o.W2 += o2_W2

    ###########################################

    # μ = o.μ
    #
    # γ = o2_W / (o2_W + o.W)
    # δ = o2_μ - o.μ
    #
    # o.σ2 = smooth(o.σ2, o2_σ2, γ) + (δ ^ 2) * γ * (1.0 - γ)
    # o.μ = smooth(o.μ, o2_μ, γ)
    # # o.σ2 = o.σ2 + (o.W + o2_W) * (μ)
    #
    # o.W += o2_W
    # o.W2 += o2_W2

    return o
end

value(o::WeightedVariance) = o.σ2
Base.sum(o::WeightedVariance) = o.μ * o.W
mean(o::WeightedVariance) = o.μ
function var(o::WeightedVariance; corrected = false, weight_type = :analytic)
    if corrected
        if weight_type == :analytic
            value(o) / (1 - o.W2 / (weightsum(o) ^ 2))
        elseif weight_type == :frequency
            value(o) / (weightsum(o) - 1) * weightsum(o)
        elseif weight_type == :probability
            error("If you need this, please make a PR or open an issue")
        else
            throw(ArgumentError("weight type $weight_type not implemented"))
        end
    else
        value(o)
    end
end
std(o::WeightedVariance; kw...) = sqrt.(var(o; kw...))
Base.copy(o::WeightedVariance) = WeightedVariance(o.μ, o.σ2, o.W, o.W2)
