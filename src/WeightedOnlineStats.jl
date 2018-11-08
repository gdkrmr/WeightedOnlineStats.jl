module WeightedOnlineStats

export WeightedMean, WeightedVariance

import OnlineStatsBase: OnlineStat, name, fit!, merge!, _fit!, _merge!
import Statistics: mean, var, std

smooth(a, b, γ) = a + γ * (b - a)

abstract type WeightedOnlineStat{T} <: OnlineStat{T} end
weightsum(o::WeightedOnlineStat) = o.W


##############################################################
# Define our own interface so that it accepts two inputs.
##############################################################

# fit single value and weight
function fit!(o::WeightedOnlineStat{T}, x::T, w::T) where T
    _fit!(o, x, w)
    o
end
# fit a tuple, allows fit(o, zip(x, w))
function fit!(o::WeightedOnlineStat{T}, x::S) where {T, S}
    T == eltype(x) && error("The input for $(name(o,false,false)) is a $T.  Found $S.")
    for xi in x
        fit!(o, xi...)
    end
    o
end
# fit two arrays, allows fit(o, x::Array, y::Array)
function fit!(o::WeightedOnlineStat{T}, x, w) where {T}
    (T == eltype(x) && T == eltype(w)) ||
        error("The input for $(name(o,false,false)) is a $T.  Found $(eltype(x)) and $(eltype(w)).")
    for (xi, wi) in zip(x, w)
        fit!(o, xi, wi)
    end
    o
end
function merge!(o::WeightedOnlineStat, o2::WeightedOnlineStat)
    (weightsum(o) > 0 || weightsum(o) > 0) && _merge!(o, o2)
    o
end
function Base.show(io::IO, o::WeightedOnlineStat)
    print(io, name(o, false, false), ": ")
    print(io, "∑wᵢ=")
    show(IOContext(io, :compact => true), weightsum(o))
    print(io, " | value=")
    show(IOContext(io, :compact => true), value(o))
end

##############################################################
# Weighted Mean
##############################################################
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
    w = convert(T, w)
    o.W += w
    o.μ = smooth(o.μ, x, w / o.W)
    o
end

function _merge!(o::WeightedMean{T}, o2::WeightedMean) where T
    o.W += convert(T, o2.W)
    o.μ = smooth(o.μ, convert(T, o2.μ), o2.W / o.W)
    o
end
value(o::WeightedMean) = o.μ
mean(o::WeightedMean) = value(o)
Base.copy(o::WeightedMean) = WeightedMean(o.μ, o.W)

##############################################################
# Weighted Variance
##############################################################
mutable struct WeightedVariance{T} <: WeightedOnlineStat{T}
    μ::T
    σ2::T
    W::T
    W2::T
    function WeightedVariance{T}(μ = T(0), σ2 = T(0), W = T(0), W2 = T(0)) where T
        new{T}(T(μ), T(σ2), T(W), T(W2))
    end
end

WeightedVariance(μ::T, σ2::T, W::T, W2::T) where T = WeightedVariance{T}(μ, σ2, W, W2)
WeightedVariance(::Type{T}) where T = WeightedVariance(T(0), T(0), T(0), T(0))
WeightedVariance() = WeightedVariance(Float64)
function _fit!(o::WeightedVariance{T}, x, w) where T
    x = convert(T, x)
    w = convert(T, w)

    o.W += w
    o.W2 += w * w
    γ = w / o.W
    μ = o.μ

    o.μ = smooth(o.μ, x, γ)
    # o.S += w * (x - μ) * (x - o.μ)
    o.σ2 = smooth(o.σ2, (x - o.μ) * (x - μ), γ)

    return o
end
# function _fit!(o::Variance, x)
#     μ = o.μ
#     γ = o.weight(o.n += 1)
#     o.μ = smooth(o.μ, x, γ)
#     o.σ2 = smooth(o.σ2, (x - o.μ) * (x - μ), γ)
# end
function _merge!(o::WeightedVariance{T}, o2::WeightedVariance) where T

    W = o.W + o2.W
    γ1 = o.W / W
    γ2 = o2.W / W

    μ = smooth(o.μ, o2.μ, γ2)

    # o.σ2 =
    #     γ1 * ( o.σ2 +  o.μ ^ 2) +
    #     γ2 * (o2.σ2 + o2.μ ^ 2) -
    #     μ ^ 2
    o.σ2 =
        γ1 * ( o.σ2  + (o.μ - μ) ^ 2) +
        γ2 * (o2.σ2 + (o2.μ - μ) ^ 2)

    o.μ = μ
    o.W = W
    o.W2 += o2.W2

    #######################################

    # γ = o2.W / (o.W += o2.W)
    # δ = o2.μ - o.μ
    # o.σ2 = smooth(o.σ2, o2.σ2, γ) + δ ^ 2 * γ * (1.0 - γ)
    # o.μ = smooth(o.μ, o2.μ, γ)

    # o.W2 += o2.W2

    ###########################################

    ###########################################

    # μ = o.μ

    # γ = o2.W / (o2.W + o.W)
    # δ = o2.μ - o.μ

    # # o.σ2 = smooth(o.σ2, o2.σ2, γ) + (δ ^ 2) * γ * (1.0 - γ)
    # o.μ = smooth(o.μ, o2.μ, γ)
    # o.σ2 = o.σ2 + (o.W + o2.W) * (μ)

    # o.W += o2.W
    # o.W2 += o2.W2
    ########

    # o.W += o2.W
    # o.W2 += o2.W2

    # μ = o.μ
    # o.μ = smooth(o.μ, o2.μ, o2.W / o.W)
    # o.S =
    #     smooth(o.S, o2.S, o2.W / o.W) +
    #     smooth(o.S, (o2.μ - μ) * (o2.μ - o.μ), o2.W / o.W)

    ########

    # o.S += o2_W / o.W * (o2_S - o.S)
    # o.S = smooth(o.S, o2_S, o2_W / o.W)
    # o.S =
    #     o.S * o2_W / o.W +
    #     o2_S * o_W / o.W +
    #     ((o2_μ - o.μ) ^ 2) * o2_W * o_W / o.W

    # o.S =
    #     o.S +
    #     o2.S +
    #     o.W / o2.W / (o.W + o2.W) *
    #     ((o2.W * o.μ - (o.W + o2.W) * (o2.μ + o.μ)) ^ 2)
    # o.S = o.S + o2.S - o.W * o.μ^2 - o2.W * o2.μ^2

    # o.μ = smooth(o.μ, o2.μ, o2.W / (o.W + o2.W))


    return o
end
# function _merge!(o::Variance, o2::Variance)
#     γ = o2.n / (o.n += o2.n)
#     δ = o2.μ - o.μ
#     o.σ2 = smooth(o.σ2, o2.σ2, γ) + δ ^ 2 * γ * (1.0 - γ)
#     o.μ = smooth(o.μ, o2.μ, γ)
#     o
# end
value(o::WeightedVariance) = o.σ2
mean(o::WeightedVariance) = o.μ
function var(o::WeightedVariance; corrected = false, weight_type = :analytic)
    if corrected
        if weight_type == :analytic
            # o.S / (weightsum(o) - o.W2 / weightsum(o))
            # o.S / ((weightsum(o) ^ 2) - o.W2)
            # o.S / (weightsum(o) - o.W2 / weightsum(o)) * weightsum(o)
            value(o) / (1 - o.W2 / (weightsum(o) ^ 2))
        elseif weight_type == :frequency
            value(o) / (weightsum(o) - 1) * weightsum(o)
            # o.S / (weightsum(o) - 1) * (weightsum(o) ^ 2)
        elseif weight_type == :probability
            error("If you need this, please make a PR or open an issue")
        else
            throw(ArgumentError("weight type $weight_type not implemented"))
        end
    else
        value(o)
    end
end
std(o::WeightedOnlineStat; kw...) = sqrt.(var(o; kw...))

end # module WeightedOnlineStats
