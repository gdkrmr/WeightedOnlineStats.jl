module WeightedOnlineStats

export WeightedSum, WeightedMean, WeightedVariance, WeightedCovMatrix,
    fit!, merge!

import OnlineStats: Tup, VectorOb,
    TwoThings,
    smooth, smooth!, smooth_syr!
import OnlineStatsBase:
    OnlineStat, name,
    fit!, merge!,
    _fit!, _merge!,
    eachrow, eachcol
import Statistics: mean, var, std, cov, cor
import LinearAlgebra: Hermitian, lmul!, rmul!, Diagonal, diag

abstract type WeightedOnlineStat{T} <: OnlineStat{T} end
weightsum(o::WeightedOnlineStat) = o.W
Base.eltype(o::WeightedOnlineStat{T}) where T = T

##############################################################
# Define our own interface so that it accepts two inputs.
##############################################################

# # fit single value and weight
# function fit!(o::WeightedOnlineStat{T}, x::S1, w::S2) where {T, S1<:Number, S2<:Number}
#     _fit!(o, x, w)
#     o
# end
# # fit a tuple, allows fit(o, zip(x, w))
# function fit!(o::WeightedOnlineStat{T}, x::S) where {T, S}
#     for xi in x
#         _fit!(o, xi...)
#     end
#     o
# end
# # fit two iterators, allows fit(o, x::Array, y::Array)
# function fit!(o::WeightedOnlineStat{T}, x, w) where {T}
#     for (xi, wi) in zip(x, w)
#         fit!(o, xi, wi)
#     end
#     o
# end
#
# function fit!(o::WeightedOnlineStat{T}, x::Vector{T}, w::T) where {T<:Number}
#     _fit!(o, x, w)
#     o
# end
#
# function fit!(o::WeightedOnlineStat{T}, x::Matrix{T}, w::Vector{T}) where {T<:Number}
#     for j in 1:size(x,1)
#         fit!(o, x[j,:], w[j])
#     end
#     o
# end
function fit!(o::WeightedOnlineStat{T}, xi::S1, wi::S2) where {T,
                                                               S1<:Number,
                                                               S2<:Number}
    _fit!(o, xi, wi)
    return o
end

function fit!(o::WeightedOnlineStat{VectorOb}, x::VectorOb, w::Number)
    _fit!(o, x, w)
    return o
end

function fit!(o::WeightedOnlineStat{I}, y::T, w::S) where {I, T, S}
    T == eltype(y) &&
        error("The input for $(name(o,false,false)) is a $I.  Found $T.")
    for i in 1:length(w)
        fit!(o, y[i], w[i])
    end
    o
end

fit!(o::WeightedOnlineStat{T}, x::TwoThings{R,S}) where {T, R, S} =
    fit!(o, x[1], x[2])
fit!(o::WeightedOnlineStat{VectorOb}, x::AbstractMatrix, w::AbstractVector) =
    fit!(o, eachrow(x), w)

function merge!(o::WeightedOnlineStat, o2::WeightedOnlineStat)
    (weightsum(o) > 0 || weightsum(o2) > 0) && _merge!(o, o2)
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
# Weighted Sum
##############################################################
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
    o.W += w
    o.∑ += x * w
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

##############################################################
# Weighted Mean
##############################################################
mutable struct WeightedMean{T} <: WeightedOnlineStat{T}
    μ::T
    W::T
    function WeightedMean{T}(μ = T(0), W = T(0)) where T
        new{T}(μ, W)
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
    o2_W = convert(T, o2.W)
    o.W += o2_W
    o.μ = smooth(o.μ, convert(T, o2.μ), o2_W / o.W)
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
    function WeightedVariance{T}(
            μ = T(0), σ2 = T(0),
            W = T(0), W2 = T(0)
        ) where T
        new{T}(μ, σ2, W, W2)
    end
end

WeightedVariance(μ::T, σ2::T, W::T, W2::T) where T =
    WeightedVariance{T}(μ, σ2, W, W2)
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
Base.copy(o::WeightedVariance) = WeightedVariance(o.μ, o.σ2, o.W, o.W2)

##############################################################
# Weighted Covariance Matrix
##############################################################
mutable struct WeightedCovMatrix{T} <: WeightedOnlineStat{VectorOb}
    C::Matrix{T}
    A::Matrix{T}
    b::Vector{T}
    W::T
    W2::T
    n::Int
    function WeightedCovMatrix{T}(
            C = zeros(T, 0, 0), A = zeros(T, 0, 0),
            b = zeros(T, 0), W = T(0), W2 = T(0),
            n = Int(0)
        ) where T
        new{T}(C, A, b, W, W2, n)
    end
end

WeightedCovMatrix(C::Matrix{T}, A::Matrix{T}, b::Vector{T},
                  W::T, W2::T,
                  n::Int) where T = WeightedCovMatrix{T}(C, A, b, W, W2, n)
WeightedCovMatrix(::Type{T}, p::Int=0) where T =
    WeightedCovMatrix(zeros(T, p, p), zeros(T, p, p), zeros(T, p),
                             T(0), T(0), Int(0))
WeightedCovMatrix() = WeightedCovMatrix(Float64)

Base.eltype(o::WeightedCovMatrix{T}) where T = T

function _fit!(o::WeightedCovMatrix{T}, x, w) where T
    x = convert(Vector{T}, x)
    w = convert(T, w)

    o.W += w
    o.W2 += w * w
    o.n += 1
    γ = w / o.W
    if isempty(o.A)
        p = length(x)
        o.b = zeros(T, p)
        o.A = zeros(T, p, p)
        o.C = zeros(T, p, p)
    end
    smooth!(o.b, x, γ)
    smooth_syr!(o.A, x, γ)
end

function _merge!(o::WeightedCovMatrix{T}, o2::WeightedCovMatrix) where T
    o2_A = convert(Matrix{T}, o2.A)
    o2_b = convert(Vector{T}, o2.b)
    o2_W = convert(T, o2.W)
    o2_W2 = convert(T, o2.W2)

    if isempty(o.A)
        o.C = convert(Matrix{T}, o2.C)
        o.A = o2_A
        o.b = o2_b
        o.W = o2_W
        o.W2 = o2_W2
        o.n = o2.n
    else
        W = o.W + o2_W
        γ = o2_W / W
        smooth!(o.A, o2_A, γ)
        smooth!(o.b, o2_b, γ)
        o.W = W
        o.W2 += o2_W2
        o.n += o2.n
    end

    return o
end

nvars(o::WeightedCovMatrix) = size(o.A, 1)

function value(o::WeightedCovMatrix)
    o.C[:] = Matrix(Hermitian((o.A - o.b * o.b')))
    o.C
end

mean(o::WeightedCovMatrix) = o.b
function cov(o::WeightedCovMatrix;
             corrected = false, weight_type = :analytic)
    if corrected
        if weight_type == :analytic
            rmul!(value(o), 1 / (1 - o.W2 / (weightsum(o) ^ 2)))
        elseif weight_type == :frequency
            rmul!(value(o), 1 / (weightsum(o) - 1) * weightsum(o))
        elseif weight_type == :probability
            error("If you need this, please make a PR or open an issue")
        else
            throw(ArgumentError("weight type $weight_type not implemented"))
        end
    else
        value(o)
    end
end

function cor(o::WeightedCovMatrix; kw...)
    cov(o; kw...)
    v = 1 ./ sqrt.(diag(o.C))
    rmul!(o.C, Diagonal(v))
    lmul!(Diagonal(v), o.C)
    o.C
end

var(o::WeightedCovMatrix; kw...) = diag(cov(o; kw...))

Base.copy(o::WeightedCovMatrix) =
    WeightedCovMatrix(o.C, o.A, o.b, o.W, o.W2, o.n)

end # module WeightedOnlineStats

