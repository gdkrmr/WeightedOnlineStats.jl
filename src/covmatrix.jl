"""
    WeightedCovMatrix(T = Float64)

Weighted covariance matrix, tracked as a matrix of type `T`.

*After* a call to `cov` the covariance matrix is stored in `o.C`.

# Example:
    o = fit!(WeightedCovMatrix(), rand(100, 4), rand(100))
    sum(o)
    mean(o)
    var(o)
    std(o)
    cov(o)
    cor(o)
"""
mutable struct WeightedCovMatrix{T} <: WeightedOnlineStat{VectorOb}
    C::Matrix{T}
    A::Matrix{T}
    b::Vector{T}
    W::T
    W2::T
    n::Int
    function WeightedCovMatrix{T}(
            C = zeros(T, 0, 0),
            A = zeros(T, 0, 0),
            b = zeros(T, 0),
            W = T(0),
            W2 = T(0),
            n = 0
        ) where T
        new{T}(C, A, b, W, W2, n)
    end
end

function WeightedCovMatrix(
        C::Matrix{T},
        A::Matrix{T},
        b::Vector{T},
        W::T,
        W2::T,
        n::Int
    ) where T
    WeightedCovMatrix{T}(C, A, b, W, W2, n)
end

WeightedCovMatrix(::Type{T}, p::Int=0) where T =
    WeightedCovMatrix(zeros(T, p, p), zeros(T, p, p), zeros(T, p), T(0), T(0), 0)
WeightedCovMatrix() = WeightedCovMatrix(Float64)

function _fit!(o::WeightedCovMatrix{T}, x, w) where T
    xx = convert(Vector{T}, x)
    ww = convert(T, w)

    o.n += 1
    γ1 = T(1) / o.n
    o.W = smooth(o.W, ww, γ1)
    o.W2 = smooth(o.W2, ww*ww, γ1)
    γ2 = ww / (o.W * o.n)
    if isempty(o.A)
        p = length(xx)
        o.b = zeros(T, p)
        o.A = zeros(T, p, p)
        o.C = zeros(T, p, p)
    end
    smooth!(o.b, xx, γ2)
    smooth_syr!(o.A, xx, γ2)
end

function _fit!(o::WeightedCovMatrix{T1}, x::Vector{Union{T2, Missing}}, w) where {T1, T2}
    if !mapreduce(ismissing, |, x)
        xx = convert(Vector{T1}, x)
        _fit!(o, xx, w)
    end
    return o
end
_fit!(o::WeightedCovMatrix, x, w::Missing) = o

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
        n = o.n + o2.n
        W = smooth(o.W, o2_W, o2.n / n)
        γ = (o2_W * o2.n) / (W * n)
        smooth!(o.A, o2_A, γ)
        smooth!(o.b, o2_b, γ)

        o.n = n
        o.W = W
        o.W2 = smooth(o.W2, o2_W2, o2.n / o.n)
    end

    return o
end

nvars(o::WeightedCovMatrix) = size(o.A, 1)

function OnlineStatsBase.value(o::WeightedCovMatrix)
    # o.A is only the upper triangle:
    # o.C .= o.A .- o.b .* o.b'
    @inbounds for i in 1:size(o.A, 1)
        for j in 1:i
            o.C[j, i] = o.A[j, i] - o.b[i] * o.b[j]
        end

    end
    LinearAlgebra.copytri!(o.C, 'U')
    o.C
end

function Statistics.cov(o::WeightedCovMatrix; corrected = false, weight_type = :analytic)
    if corrected
        if weight_type == :analytic
            LinearAlgebra.rmul!(
                value(o), 1 / (1 - (o.W2 * nobs(o)) / (weightsum(o) ^ 2))
            )
        elseif weight_type == :frequency
            LinearAlgebra.rmul!(
                value(o), 1 / (weightsum(o) - 1) * weightsum(o)
            )
        elseif weight_type == :probability
            error("If you need this, please make a PR or open an issue")
        else
            throw(ArgumentError("weight type $weight_type not implemented"))
        end
    else
        value(o)
    end
end

function Statistics.cor(o::WeightedCovMatrix; kw...)
    cov(o; kw...)
    v = diag(o.C)
    v .= 1 ./ sqrt.(v)
    return o.C .* v .* v'
end

Base.sum(o::WeightedCovMatrix) = o.b .* (meanweight(o) * nobs(o))
Statistics.mean(o::WeightedCovMatrix) = copy(o.b)
Statistics.var(o::WeightedCovMatrix; kw...) = diag(cov(o; kw...))
Statistics.std(o::WeightedCovMatrix; kw...) = sqrt.(var(o; kw...))

Base.eltype(o::WeightedCovMatrix{T}) where T = T
Base.copy(o::WeightedCovMatrix) =
    WeightedCovMatrix(copy(o.C), copy(o.A), copy(o.b), o.W, o.W2, o.n)

Base.size(x::WeightedCovMatrix, i) = size(x.C, i)
Base.size(x::WeightedCovMatrix) = size(x.C)


function Base.convert(::Type{WeightedCovMatrix{T}}, o::WeightedCovMatrix) where T
    WeightedCovMatrix{T}(
        convert(Matrix{T}, o.C),
        convert(Matrix{T}, o.A),
        convert(Vector{T}, o.b),
        convert(T,         o.W),
        convert(T,         o.W2),
        o.n
    )
end
