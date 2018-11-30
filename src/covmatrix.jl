"""
WeightedCovMatrix(T = Float64)

Weighted covariance matrix, tracked as a matrix of type `T`.

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
    function WeightedCovMatrix{T}(
            C = zeros(T, 0, 0), A = zeros(T, 0, 0),
            b = zeros(T, 0), W = T(0), W2 = T(0)) where T
        new{T}(C, A, b, W, W2)
    end
end

WeightedCovMatrix(
        C::Matrix{T}, A::Matrix{T}, b::Vector{T}, W::T, W2::T
    ) where T =
    WeightedCovMatrix{T}(C, A, b, W, W2)
WeightedCovMatrix(::Type{T}, p::Int=0) where T =
    WeightedCovMatrix(zeros(T, p, p), zeros(T, p, p), zeros(T, p), T(0), T(0))
WeightedCovMatrix() = WeightedCovMatrix(Float64)

function _fit!(o::WeightedCovMatrix{T}, x, w) where T
    xx = convert(Vector{T}, x)
    ww = convert(T, w)

    o.W += ww
    o.W2 += ww * ww
    γ = ww / o.W
    if isempty(o.A)
        p = length(xx)
        o.b = zeros(T, p)
        o.A = zeros(T, p, p)
        o.C = zeros(T, p, p)
    end
    smooth!(o.b, xx, γ)
    smooth_syr!(o.A, xx, γ)
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
    else
        W = o.W + o2_W
        γ = o2_W / W
        smooth!(o.A, o2_A, γ)
        smooth!(o.b, o2_b, γ)
        o.W = W
        o.W2 += o2_W2
    end

    return o
end

nvars(o::WeightedCovMatrix) = size(o.A, 1)

function value(o::WeightedCovMatrix)
    o.C[:] = Matrix(Hermitian((o.A - o.b * o.b')))
    o.C
end

function cov(o::WeightedCovMatrix; corrected = false, weight_type = :analytic)
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

Base.sum(o::WeightedCovMatrix) = o.b .* o.W
mean(o::WeightedCovMatrix) = o.b
var(o::WeightedCovMatrix; kw...) = diag(cov(o; kw...))
std(o::WeightedCovMatrix; kw...) = sqrt.(var(o; kw...))

Base.eltype(o::WeightedCovMatrix{T}) where T = T
Base.copy(o::WeightedCovMatrix) =
    WeightedCovMatrix(o.C, o.A, o.b, o.W, o.W2)
