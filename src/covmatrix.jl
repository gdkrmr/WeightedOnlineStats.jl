"""
    WeightedCovMatrix(T = Float64)

Weighted covariance matrix, tracked as a matrix of type `T`.

*After* a call to `cov` the covariance matrix is stored in `o.C`.

# Example:
    o = fit!(WeightedCovMatrix(), rand(100, 4) |> eachrow, rand(100))
    o = fit!(WeightedCovMatrix(), rand(4, 100) |> eachcol, rand(100))
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
    ) where {T}
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
) where {T}
    WeightedCovMatrix{T}(C, A, b, W, W2, n)
end

WeightedCovMatrix(::Type{T}, p::Int = 0) where {T} =
    WeightedCovMatrix(zeros(T, p, p), zeros(T, p, p), zeros(T, p), T(0), T(0), 0)
WeightedCovMatrix() = WeightedCovMatrix(Float64)

# function _fit!(o::WeightedCovMatrix{T}, x, w) where {T}
#     if eltype(x) != T
#         xx = convert(Array{T}, x)
#     else
#         xx = x
#     end
#     ww = convert(T, w)

#     o.n += 1
#     γ1 = T(1) / o.n
#     o.W = smooth(o.W, ww, γ1)
#     o.W2 = smooth(o.W2, ww * ww, γ1)
#     γ2 = ww / (o.W * o.n)
#     if isempty(o.A)
#         p = length(xx)
#         o.b = zeros(T, p)
#         o.A = zeros(T, p, p)
#         o.C = zeros(T, p, p)
#     end
#     smooth!(o.b, xx, γ2)
#     smooth_syr!(o.A, xx, γ2)
# end

function _fit!(o::WeightedCovMatrix{T}, x, w) where {T}
    if eltype(x) != T
        xx = convert(Array{T}, x)
    else
        xx = x
    end
    ww = convert(T, w)

    o.n += 1
    γ1 = T(1) / o.n
    o.W = smooth(o.W, ww, γ1)
    o.W2 = smooth(o.W2, ww * ww, γ1)

    # wsum = o.W * o.n
    γ2 = ww / weightsum(o)

    if isempty(o.A)
        p = length(xx)
        o.b = zeros(T, p)
        o.A = zeros(T, p, p)
        o.C = zeros(T, p, p)
    end

    dx = xx .- o.b
    smooth!(o.b, xx, γ2)

    dx2 = xx .- o.b
    # o.A .+= γ2 .* (dx .* dx2' .- o.A)
    @inbounds for j in 1:size(o.A, 2), i in 1:j
        o.A[i, j] = o.A[i, j] + γ2 * (dx[i] * dx2[j] - o.A[i, j])
    end
    return o
end


function _fit!(o::WeightedCovMatrix{T1},
    x::AbstractVector{Union{T2, Missing}}, w) where {T1, T2}
    if !mapreduce(ismissing, |, x)
        xx = convert(Vector{T1}, x)
        _fit!(o, xx, w)
    end
    return o
end
_fit!(o::WeightedCovMatrix, x, w::Missing) = o

function _merge!(o::WeightedCovMatrix{T}, o2::WeightedCovMatrix) where {T}
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

    ws = weightsum(o)
    ws2 = o2_W * o2.n
    ws3 = W * n
    # o.A .=
    #     (
    #         o.A .* ws .+ o2_A .* ws2 .+
    #         (o.b .- o2_b) .* (o.b .- o2_b)' .* (ws * ws2 / ws3)
    #     ) ./ ws3

    @inbounds for j in 1:size(o.A, 2), i in 1:j
        o.A[i, j] = (
            o.A[i, j] * ws + o2_A[i, j] * ws2 +
                (o.b[i] - o2_b[i]) * (o.b[j] - o2_b[j]) * (ws * ws2 / ws3)
        ) ./ ws3
    end


    # dx = o.b .- o2_b
    # smooth!(o.A, o2_A, γ)
    # o.A .+= o2_A .+ (o.b .- o2_b) .* (o.b .- o2_b)' .* (o.W * o.n * o2_W * o2.n / (W * n))
    # o.A .+= o2_A .+ (o.b .- o2_b) .* (o.b .- o2_b)' .* (o2_W * o2.n)
    # o.A .+= o2_A .+ dx .* dx' .* (o2_W * o2.n)

    γ = (o2_W * o2.n) / (W * n)
    smooth!(o.b, o2_b, γ)
    o.n = n
    o.W = W
    o.W2 = smooth(o.W2, o2_W2, o2.n / n)
end

    return o
end

nvars(o::WeightedCovMatrix) = length(o.b)

function OnlineStatsBase.value(o::WeightedCovMatrix)
    # TODO: copy only the triangle?
    o.C .= o.A
    LinearAlgebra.copytri!(o.C, 'U')
    o.C
end

function Statistics.cov(o::WeightedCovMatrix; corrected = true, weight_type = :analytic)
    if corrected
        if weight_type == :analytic
            value(o) .* (weightsum(o) / (weightsum(o) - o.W2 / o.W))
        elseif weight_type == :frequency
            value(o) .* (weightsum(o) / (weightsum(o) - 1))
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

Base.sum(o::WeightedCovMatrix) = o.b .* weightsum(o)
Statistics.mean(o::WeightedCovMatrix) = copy(o.b)
Statistics.var(o::WeightedCovMatrix; kw...) = diag(cov(o; kw...))
Statistics.std(o::WeightedCovMatrix; kw...) = sqrt.(var(o; kw...))

Base.eltype(o::WeightedCovMatrix{T}) where {T} = T
Base.copy(o::WeightedCovMatrix) =
    WeightedCovMatrix(copy(o.C), copy(o.A), copy(o.b), o.W, o.W2, o.n)

Base.size(x::WeightedCovMatrix, i) = size(x.C, i)
Base.size(x::WeightedCovMatrix) = size(x.C)


function Base.convert(::Type{WeightedCovMatrix{T}}, o::WeightedCovMatrix) where {T}
    WeightedCovMatrix{T}(
        convert(Matrix{T}, o.C),
        convert(Matrix{T}, o.A),
        convert(Vector{T}, o.b),
        convert(T, o.W),
        convert(T, o.W2),
        o.n
    )
end
