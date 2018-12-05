"""
WeightedPCA(T = Float64)

Wrapper for WeightedCovMatrix. Behaves like WeightedCovMatrix, except the
value()-function returns a PCA model instead of the covariance matrix.

# Example:
    o = fit!(WeightedPCA(), rand(100, 4), rand(100))
    sum(o)
    mean(o)
    var(o)
    std(o)
    cov(o)
    cor(o)
    ^^^^^^____ functions above work the same as WeightedCovMatrix
    value(o)
"""
mutable struct WeightedPCA{T} <: WeightedOnlineStat{VectorOb}
    WCM::WeightedCovMatrix{T}
    model::PCA{T}
    function WeightedPCA{T}(
            WCM = WeightedCovMatrix(T),
            model = PCA(zeros(T,0), zeros(T,0,0),zeros(T,0), T(0), T(0))) where T
        new{T}(WCM, model)
    end
end

WeightedPCA(WCM::WeightedCovMatrix{T}, model::PCA{T}) where T = WeightedPCA{T}(WCM, model)
WeightedPCA(WCM::WeightedCovMatrix{T}) where T = WeightedPCA(WCM, PCA(zeros(T,0), zeros(T,0,0),zeros(T,0), T(0), T(0)))
WeightedPCA(::Type{T}, p::Int=0) where T = WeightedPCA(WeightedCovMatrix(T, p),PCA(zeros(T,0), zeros(T,0,0),zeros(T,0), T(0), T(0)))
WeightedPCA() = WeightedPCA(Float64)

function _fit!(o::WeightedPCA{T}, x, w) where T
    fit!(o.WCM, x, w)
end

# function _fit(o::WeightedPCA{T1}, x::Vector{Union{T2, Missing}}, w) where {T1, T2}
#     if !mapreduce(ismissing, |, x)
#         xx = convert(Vector{T1}, x)
#         fit!(o.WCM, xx, w)
#     end
# end

function _merge!(o::WeightedPCA{T}, o2::WeightedPCA) where T
    merge!(o.WCM, o2.WCM)
end

nvars(o::WeightedPCA) = nvars(o.WCM)
nobs(o::WeightedPCA) = nobs(o.WCM)
meanweight(o::WeightedPCA) = meanweight(o.WCM)
weightsum(o::WeightedPCA) = weightsum(o.WCM)

function value(o::WeightedPCA; maxoutdim::Int=min(nvars(o), nobs(o)), pratio::Number=0.99)
    if maxoutdim == 0
        0
    else
        o.model = pcacov(value(o.WCM), mean(o.WCM), maxoutdim=maxoutdim, pratio=pratio)
        o.model
    end
end

function cov(o::WeightedPCA; corrected = false, weight_type = :analytic)
    cov(o.WCM, corrected = corrected, weight_type = weight_type)
end

function cor(o::WeightedPCA; kw...)
    cor(o.WCM; kw...)
end

Base.sum(o::WeightedPCA) = sum(o.WCM)
mean(o::WeightedPCA) = mean(o.WCM)
var(o::WeightedPCA; kw...) = var(o.WCM; kw...)
std(o::WeightedPCA; kw...) = std(o.WCM; kw...)

Base.eltype(o::WeightedPCA{T}) where T = T
# PCA does not have a copy function
Base.copy(o::WeightedPCA) = WeightedPCA(copy(o.WCM), o.model)

function Base.:(==)(o1::T, o2::T) where {T<:WeightedPCA}
    nms = fieldnames(typeof(o1.WCM))
    all(getfield.(Ref(o1.WCM), nms) .== getfield.(Ref(o2.WCM), nms))
end
