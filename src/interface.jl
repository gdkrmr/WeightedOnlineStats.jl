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
fit!(o::WeightedOnlineStat, xi::Missing, wi) = o
fit!(o::WeightedOnlineStat, xi, wi::Missing) = o


# The missing cases in x are dealt with in the dispatch of _fit!
function fit!(o::WeightedOnlineStat{VectorOb}, x::VectorOb, w::T) where {T <: Number}
    _fit!(o, x, w)
    return o
end
fit!(o::WeightedOnlineStat{VectorOb}, xi::VectorOb, wi::Missing) = o

function fit!(o::WeightedOnlineStat{I}, y, w::AbstractVector) where I
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
