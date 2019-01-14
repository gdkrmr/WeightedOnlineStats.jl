abstract type WeightedOnlineStat{T} <: OnlineStat{T} end

meanweight(o::WeightedOnlineStat) = o.W
weightsum(o::WeightedOnlineStat) = meanweight(o) * nobs(o)
Base.eltype(o::WeightedOnlineStat{T}) where T = T

##############################################################
# Define our own interface so that it accepts two inputs.
##############################################################

function OnlineStatsBase.fit!(o::WeightedOnlineStat, xi::Number, wi::Number)
    _fit!(o, xi, wi)
    return o
end
OnlineStatsBase.fit!(o::WeightedOnlineStat, xi::Missing, wi::Number) = o
OnlineStatsBase.fit!(o::WeightedOnlineStat, xi::Number, wi::Missing) = o


# The missing cases in x are dealt with in the dispatch of _fit!
function OnlineStatsBase.fit!(o::WeightedOnlineStat{VectorOb}, x::VectorOb, w::Number)
    _fit!(o, x, w)
    return o
end
OnlineStatsBase.fit!(o::WeightedOnlineStat{VectorOb}, xi::VectorOb, wi::Missing) = o

function OnlineStatsBase.fit!(o::WeightedOnlineStat, y, w::AbstractVector)
    for i in 1:length(w)
        fit!(o, y[i], w[i])
    end
    o
end


OnlineStatsBase.fit!(o::WeightedOnlineStat, x::TwoThings) = fit!(o, x[1], x[2])
OnlineStatsBase.fit!(o::WeightedOnlineStat{VectorOb}, x::AbstractMatrix, w::AbstractVector) =
    fit!(o, eachrow(x), w)

function OnlineStatsBase.merge!(o::WeightedOnlineStat, o2::WeightedOnlineStat)
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
