# using Revise
# using Pkg
# cd("..")
# Pkg.activate(;temp=true)
# Pkg.add("MultivariateStats")
# Pkg.add("OnlineStatsBase")
# Pkg.add("OnlineStats")
# Pkg.add("StatsBase")
# Pkg.develop(path=".")
# using Revise

using Test
using WeightedOnlineStats
import OnlineStatsBase: eachcol, eachrow
using StatsBase
import OnlineStats: Extrema
using Statistics
using Random
using MultivariateStats


Random.seed!(124)
l = 1000
x = rand(l);
xmis = convert(Array{Union{Float64,Missing}}, x);
xmis[end] = missing
xmis[end-1] = missing
d = 5
x2 = rand(l, d)
x2mis = convert(Array{Union{Float64,Missing}}, x2)
x2mis[end, 1] = missing
x2mis[end-1, 1] = missing
x2mis[end-2, :] .= missing
x2mis[end-3, :] .= missing
w = rand(l);
wmis = convert(Array{Union{Float64,Missing}}, w);
wmis[end] = missing
wmis[end-2] = missing
wmis[end-4] = missing

Base.isapprox(x::Tuple, y::Tuple) =
    reduce(&, map((a, b) -> a ≈ b, x, y))

missing_to_nan(x::Array{Union{Missing,T}}) where {T} = map(y -> y === missing ? T(NaN) : y, x)

function test_fit(
    T::Type{<:WeightedOnlineStats.WeightedOnlineStat{S}},
    data, weights, unpack_fun, jfun
) where {S}
    l = length(data)
    @assert l == length(weights)

    o = T()
    for (xi, wi) in zip(data, weights)
        fit!(o, xi, wi)
    end

    @test unpack_fun(o) ≈ jfun(data, weights)
    @test eltype(unpack_fun(o)) == eltype(o)

    i_nonmissing = .!(ismissing.(data) .| ismissing.(weights))
    l_nonmissing = sum(i_nonmissing)
    wsum_nonmissing = sum(weights[i_nonmissing])

    @test nobs(o) == l_nonmissing
    @test weightsum(o) ≈ wsum_nonmissing
end

include("test_hist.jl")
include("test_sum.jl")
include("test_mean.jl")
include("test_var.jl")
include("test_cov.jl")
include("test_pca.jl")
