module LTC

using Reexport

@reexport using Flux
@reexport using DiffEqBase
@reexport using OrdinaryDiffEq
@reexport using DiffEqSensitivity
#using DiffEqCallbacks
using Distributions
#using BenchmarkTools
#using Plots
#using Noise
#@reexport using Functors
#export Functors
#using ComponentArrays
#using Parameters: @unpack
using NPZ
using Tullio
using Zygote
#using Sundials
#using ODEInterfaceDiffEq
#using RecursiveArrayTools

#using ComponentArrays
#using Parameters
#using UnicodePlots


#@reexport using Statistics
#@reexport using Zygote

export Flux
import NPZ: npzread
export npzread

#include("ltc-modelcleanup.jl")
include("wiring.jl")
include("ncp.jl")
include("custom_train.jl")

get_bounds(m::Function) = [],[]
get_bounds(m::Flux.Chain) = [reduce(vcat, [get_bounds(l)[1] for l in m.layers]), reduce(vcat, [get_bounds(l)[2] for l in m.layers])]
get_bounds(m::Flux.Dense) = [vcat([-1f0 for _ in 1:length(m.weight)], [-1f0 for _ in 1:length(m.bias)]), vcat([1f0 for _ in 1:length(m.weight)], [1f0 for _ in 1:length(m.bias)])]
export get_bounds

export Wiring, NCPWiring
export NCP, Mapper, get_bounds, my_custom_train!

end



# function Functors.functor(m::Mapper)
#   function reconstruct_Mapper(xs)
#     return Mapper(xs.W, xs.b)
#   end
#   return copy(ComponentArray(W=m.W, b=m.b)), reconstruct_Mapper
# end
#
# function Functors.functor(m::LTCCell)
#     function reconstruct_LTCCell(xs,mapin_re,mapout_re)
#         return LTCCell(m.wiring, mapin_re(xs.mapin_p), mapout_re(xs.mapout_p), m.sens_f, xs.sens_p, m.sens_pl, m.sens_re, m.syn_f, xs.syn_p, m.syn_pl, m.syn_re, xs.cm, xs.Gleak, xs.Eleak, m.state0)
#     end
#     mapin_p,mapin_re = Functors.functor(m.mapin)
#     mapout_p,mapout_re = Functors.functor(m.mapout)
#     return copy(ComponentArray(mapin_p = mapin_p, mapout_p = mapout_p, sens_p=m.sens_p, syn_p=m.syn_p, cm=m.cm, Gleak=m.Gleak, Eleak=m.Eleak,)), (x) -> reconstruct_LTCCell(x,mapin_re,mapout_re)
# end
#
# function Functors.functor(m::Flux.Recur)
#   function reconstruct_Recur(xs, cell_re)
#     return Flux.Recur(cell_re(xs.cell_p), xs.state)
#   end
#   cell_p, cell_re = Functors.functor(m.cell)
#   return copy(ComponentArray(cell_p=cell_p, state=m.state)), (xs) -> reconstruct_Recur(xs, cell_re)
# end
