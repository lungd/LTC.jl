module LTC

using Reexport

using Flux
using DiffEqBase
using OrdinaryDiffEq
using DiffEqSensitivity
using DiffEqCallbacks
using Distributions
using NPZ
using BenchmarkTools
using Plots
using Noise
@reexport using Functors
export Functors
using ComponentArrays
using Parameters: @unpack
using NPZ
using Tullio
using Sundials
using ODEInterfaceDiffEq

#using ComponentArrays
using Parameters
#using UnicodePlots

import Flux: OneHotArray, params

#@reexport using Statistics
@reexport using Flux, Flux.Zygote


#include("ltc-modelcleanup.jl")
include("wiring.jl")
include("ncp.jl")


export Wiring, NCPWiring
export NCP, Mapper, get_bounds

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
