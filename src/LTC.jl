module LTC

using Reexport

using Distributions
import NPZ: npzread
export npzread
using Juno
using DiffEqBase
using OrdinaryDiffEq
using DiffEqSensitivity
using DiffEqFlux
using DiffEqFlux: initial_params, paramlength, FastChain, FastDense, sciml_train
import DiffEqFlux: initial_params, paramlength, FastChain, FastDense, sciml_train
import DifferentialEquations: PresetTimeCallback, PeriodicCallback
export sciml_train
using GalacticOptim
using ModelingToolkit
using Zygote
using Zygote: @adjoint, Numeric, literal_getproperty, accum
export Zygote
using Flux: reset!, Zeros, Data.DataLoader
using Flux: Data.DataLoader
import Flux: reset!
export DataLoader
using IterTools: ncycle
export ncycle


rand_uniform(TYPE, lb,ub,dims...) = TYPE.(rand(Uniform(lb,ub),dims...))

#Zygote.@nograd rand_uniform, reshape

include("layers.jl")
include("mtk_recur.jl")
include("optimization.jl")
include("losses.jl")

include("ncp/ncp_sys_gen.jl")
include("ncp/wiring.jl")


export MTKRecur, MTKCell, Mapper, Broadcaster, get_bounds
export initial_params, paramlength
export reset_state!
export optimize
export loss_seq

export Wiring, FWiring, Net
end
