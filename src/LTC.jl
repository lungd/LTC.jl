module LTC

using Reexport

# using LazyArrays
using Distributions
import NPZ: npzread
export npzread
using Juno
# using LoopVectorization
# using StructArrays
# using MacroTools: @forward

#using Flux
using DiffEqBase
using OrdinaryDiffEq
using DiffEqSensitivity
using DiffEqFlux
using DiffEqFlux: initial_params, paramlength, FastChain, FastDense, sciml_train
import DiffEqFlux: initial_params, paramlength, FastChain, FastDense, sciml_train
import DifferentialEquations: PresetTimeCallback, PeriodicCallback
export sciml_train
using GalacticOptim

# using Folds
# export Folds
# using ThreadsX
# export ThreadsX

#export initial_params
#using DiffRules
#using RecursiveArrayTools
#import RecursiveArrayTools: DiffEqArray
#using DiffEqCallbacks
#using DistributionsAD
using ModelingToolkit
#using BenchmarkTools
#using Plots
#using Noise
#@reexport using Functors
#export Functors
#using ComponentArrays
#using Parameters: @unpack
# using Tullio
using Zygote
using Zygote: @adjoint, Numeric, literal_getproperty, accum
export Zygote
using Flux: reset!, Zeros, Data.DataLoader
using Flux: Data.DataLoader
import Flux: reset!
export DataLoader
#export reset!
# using ReverseDiff
# using ForwardDiff
using IterTools: ncycle
export ncycle
#using ZygoteRules
#using ForwardDiff
#using Sundials
#using ODEInterfaceDiffEq


#@reexport using Statistics


rand_uniform(TYPE, lb,ub,dims...) = TYPE.(rand(Uniform(lb,ub),dims...))

#Zygote.@nograd rand_uniform, reshape

#include("ltc-modelcleanup.jl")
include("wiring.jl")
#include("ncp_ensemble.jl")
#include("ncp.jl")
#include("_ncp.jl")

# include("ltcnet.jl")

# include("ltcnet_MTK.jl")
include("mtk_recur.jl")
include("problem_generator.jl")

#include("_ncp_1.jl")
#include("new_ncp.jl")
#include("new_ncp_ensemble.jl")
# include("custom_train.jl")
#include("tullio-sigmoid.jl")
# include("param_ranges.jl")

include("optimization.jl")
export optimize
export get_bounds


export Wiring, NCPWiring
export MTKRecur, MTKCell, Net, Mapper, Broadcaster
# export LTCNet, Mapper, Broadcaster, my_custom_train!
export reset!, reset_state!
export initial_params, paramlength
end
