module LTC

using Reexport

using LazyArrays
using Distributions
import NPZ: npzread
export npzread
using Juno
using LoopVectorization
using StructArrays
using MacroTools: @forward

using Flux
using DiffEqBase
using OrdinaryDiffEq
using DiffEqSensitivity
#using DiffEqFlux
#import DiffEqFlux: initial_params, paramlength, FastChain, FastDense
#using GalacticOptim
#export initial_params
#using DiffRules
#using RecursiveArrayTools
#import RecursiveArrayTools: DiffEqArray
#using DiffEqCallbacks
#using DistributionsAD
# using ModelingToolkit
#using BenchmarkTools
#using Plots
#using Noise
#@reexport using Functors
#export Functors
#using ComponentArrays
#using Parameters: @unpack
using Tullio
using Zygote
using Zygote: @adjoint, Numeric, literal_getproperty, accum
# using ReverseDiff

#using ZygoteRules
#using ForwardDiff
#using Sundials
#using ODEInterfaceDiffEq


#@reexport using Statistics

export Flux


rand_uniform(TYPE, lb,ub,dims...) = TYPE.(rand(Uniform(lb,ub),dims...))# |> f64

#Zygote.@nograd rand_uniform, reshape

#include("ltc-modelcleanup.jl")
include("wiring.jl")
#include("ncp_ensemble.jl")
#include("ncp.jl")
#include("_ncp.jl")
include("ltcnet.jl")
#include("_ncp_1.jl")
#include("new_ncp.jl")
#include("new_ncp_ensemble.jl")
include("custom_train.jl")
#include("tullio-sigmoid.jl")
# include("param_ranges.jl")

export get_bounds


export Wiring, NCPWiring
export LTCNet, Mapper, gNN, get_bounds, my_custom_train!


end

