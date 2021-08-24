module LTC

using Reexport

using Distributions
using DiffEqBase
using OrdinaryDiffEq
using DiffEqSensitivity
import DiffEqFlux: initial_params, paramlength, FastChain, FastDense, sciml_train
using GalacticOptim
using Zygote
using ModelingToolkit
using Flux
using NNlib: sigmoid
using ForwardDiff
import LinearAlgebra: Diagonal
import DataInterpolations: ConstantInterpolation, LinearInterpolation
using Random

import IterTools: ncycle

rand_uniform(T::DataType, lb,ub,dims...) = T.(rand(Uniform(lb,ub),dims...))
rand_uniform(T::DataType, lb,ub) = rand_uniform(T, lb,ub,1)[1]
rand_uniform(lb,ub,dims...) = rand_uniform(Float32, lb,ub,dims...)

add_dim(x::Array{T, N}) where {T,N} = reshape(x, Val(N+1))



include("systems/ncp/wiring.jl")

include("layers.jl")
include("mtk_recur.jl")
include("mtk_node.jl")
include("optimization.jl")
include("losses.jl")
include("callback.jl")
include("variables.jl")

include("data.jl")
include("utils.jl")
# include("mkt_sysstruct.jl")
# include("zygote.jl")

include("systems/systems.jl")
include("systems/ncp/ncp_sys_gen.jl")



export get_bounds
export initial_params, paramlength
export reset_state!
export optimize
export loss_seq

export Wiring, FWiring, Net
end
