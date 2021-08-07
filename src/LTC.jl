module LTC

using Reexport

using Distributions
using DiffEqBase
using OrdinaryDiffEq
using DiffEqSensitivity
import DiffEqFlux: initial_params, paramlength, FastChain, FastDense, sciml_train
using GalacticOptim
using ModelingToolkit
using Flux
using NNlib: sigmoid

# using CatViews

# using DiffEqCallbacks
# import DiffEqCallbacks: SavedValues
import DiffEqCallbacks: PeriodicCallback, PresetTimeCallback
import LinearAlgebra: Diagonal
import DataInterpolations: ConstantInterpolation
using Random

rand_uniform(TYPE, lb,ub,dims...) = TYPE.(rand(Uniform(lb,ub),dims...))
rand_uniform(TYPE, lb,ub) = rand_uniform(TYPE, lb,ub,1)[1]

#Zygote.@nograd rand_uniform, reshape

include("layers.jl")
include("mtk_recur.jl")
include("mtk_recur_autonomous.jl")
include("mtk_node_tvp.jl")
include("mtk_node_p.jl")
include("mtk_node.jl")
include("optimization.jl")
include("losses.jl")
include("variables.jl")

include("data.jl")
export create_mini_batches

# include("mkt_sysstruct.jl")
# include("zygote.jl")

include("systems/systems.jl")
include("systems/ncp/ncp_sys_gen.jl")
include("systems/ncp/wiring.jl")


export MTKRecur, MTKCell, Mapper, Broadcaster, get_bounds
export MTKRecurAut, MTKCellAut
export initial_params, paramlength
export reset_state!
export optimize
export loss_seq

export Wiring, FWiring, Net
end
