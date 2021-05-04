get_bounds(m::Function) = [],[]
get_bounds(m::Flux.Chain) = [reduce(vcat, [get_bounds(l)[1] for l in m.layers]), reduce(vcat, [get_bounds(l)[2] for l in m.layers])]
function get_bounds(m::Flux.Dense)
  lb = [[-20.0 for _ in 1:length(m.weight)]...,
        [-20.0 for _ in 1:length(m.bias)]...] |> f32
  ub = [[20.0 for _ in 1:length(m.weight)]...,
        [20.0 for _ in 1:length(m.bias)]...] |> f32
  return lb, ub
end

function get_bounds(m::DiffEqFlux.FastChain)
  lb = vcat([get_bounds(layer)[1] for layer in m.layers]...)
  ub = vcat([get_bounds(layer)[2] for layer in m.layers]...)
  return lb, ub
end

function get_bounds(m::DiffEqFlux.FastDense)
  lb = [[-2.0 for _ in 1:m.out*m.in]...,
        [-2.0 for _ in 1:m.out]...] |> f32
  ub = [[2.0 for _ in 1:m.out*m.in]...,
        [2.0 for _ in 1:m.out]...] |> f32
  return lb, ub
end


get_bounds(m::Flux.Recur) = get_bounds(m.cell)
#get_bounds(m::MyRecur) = get_bounds(m.cell)


function get_bounds(m::Mapper)
  lb = [[-20.1 for i in 1:length(m.W)]...,
        [-20.1 for i in 1:length(m.b)]...] |> f32

  ub = [[20.1 for i in 1:length(m.W)]...,
        [20.1 for i in 1:length(m.b)]...] |> f32
  return lb, ub
end

# function get_bounds(m::gNN)
#   lb = [[0 for _ in m.G]...,          #
#         [0.1 for _ in m.μ]...,
#         [1 for _ in m.σ]...,
#         [-2 for _ in m.E]...,] |> f32
#
#   ub = [[1 for _ in m.G]...,          #
#         [0.9 for _ in m.μ]...,
#         [10 for _ in m.σ]...,
#         [2 for _ in m.E]...,] |> f32
#   return lb, ub
# end
function get_bounds(m::gNN)
  lb = [[0.1 for _ in m.μ]...,
        [1 for _ in m.σ]...,
        [0.001 for _ in m.G]...,
        [-1 for _ in m.E]...,
        ] |> f32

  ub = [[0.9 for _ in m.μ]...,
        [10 for _ in m.σ]...,
        [1 for _ in m.G]...,
        [1 for _ in m.E]...,
        ] |> f32
  return lb, ub
end

# function get_bounds(m::LTCCell)
#   lower = [
#       get_bounds(m.mapin)[1]...,
#       get_bounds(m.mapout)[1]...,
#       get_bounds(m.sens_f)[1]...,
#       get_bounds(m.syn_f)[1]...,
#       [0.9 for _ in m.cm]...,               #
#       [0.01 for _ in m.Gleak]...,            #
#       [-2 for _ in m.Eleak]...,
#       [0.001 for _ in m.state0]...,
#   ] |> f32
#   upper = [
#     get_bounds(m.mapin)[2]...,
#     get_bounds(m.mapout)[2]...,
#     get_bounds(m.sens_f)[2]...,
#     get_bounds(m.syn_f)[2]...,
#     [4 for _ in m.cm]...,
#     [2.0 for _ in m.Gleak]...,
#     [2 for _ in m.Eleak]...,
#     [0.1 for _ in m.state0]...,
#   ] |> f32
#
#   lower, upper
# end

#function get_bounds(m::LTCCell)
#  lower = [
#      get_bounds(m.mapin)[1]...,
#      get_bounds(m.mapout)[1]...,
#      #reduce(vcat,[get_bounds(s)[1] for s in m.sens.synapses])...,
#      #reduce(vcat,[get_bounds(s)[1] for s in m.syns.synapses])...,
#      get_bounds(m.sens_f)[1]...,
#      get_bounds(m.syn_f)[1]...,
#      [0.001 for _ in m.cm]...,               #
#      [0.01 for _ in m.Gleak]...,            #
#      [-2 for _ in m.Eleak]...,
#      [-2 for _ in m.Esens]...,
#      [0 for _ in m.Gsens]...,
#      [-2 for _ in m.Esyn]...,
#      [0 for _ in m.Gsyn]...,
#      [-0.5 for _ in m.state0]...,
#  ] |> f32
#  upper = [
#    get_bounds(m.mapin)[2]...,
#    get_bounds(m.mapout)[2]...,
#    #reduce(vcat,[get_bounds(s)[2] for s in m.sens.synapses])...,
#    #reduce(vcat,[get_bounds(s)[2] for s in m.syns.synapses])...,
#    get_bounds(m.sens_f)[2]...,
#    get_bounds(m.syn_f)[2]...,
#    [5 for _ in m.cm]...,
#    [10.0 for _ in m.Gleak]...,
#    [1 for _ in m.Eleak]...,
#    [2 for _ in m.Esens]...,
#    [1 for _ in m.Gsens]...,
#    [2 for _ in m.Esyn]...,
#    [1 for _ in m.Gsyn]...,
#    [0.5 for _ in m.state0]...,
#  ] |> f32
#
#  lower, upper
#end


function get_bounds(m::LTCCell)
  lower = [
      get_bounds(m.mapin)[1]...,
      get_bounds(m.mapout)[1]...,
      reduce(vcat,[get_bounds(s)[1] for s in m.sens.synapses])...,
      reduce(vcat,[get_bounds(s)[1] for s in m.syns.synapses])...,
      #get_bounds(m.sens_f)[1]...,
      #get_bounds(m.syn_f)[1]...,
      [0.001 for _ in m.cm]...,               #
      [0.01 for _ in m.Gleak]...,            #
      [-2 for _ in m.Eleak]...,
      #[-2 for _ in m.Esens]...,
      #[0 for _ in m.Gsens]...,
      #[-2 for _ in m.Esyn]...,
      #[0 for _ in m.Gsyn]...,
      [-50 for _ in m.state0]...,
  ] |> f32
  upper = [
    get_bounds(m.mapin)[2]...,
    get_bounds(m.mapout)[2]...,
    reduce(vcat,[get_bounds(s)[2] for s in m.sens.synapses])...,
    reduce(vcat,[get_bounds(s)[2] for s in m.syns.synapses])...,
    #get_bounds(m.sens_f)[2]...,
    #get_bounds(m.syn_f)[2]...,
    [5 for _ in m.cm]...,
    [10.0 for _ in m.Gleak]...,
    [-0.1 for _ in m.Eleak]...,
    #[2 for _ in m.Esens]...,
    #[1 for _ in m.Gsens]...,
    #[2 for _ in m.Esyn]...,
    #[1 for _ in m.Gsyn]...,
    [20 for _ in m.state0]...,
  ] |> f32

  lower, upper
end







# function get_bounds(m::NCP)
#   lower = [
#       get_bounds(m.mapin)[1]...,
#       [0.9 for _ in m.ltc.cell.cm]...,               #
#       [0.0009 for _ in m.ltc.cell.Gleak]...,            #
#       [-2 for _ in m.ltc.cell.Eleak]...,
#       get_bounds(m.ltc.cell.sens)[1]...,
#       get_bounds(m.ltc.cell.syn)[1]...,
#       [-0.1 for _ in m.ltc.cell.state0]...,
#       get_bounds(m.mapout)[1]...,
#   ] |> f32
#   upper = [
#     get_bounds(m.mapin)[2]...,
#     [4 for _ in m.ltc.cell.cm]...,
#     [2.0 for _ in m.ltc.cell.Gleak]...,
#     [2 for _ in m.ltc.cell.Eleak]...,
#     get_bounds(m.ltc.cell.sens)[2]...,
#     get_bounds(m.ltc.cell.syn)[2]...,
#     [0.2 for _ in m.ltc.cell.state0]...,
#     get_bounds(m.mapout)[2]...,
#   ] |> f32
#
#   lower, upper
# end
