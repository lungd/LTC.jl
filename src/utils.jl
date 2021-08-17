function print_cell_info(cell, train_u0)
  println("--------------- MTK Cell ---------------")
  println("in:                         $(cell.wiring.n_in)")
  println("out:                        $(cell.wiring.n_out)")
  println("# neurons:                  $(cell.wiring.n_total)")
  println("# states:                   $(size(cell.state0,1))")
  println("# input-neuron synapses:    $(Int(sum(cell.wiring.sens_mask)))")
  println("# neuron-neuron synapses:   $(Int(sum(cell.wiring.syn_mask)))")
  println("# params:                   $(length(cell.p))")
  println("# train states:             $(train_u0)")
end

# @show param_names
# @show prob.u0
# @show size(state0)
# @show prob.f.syms
# @show length(prob.p)
# @show length(prob.p)
# @show input_idxs
# @show outpins
# @show length(p)
#
# @show typeof(p_ode)
# @show typeof(prob.u0)
# @show eltype(p_ode)
# @show eltype(prob.u0)
