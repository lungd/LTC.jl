abstract type WiringT <: Function end

struct Wiring <: WiringT
    n_in::Int
    out::Int
    n_sensory::Int
    n_inter::Int
    n_command::Int
    n_motor::Int
    n_total::Int

    sensory_out::Int
    inter_out::Int
    rec_command_out::Int
    motor_in::Int

    sens_mask::Matrix{Float32}
    syn_mask::Matrix{Float32}

    sens_pol::Matrix{Float32}
    syn_pol::Matrix{Float32}

    # function Wiring(n_in,out,n_sensory,n_inter,n_command,n_motor,n_total,sensory_out,inter_out,rec_command_out,motor_in,sens_mask,syn_mask,sens_pol,syn_pol)
    #     new{typeof(sens_mask),typeof(syn_mask),typeof(sens_pol),typeof(syn_pol)}(
    #                 n_in,out,n_sensory,n_inter,n_command,n_motor,n_total,sensory_out,inter_out,rec_command_out,motor_in,sens_mask,syn_mask,sens_pol,syn_pol)
    # end
end

get_n_in(m::Wiring) = m.n_in
get_n_total(m::Wiring) = m.n_total

function Wiring(in::Int, out::Int;
                n_sensory=2, n_inter=5, n_command=0, n_motor=1,
                sensory_in=-1, rec_sensory=-1, sensory_out=-1,
                rec_inter=-1, inter_out=-1,                       # inter_in = sensory_out
                rec_command=-1, command_out=-1,                   # command_in = inter_out
                rec_motor=-1)                                     # motor_in = command_out, motor_out = out
  sensory_s = 1
  inter_s   = n_sensory + 1
  command_s = n_sensory + n_inter + 1
  motor_s   = n_sensory + n_inter + n_command + 1
  n_total   = n_sensory + n_inter + n_command + n_motor

  sens_mask = ones(Float32, in, n_total)
  syn_mask  = ones(Float32, n_total, n_total)

  sens_pol = ones(Float32, in, n_total)
  syn_pol  = ones(Float32, n_total, n_total)
  for i in eachindex(sens_pol)
    sens_pol[i] = [-1,1,1][rand(1:3)]
  end
  for i in eachindex(syn_pol)
    syn_pol[i] = [-1,1,1][rand(1:3)]
  end

  Wiring(in,out,n_sensory, n_inter,n_command,n_motor,n_total,sensory_out,inter_out,rec_command,command_out,sens_mask,syn_mask,sens_pol,syn_pol)
end



function add_synapses!(srcs,dsts,n_out,bitmask,polmask)
    n_out == 0 && return
    n_out = n_out == -1 ? length(dsts) : n_out
    n_out = min(n_out,length(dsts))
    for src in srcs
        for _ in n_out
            dst = dsts[rand(1:length(dsts))]
            bitmask[src,dst] = 1
            polmask[src,dst] = [-1,1,1][rand(1:3)]
        end
    end
end

function add_missing_synapses!(srcs, dsts, bitmask, polmask; n=1)
    for dst in dsts
        bm = @view bitmask[:,dst]
        pm = @view polmask[:,dst]

        if 1 ∉ filter(x -> x != dst, bm[srcs])
        #if 1 ∉ bm[srcs][bm[srcs] .!= dst]
            for i in 1:n
                random_src = srcs[rand(1:length(srcs))]
                bm[random_src] = 1
                pm[random_src] = [-1,1,1][rand(1:3)]
            end
        end
    end
end


function NCPWiring(in::Int, out::Int;
                n_sensory=2, n_inter=3, n_command=5, n_motor=1,
                sensory_in=-1, rec_sensory=0, sensory_inter=2, sensory_command=0, sensory_motor=0,
                inter_in=0, rec_inter=2, inter_command=2, inter_motor=0,                       # inter_in = sensory_out
                command_in=0, rec_command=1, command_motor=3,                   # command_in = inter_out
                motor_in=0, rec_motor=1)                                     # motor_in = command_out, motor_out = out
  sensory_s = 1
  inter_s   = n_sensory + 1
  command_s = n_sensory + n_inter + 1
  motor_s   = n_sensory + n_inter + n_command + 1
  n_total   = n_sensory + n_inter + n_command + n_motor

  out = min(n_total, out)

  sens_mask = zeros(Float32, in, n_total)
  syn_mask  = zeros(Float32, n_total, n_total)
  sens_pol = ones(Float32, in, n_total)
  syn_pol  = ones(Float32, n_total, n_total)



  input_range = range(1, length=in)
  sens_range = range(sensory_s, length=n_sensory)
  inter_range = range(inter_s, length=n_inter)
  command_range = range(command_s, length=n_command)
  motor_range = range(motor_s, length=n_motor)

  add_synapses!(input_range,    sens_range,    sensory_in, sens_mask, sens_pol)
  add_synapses!(input_range,    inter_range,    inter_in, sens_mask, sens_pol)
  add_synapses!(input_range,    command_range,    command_in, sens_mask, sens_pol)
  add_synapses!(input_range,    motor_range,    motor_in, sens_mask, sens_pol)

  add_synapses!(sens_range,    sens_range,    rec_sensory, syn_mask, syn_pol)
  add_synapses!(sens_range,    inter_range,   sensory_inter, syn_mask, syn_pol)
  add_synapses!(sens_range,    command_range, sensory_command, syn_mask, syn_pol)
  add_synapses!(sens_range,    motor_range,   sensory_motor, syn_mask, syn_pol)
  add_synapses!(inter_range,   inter_range,   rec_inter, syn_mask, syn_pol)
  add_synapses!(inter_range,   command_range, inter_command, syn_mask, syn_pol)
  add_synapses!(inter_range,   motor_range,   inter_motor, syn_mask, syn_pol)
  add_synapses!(command_range, command_range, rec_command, syn_mask, syn_pol)
  add_synapses!(command_range, motor_range,   command_motor, syn_mask, syn_pol)
  add_synapses!(motor_range,   motor_range,   rec_motor, syn_mask, syn_pol)

  add_missing_synapses!(sens_range, inter_range, syn_mask, syn_pol)
  add_missing_synapses!(inter_range, command_range, syn_mask, syn_pol)
  add_missing_synapses!(command_range, motor_range, syn_mask, syn_pol)

  Wiring(in,out,n_sensory, n_inter,n_command,n_motor,n_total,sensory_inter,inter_command,rec_command,command_motor,sens_mask,syn_mask,sens_pol,syn_pol)
end


Flux.trainable(m::Wiring) = ()
