# using MLDataUtils



function create_rnn_sequence(sequence)
  sequence[:, 1:end-1], sequence[:, 2:end]
end


# One sequence
function create_mini_batches(sequence, t; batch_length=20, batchsize=1, stride=1)
  # TODO: split train (,val), test
  xs = Vector{typeof(sequence)}()
  ts = Vector{typeof(t)}()
  @show size(t)
  for i in 1:stride:size(sequence,2)-batch_length+1
    push!(xs, sequence[:, i:i+batch_length-1])
    push!(ts, t[:, i:i+batch_length-1])
  end
  Flux.Data.DataLoader(((xs,ts)), batchsize=batchsize, shuffle=true)
end


function create_mini_batch_sequence(sequence, t; batch_length=20, batchsize=1, stride=1)
  # TODO: split train (,val), test
  xs = Vector{typeof(sequence)}[]
  ts = Vector{typeof(t)}[]
  @show size(t)
  for i in 1:stride:size(sequence,2)-batch_length+1
    push!(xs, [sequence[:, i:i+batch_length-1][:, j:j] for j in 1:batch_length])
    push!(ts, [t[:,i:i+batch_length-1][:,j:j] for j in 1:batch_length])
  end
  Flux.Data.DataLoader((xs,ts), batchsize=1, shuffle=true)
end



# dl = create_mini_batchex(rand(17,100), collect(1.0:100), batchsize=1)
# fdl = first(dl)
#
# for dlw in dl
#   x, t = dlw
#   display(plot(t[1],x[1][1,:]))
# end

# Multiple sequences
# function create_mini_batches(sequence::Vector{AbstractArray}, t; k=30, epochs=10)
#   train, val_X, val_Y = kfolds((sequence, t))
#   batches = eachbatch(train, size=k)
#   ncycle(eachbatch(train[1], k), epochs)
# end
