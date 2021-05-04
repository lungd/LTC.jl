function my_custom_train!(m, loss, ps, data, opt; data_range=nothing, lower=nothing, upper=nothing, cb=()->nothing)
  # training_loss is declared local so it will be available for logging outside the gradient calculation.
  local training_loss
  ps = Zygote.Params(ps)
  for d in data
    gs = Zygote.gradient(ps) do
      training_loss = loss(d...)
      # Code inserted here will be differentiated, unless you need that gradient information
      # it is better to do the work outside this block.
      return training_loss
    end
    cb(d..., training_loss, m)
    # Insert whatever code you want here that needs training_loss, e.g. logging.
    # logging_callback(training_loss)
    # Insert what ever code you want here that needs gradient.
    # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
    Flux.Optimise.update!(opt, ps, gs)
    # Here you might like to check validation set accuracy, and break out to do early stopping.
    for i = 1:length(lower)
      # clamp!(ps[1][i], lower[i], upper[i])
      ps[1][i] = max(lower[i],ps[1][i])
      ps[1][i] = min(upper[i],ps[1][i])
    end
  end
end



function my_custom_train2!(m, loss, ps, data, opt; data_range=nothing, lower=nothing, upper=nothing, cb=()->nothing)
  ps = Flux.Params(ps)
  #losses = []
  for d in data

    #Flux.reset!(m)

    x, y = d[1], d[2]

    # if data_range !== nothing
    #   seq_start = data_range[1]
    #   seq_end = length(data_range) == 2 ? data_range[2] : length(x)
    #
    #   @views x = x[1:seq_end]
    #   @views y = y[1:seq_end]
    #
    #   if seq_start != 1
    #     skipx = x[1:seq_start-1]
    #     m.(skipx)
    #     @views x = x[seq_start:end]
    #     @views y = y[seq_start:end]
    #   end
    # end

    # back is a method that computes the product of the gradient so far with its argument.
    train_loss, back = Zygote.pullback(() -> loss(x,y), ps)
    #push!(losses, train_loss)
    cb(x,y,train_loss,m)
    # Insert whatever code you want here that needs training_loss, e.g. logging.
    # logging_callback(training_loss)
    # Apply back() to the correct type of 1.0 to get the gradient of loss.
    gs = back(one(train_loss))
    # Insert what ever code you want here that needs gradient.
    # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.

    Flux.Optimise.update!(opt, ps, gs)

    # Here you might like to check validation set accuracy, and break out to do early stopping.

    # lower == nothing && continue
    # upper == nothing && continue
    #
    #
    for (i,p) in enumerate(ps[1])
      # clamp!(ps[i], lower[i], upper[i])
      ps[1][i] = max(lower[i],p)
      ps[1][i] = min(upper[i],p)
    end
  end
  #x,y = first(data)
  #cb(x,y,sum(losses)/length(losses),m)
end
