function my_custom_train!(m, loss, ps, data, opt; data_range=nothing, lower=nothing, upper=nothing, cb=()->nothing)
  ps = Flux.Params(ps)
  for d in data

    Flux.reset!(m)

    x, y = d

    if data_range !== nothing
      seq_start = data_range[1]
      seq_end = length(data_range) == 2 ? data_range[2] : length(x)

      @views x = x[1:seq_end]
      @views y = y[1:seq_end]

      if seq_start != 1
        skipx = x[1:seq_start-1]
        m.(skipx)
        @views x = x[seq_start:end]
        @views y = y[seq_start:end]
      end
    end

    # back is a method that computes the product of the gradient so far with its argument.
    train_loss, back = Zygote.pullback(() -> loss(x,y), ps)
    cb(x,y,train_loss,m)
    # Insert whatever code you want here that needs training_loss, e.g. logging.
    # logging_callback(training_loss)
    # Apply back() to the correct type of 1.0 to get the gradient of loss.
    gs = back(one(train_loss))
    # Insert what ever code you want here that needs gradient.
    # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.


    Flux.Optimise.update!(opt, ps, gs)

    # Here you might like to check validation set accuracy, and break out to do early stopping.

    lower == nothing && continue
    upper == nothing && continue


    for (i,p) in enumerate(ps[1])
      ps[1][i] = max(lower[i],p)
      ps[1][i] = min(upper[i],p)
    end
  end
end
