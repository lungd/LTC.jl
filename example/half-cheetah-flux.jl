using LTC

include("half_cheetah_data_loader.jl")


function loss_model(x,y)
  Flux.reset!(model)

  x = Flux.unstack(x,3)
  y = Flux.unstack(y,3)

  ŷ = model.(x)

  sum(sum([(ŷ[i][end-1:end,:] .- y[i][end-1:end,:]) .^ 2 for i in 1:length(y)]))/length(y)
end


function logg(l)
  println(l)
  tx,ty = first(train_dl)
  tx = Flux.unstack(tx,3)
  ty = Flux.unstack(ty,3)
  pred = model.(tx)
  fig = plot([ŷ[end-1] for ŷ in pred])
  plot!(fig, [ŷ[end] for ŷ in pred])

  plot!(fig, [y[end-1] for y in ty])
  plot!(fig, [y[end] for y in ty])

  display(fig)
end



function my_custom_train!(loss, ps, data, opt; data_range=nothing, lower=nothing, upper=nothing, cb=()->nothing)
  ps = Params(ps)
  for d in data

    # back is a method that computes the product of the gradient so far with its argument.
    train_loss, back = Zygote.pullback(() -> loss(d...), ps)
    logg(train_loss)
    # Insert whatever code you want here that needs training_loss, e.g. logging.
    # logging_callback(training_loss)
    # Apply back() to the correct type of 1.0 to get the gradient of loss.
    gs = back(one(train_loss))
    # Insert what ever code you want here that needs gradient.
    # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.

    GalacticOptim.Flux.Optimise.update!(opt, ps, gs)

    # Here you might like to check validation set accuracy, and break out to do early stopping.

    lower == nothing && continue
    upper == nothing && continue


    for (i,p) in enumerate(ps[1])
        ps[1][i] = max(lower[i],ps[1][i])
        ps[1][i] = min(upper[i],ps[1][i])
    end
  end
end


train_dl, test_dl, valid_dl = get_dl(batchsize=32, seq_len=32)


model = NCP(Wiring(17,6; n_sensory=4, n_inter=2, n_command=0, n_motor=6))
y = model(rand(Float32,17,32))
@time y = model(rand(Float32,17,32))

@time logg(loss_model(first(train_dl)...))
@time logg(loss_model(first(train_dl)...))


opt = GalacticOptim.Flux.Optimiser(ClipValue(0.01), ADAM(0.001))

my_custom_train!((x,y)->loss_model(x,y),ps,train_dl,opt;lower,upper)
for i in 1:100
  my_custom_train!((x,y)->loss_model(x,y),ps,train_dl,opt;lower,upper)
end
