mutable struct MyCallback{L,F,F2}
  losses::L
  cb::F
  ecb::F2
  nepochs::Int
  nsamples::Int
  iter::Int
  epoch::Int
  MyCallback(losses, cb, ecb, nepochs, nsamples, iter, epoch) =
    new{typeof(losses), typeof(cb), typeof(ecb)}(losses, cb, ecb, nepochs, nsamples, iter, epoch)
end

MyCallback(T::DataType=Float32;
              cb=(args...;kwargs...)->false, ecb=DEFAULT_ECB,
              nepochs=1, nsamples=1) =
  MyCallback(T[], cb, ecb, nepochs, nsamples, 0, 0)

function (mcb::MyCallback)(p,l,ŷ,y; kwargs...)
  cbout = invoke_sample_cb!(mcb, p,l,ŷ,y; kwargs...)
  mcb.iter % mcb.nsamples == 0 && invoke_epoch_cb!(mcb)
  return cbout
end

function reset!(mcb::MyCallback)
  mcb.losses = Vector{eltype(mcb.losses)}(undef, 0)
  mcb.iter = 0
end

function invoke_sample_cb!(mcb::MyCallback, p,l,ŷ,y; kwargs...)
  cbout = mcb.cb(p,l,ŷ,y,kwargs...)
  push!(mcb.losses, l)
  mcb.iter += 1
  return cbout
end

function invoke_epoch_cb!(mcb::MyCallback)
  mcb.epoch += 1
  mcb.ecb(mcb)
  reset!(mcb)
end

DEFAULT_ECB = (mcb::MyCallback) -> println("Epoch $(mcb.epoch) mean loss: $(mean(mcb.losses))")
