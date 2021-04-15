using LTC
using BenchmarkTools
using Zygote

function test()

  function generate_data()
      in_features = 2
      out_features = 1
      N = 48
      data_x = [sin.(range(0,stop=3π,length=N)), cos.(range(0,stop=3π,length=N))]
      data_x = [reshape([Float32(data_x[1][i]),Float32(data_x[2][i])],2,1) for i in 1:N]# |> f32
      data_y = [reshape([Float32(y)],1) for y in sin.(range(0,stop=6π,length=N))]# |> f32

      data_x = [repeat(x,1,20) for x in data_x]
      data_y = [repeat(x,1,20) for x in data_y]

      data_x, data_y
  end

  function lossf(x,y)
    Flux.reset!(ltc)
    ŷ = ltc.(x)
    sum(sum([(ŷ[i] .- y[i]) .^ 2 for i in 1:length(y)]))/length(y)#, ŷ
  end

  data_x,data_y = generate_data()

  ltc = NCP(Wiring(2,1))

  θ = Flux.params(ltc)

  @time gs = Zygote.gradient(θ) do
    lossf(data_x,data_y)#[1]
  end
  @time gs = Zygote.gradient(θ) do
    lossf(data_x,data_y)#[1]
  end
end
@time test()
