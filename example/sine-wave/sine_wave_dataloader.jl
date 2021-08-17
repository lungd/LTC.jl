import Flux: Data.DataLoader

function generate_2d_arr_data(T::DataType=Float32)
    in_features = 2
    out_features = 1
    N = 48
    data_x = [sin.(range(0,stop=3π,length=N)), cos.(range(0,stop=3π,length=N))]
    data_x = [reshape([T(data_x[1][i]),T(data_x[2][i])],2,1) for i in 1:N]# |> f32
    data_y = [reshape([T(y)],1,1) for y in sin.(range(0,stop=6π,length=N))]# |> f32
    dl = Flux.Data.DataLoader((data_x, data_y), batchsize=N)
    @show length(dl)
    fx, fy = first(dl)
    @show size(fx)
    @show size(fx[1])
    @show size(fy)
    @show size(fy[1])
    fig = plot([x[1,1] for x in fx], label="x1")
    plot!(fig, [x[2,1] for x in fx], label="x2")
    plot!(fig, [y[1,1] for y in fy], label="y1")
    display(fig)
    dl
end

function generate_3d_data(T::DataType=Float32) #where T
  in_features = 2
  out_features = 1
  N = 48
  data_x = [sin.(range(0,stop=3π,length=N)), cos.(range(0,stop=3π,length=N))]
  data_x = [reshape([T(data_x[1][i]),T(data_x[2][i])],2,1) for i in 1:N]# |> f32
  data_y = [reshape([T(y)],1,1) for y in sin.(range(0,stop=6π,length=N))]# |> f32

  data_x = Flux.stack(data_x,2)
  data_y = Flux.stack(data_y,2)

  dl = Flux.Data.DataLoader((data_x, data_y), batchsize=1)
  @show length(dl)
  fx, fy = first(dl)
  @show size(fx)
  @show size(fx[1])
  @show size(fy)
  @show size(fy[1])
  fig = plot([x[1,1] for x in Flux.unstack(fx,2)], label="x1")
  scatter!(fig, [x[1,1] for x in Flux.unstack(fx,2)], label="x1")
  plot!(fig, [x[2,1] for x in Flux.unstack(fx,2)], label="x2")
  plot!(fig, [y[1,1] for y in Flux.unstack(fy,2)], label="y1")
  display(fig)
  dl
end
