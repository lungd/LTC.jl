import Flux: Data.DataLoader

generate_2d_data(T) = generate_data(T; stacked=false)
generate_3d_data(T) = generate_data(T; stacked=true)

function generate_data(T::DataType=Float32; stacked=true)
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

    data_x = stacked == true ? Flux.stack(data_x,2) : data_x
    data_y = stacked == true ? Flux.stack(data_y,2) : data_y
    batchsize = stacked == true ? 1 : N
    Flux.Data.DataLoader((data_x, data_y), batchsize=batchsize)
end
