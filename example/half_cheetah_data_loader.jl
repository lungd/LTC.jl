# one batch: vector of matrices with shape (features,batchsize)
# [rand(features,batchsize) for 1:seq_len]



function get_dl(; seq_len=32, batchsize=16)
    filepath = joinpath(@__DIR__, "half-cheetah-data")
    @show filepath
    data_dir = filepath

    obs_size = size(npzread("$(data_dir)/trace_0000.npy"),2) # 17

    all_files = readdir(data_dir)
    all_files = ["$(data_dir)/$(f)" for f in all_files]
    train_files = all_files[16:25]
    test_files = all_files[6:15]
    valid_files = all_files[1:5]


    # goal:
    # size(first(train_dl)[1]) == [(17,16) for 1:32]
    # x_data == [ [(17,16) for 1:32] for 1:60 ]



    # sequences = (32,17,100)
    # sequences = [(32,17) for 1:100]
    #




    train_x, train_y = _load_files(train_files, seq_len)
    test_x, test_y = _load_files(test_files, seq_len)
    valid_x, valid_y = _load_files(valid_files, seq_len)

    # (970×32×17)
    train_x = permutedims(train_x,(3,1,2))
    train_y = permutedims(train_y,(3,1,2))
    test_x = permutedims(test_x,(3,1,2))
    test_y = permutedims(test_y,(3,1,2))
    valid_x = permutedims(valid_x,(3,1,2))
    valid_y = permutedims(valid_y,(3,1,2))
    # (17x970x32)


    train_x_new = train_y_new = Vector{Matrix{Float32}}[]
    test_x_new = test_y_new = Vector{Matrix{Float32}}[]
    valid_x_new = valid_y_new = Vector{Matrix{Float32}}[]
    # train_x_new = train_y_new = []
    # test_x_new = test_y_new = []
    # valid_x_new = valid_y_new = []
    for i in 1:batchsize:size(train_x,2)-batchsize-1
        push!(train_x_new, Flux.unstack(train_x[:,i:i+batchsize-1,:],3))
        push!(train_y_new, Flux.unstack(train_y[:,i+1:i+batchsize,:],3))
        # push!(train_x_new, train_x[:,i:i+batchsize-1,:])
        # push!(train_y_new, train_y[:,i+1:i+batchsize,:])
    end
    for i in 1:batchsize:size(test_x,2)-batchsize-1
        push!(test_x_new, Flux.unstack(test_x[:,i:i+batchsize-1,:],3))
        push!(test_y_new, Flux.unstack(test_y[:,i+1:i+batchsize,:],3))
        # push!(test_x_new, test_x[:,i:i+batchsize-1,:])
        # push!(test_y_new, test_y[:,i+1:i+batchsize,:])
    end
    for i in 1:batchsize:size(valid_x,2)-batchsize-1
        push!(valid_x_new, Flux.unstack(valid_x[:,i:i+batchsize-1,:],3))
        push!(valid_y_new, Flux.unstack(valid_y[:,i+1:i+batchsize,:],3))
        # push!(valid_x_new, valid_x[:,i:i+batchsize-1,:])
        # push!(valid_y_new, valid_y[:,i+1:i+batchsize,:])
    end

    # train_x_new = Flux.stack(train_x_new,4)
    # train_y_new = Flux.stack(train_y_new,4)
    # test_x_new = Flux.stack(test_x_new,4)
    # test_y_new = Flux.stack(test_y_new,4)
    # valid_x_new = Flux.stack(valid_x_new,4)
    # valid_y_new = Flux.stack(valid_y_new,4)

    #train_x = Flux.unstack(train_x,3)

    # train_dl = Flux.Data.DataLoader((train_x,train_y),batchsize=1)
    # test_dl = Flux.Data.DataLoader((test_x,test_y),batchsize=batchsize)
    # valid_dl = Flux.Data.DataLoader((valid_x,valid_y),batchsize=batchsize)
    # train_dl = Flux.Data.DataLoader((train_x_new,train_y_new),batchsize=1)
    # test_dl = Flux.Data.DataLoader((test_x_new,test_y_new),batchsize=1)
    # valid_dl = Flux.Data.DataLoader((valid_x_new,valid_y_new),batchsize=1)

    # @show size(train_x_new)
    # @show size(train_x_new[1])
    # @show size(train_x_new[1][1])
    #
    # @show size(train_x_new[2])
    # @show size(train_x_new[2][1])

    train_dl = zip(train_x_new, test_y_new) |> f32
    test_dl = zip(test_x_new, test_y_new) |> f32
    valid_dl = zip(valid_x_new, valid_y_new) |> f32

    # train_dl = (train_x_new, test_y_new)
    # test_dl = (test_x_new, test_y_new)
    # valid_dl = (valid_x_new, valid_y_new)

    return train_dl, test_dl, valid_dl
end


function _load_files(files, seq_len)
    all_x = []
    all_y = []
    for f in files
        arr = Float32.(npzread(f))
        x, y = _cut_in_sequences(arr, seq_len, 10)
        push!(all_x, x...)
        push!(all_y, y...)
    end

    return Flux.stack(all_x,1), Flux.stack(all_y,1)
    #return all_x, all_y
end

function _cut_in_sequences(x, seq_len, inc=1)
    seq_x = []
    seq_y = []
    for s in 1:inc:size(x,1)-seq_len-1
        i_s = s
        i_e = i_s + seq_len - 1
        push!(seq_x, x[i_s:i_e,:])
        push!(seq_y, x[i_s+1:i_e+1,:])
    end
    return seq_x, seq_y
end


# train_dl = get_dl()[1][1]
# x = train_dl[1]
