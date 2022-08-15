using Flux
using PaddedViews
using DelimitedFiles
using DrWatson

function procedure()
    #a = randn(3,1)
    #b = rpad(a, 10, 0) # expand array to length 10 with zeros
    # b ~ (10,1)
    #x = [randn(3) for i =1:3]
    #y = cat([reshape(rpad(xx, len, 0), (len,1)) for xx in x])

    a = [randn(Float32, 3,j) for j in rand(100:300, 32)];
    a_mask = [ones(size(x)) for x in a];
    max_set = maximum(size.(a))[end];
    b = map(x->Array(PaddedView(0, x, (3, max_set))), a);
    b_mask = map(x->Array(PaddedView(0, x, (3, max_set))), a_mask);
    c = cat(b..., dims=3);
    c_mask = cat(b_mask..., dims=3) .> 0; # mask as BitArray
    c_mask = Array(c_mask[1:1,:,:]); # 1:1 to keep shape (1,N,bs) -------BitArray->Array{Bool}
    #Array(PaddedView(0, a, (3, 300)))
    #@time b = reduce(vcat,map(i->Array(PaddedView(0, i, (3, 300))), a))
end


function standardize_list(X)
    m = zeros(3,1)
    m2 = zeros(3,1)
    samples = 0
    for x in X
        m += Flux.sum(x, dims=2)
        m2 += Flux.sum(x.^2, dims=2)
        samples += size(x,2)
    end
    mu = m./samples
    var = m2./samples-(m./samples).^2
    sigma = sqrt.(var)

    new_X = []
    for x in X
        push!(new_X, (x .- mu) ./ sigma) 
    end
    #println("mean = ", m./samples, " var = ", m2/samples-(m/samples).^2)
    return new_X, mu, sigma
end

function scale_list_to_interval(X, lowerb=0, upperb=1)
    max_val_ = nothing
    min_val_ = nothing
    for x in X
        max_val = maximum(x, dims=2)
        min_val = minimum(x, dims=2)
        if max_val_ === nothing
            max_val_ = max_val
        else
            common_max = cat(max_val_, max_val, dims=2)
            max_val_ = maximum(common_max, dims=2)
        end
        if min_val_ === nothing
            min_val_ = min_val
        else
            common_min = cat(min_val_, min_val, dims=2)
            min_val_ = minimum(common_min, dims=2)
        end
    end
    new_X = []
    for x in X
        scaled_0_1 = (x .- min_val_) ./ (max_val_ .- min_val_)
        scaled_lowerb_upperb = scaled_0_1 .* (upperb - lowerb) .+ lowerb 
        push!(new_X,  scaled_lowerb_upperb)
    end
    return new_X, max_val_, min_val_
end

function  process_raw_mnist()
    dp = datadir("mnist_point_cloud")

    # check if the path exists
    if !ispath(dp) || length(readdir(dp)) == 0 || !all(map(x->x in readdir(dp), ["test.csv", "train.csv"]))
        mkpath(dp)
        error("MNIST point cloud data are not present. Unfortunately no automated download is available. Please download the `train.csv.zip` and `test.csv.zip` files manually from https://www.kaggle.com/cristiangarcia/pointcloudmnist2d and unzip them in `$(dp)`.")
    end

    @info "Processing raw MNIST point cloud data..."
    for fs in ["test", "train"]
        indata = readdlm(joinpath(dp, "$fs.csv"),',',Int32,header=true)[1]
        labels = []
        data = []
        for (i,row) in enumerate(eachrow(indata))
            # get x data and specify valid values
            x = row[2:3:end]
            valid_inds = x .!= -1
            x = reshape(x[valid_inds],1,:)
            
            # get y and intensity
            y = reshape(row[3:3:end][valid_inds],1,:)
            v = reshape(row[4:3:end][valid_inds],1,:)

            # now append to the lists
            push!(labels, row[1])
            push!(data, vcat(x,y,v))
            #println(size(vcat(x,y,v)))
        end
        outdata = Dict(
            :labels => vcat(labels...),
            :data => data
            )
        bf = joinpath(dp, "$(fs).bson")
        save(bf, outdata)
        @info "Succesfuly processed and saved $bf"
    end
    @info "Done."
end

function load_mnist()
    dp = datadir("mnist_point_cloud")
    # check if the data is there
	if !ispath(dp) || !all(map(x->x in readdir(dp), ["test.bson", "train.bson"]))
		process_raw_mnist()
	end

    test = load(joinpath(dp, "test.bson"))
	train = load(joinpath(dp, "train.bson"))

    x_train, mu, sigma = standardize_list(float.(train[:data]))
    @info "train standardized μ = $(mu), σ = $(sigma)"
    x_test = [(x .- mu) ./ sigma for x in float.(test[:data])]
    @info "data were standardized"
    max_set = maximum([maximum(size.(x_train,2)), maximum(size.(x_test,2))])
    @info "maximum computed = $(max_set)"
    #compute masks
    @info "computing masks"
    train_mask = [ones(size(x)) for x in train[:data]];
    test_mask = [ones(size(x)) for x in test[:data]];
    train_mask = map(x->Array(PaddedView(0, x, (3, max_set))), train_mask);
    test_mask = map(x->Array(PaddedView(0, x, (3, max_set))), test_mask);
    @info "masks padded"
    train_mask = cat(train_mask..., dims=3) .> 0;
    test_mask = cat(test_mask..., dims=3) .> 0;
    train_mask = Array(train_mask[1:1,:,:]);
    test_mask = Array(test_mask[1:1,:,:]);
    @info "masks computed. Preparing data"
    # prepare data
    x_train = map(x->Array(PaddedView(0, x, (3, max_set))), x_train);
    x_test = map(x->Array(PaddedView(0, x, (3, max_set))), x_test);

    x_train = cat(x_train..., dims=3);
    x_test = cat(x_test..., dims=3);

    return (x_train, train_mask, train[:labels]), (x_test, test_mask, test[:labels])
end

function load_and_standardize_mnist()
    dp = datadir("mnist_point_cloud")
    # check if the data is there
	if !ispath(dp) || !all(map(x->x in readdir(dp), ["test.bson", "train.bson"]))
		process_raw_mnist()
	end

    test = load(joinpath(dp, "test.bson"))
	train = load(joinpath(dp, "train.bson"))

    x_train, mu, sigma = standardize_list(Array{Float32}.(train[:data]))
    @info "train standardized μ = $(mu), σ = $(sigma)"
    x_test = [(x .- mu) ./ sigma for x in Array{Float32}.(test[:data])]
    return (x_train, train[:labels]), (x_test, test[:labels])
end

function load_and_scale_mnist(lowerb=0, upperb=1)
    dp = datadir("mnist_point_cloud")
    # check if the data is there
	if !ispath(dp) || !all(map(x->x in readdir(dp), ["test.bson", "train.bson"]))
		process_raw_mnist()
	end

    test = load(joinpath(dp, "test.bson"))
	train = load(joinpath(dp, "train.bson"))

    x_train, min_, max_ = scale_list_to_interval(Array{Float32}.(train[:data]), lowerb, upperb)
    @info "train scaled into interval ($(lowerb), $(upperb)) \n min = $(min_), max = $(max_)"
    x_test = [(x .- min_) ./ (max_ .- min_) for x in Array{Float32}.(test[:data])]
    return (x_train, train[:labels]), (x_test, test[:labels])
end

function transform_batch(x, max=false)
    a_mask = [ones(size(a)) for a in x];
    feature_dim = size(x[1],1)
    if max
        max_set = maximum(size.(x))[end];
    else
        max_set = minimum(size.(x))[end]; #minimum
    end
    b = map(a->Array{Float32}(PaddedView(0, a, (feature_dim, max_set))), x);
    b_mask = map(a->Array(PaddedView(0, a, (feature_dim, max_set))), a_mask);
    c = cat(b..., dims=3);
    c_mask = cat(b_mask..., dims=3) .> 0; # mask as BitArray
    c_mask = Array(c_mask[1:1,:,:]);
    return c, c_mask
end