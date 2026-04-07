using DelimitedFiles


function  process_raw_mnist()
    dp = datadir("mnist_point_cloud")

    # check if the path exists
    if !ispath(dp) || length(readdir(dp)) == 0 || !all(map(x->x in readdir(dp), ["test.csv", "train.csv"]))
        mkpath(dp)
        error("MNIST point cloud data are not present. Unfortunately no automated download is available. Please download the `train.csv.zip` and `test.csv.zip` files manually from https://www.kaggle.com/cristiangarcia/pointcloudmnist2d and unzip them in `$(dp)`.")
    end

    @info "Processing raw MNIST point cloud data..."
    for fs in ["test", "train"]
        indata = readdlm(joinpath(dp, "$fs.csv"),',',Int32,header=true)[1] #FIXME
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