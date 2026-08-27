function load_cfg(path::String)
    yaml = YAML.load_file(path; dicttype=Dict{Symbol,Any})
    return Dict(Symbol(k) => v for (k, v) in yaml)
end

function resolve_activation(x)
    x isa Function && return x
    return eval(Symbol(x))
end


#function load_2d_mnist(npoints=512, normalized::Bool=true; seed=42)
#    data = deserialize(normalized ? joinpath(@__DIR__, "mnist_no_2_normalized.jls") : joinpath(@__DIR__, "mnist_no_2.jls"))
#    selected_pts = randperm(MersenneTwister(seed), size(data.x, 2))[1:npoints]
#    x = data.x[1:2, selected_pts, :]
#    return x
#end

function load_2d_mnist(npoints=512, normalized::Bool=true; seed=42)
    data = deserialize(normalized ? joinpath(@__DIR__, "mnist_no_2_normalized.jls") : joinpath(@__DIR__, "mnist_no_2.jls"))
    selected_pts = sortperm(data.x[3,:,:], dims=1, rev=true)[1:npoints]
    x = data.x[1:2, selected_pts, :]
    return x
end

function scale_markers(sizes; minsize=0.0, maxsize=10.0)
    # Map arbitrary numeric sizes to the [minsize, maxsize] interval using min-max scaling.
    isempty(sizes) && return sizes
    smin = minimum(sizes)
    smax = maximum(sizes)
    if smax == smin
        return fill((minsize + maxsize) / 2, length(sizes))
    end
    return (sizes .- smin) ./ (smax - smin) .* (maxsize - minsize) .+ minsize
end

function plot_mnist_sample(x)
    # If x has 3 rows, treat the 3rd row as marker sizes.
    θ = π/2  # 90 stupňů v radiánech
    R = [cos(θ) -sin(θ); sin(θ)  cos(θ)]
    if size(x, 1) == 3
        xy = R * x[1:2, :]
        sizes = x[3, :]
        sizes_scaled = scale_markers(sizes; minsize=0.0, maxsize=10.0)
        scatter(xy[2, :], xy[1, :], markersize = sizes_scaled)
    else
        x2 = R * x
        scatter(x2[2, :], x2[1, :])
    end
end

function plot_mnist_sample(ax::Axis, x)
    # If x has 3 rows, treat the 3rd row as marker sizes.
    θ = π/2  # 90 stupňů v radiánech
    R = [cos(θ) -sin(θ); sin(θ)  cos(θ)]
    if size(x, 1) == 3
        xy = R * x[1:2, :]
        sizes = x[3, :] 
        sizes_scaled = scale_markers(sizes; minsize=0.0, maxsize=10.0)
        scatter!(ax, xy[2, :], xy[1, :], markersize = sizes_scaled)
    else
        x2 = R * x
        scatter!(ax, x2[2, :], x2[1, :])
    end
end

function plot_mnist_samples(x, n=6)
    fig = Figure(size = (300, 300 * n))
    ax = [Axis(fig[i, 1], aspect = 1) for i in 1:n]
    for i in 1:n
        plot_mnist_sample(ax[i], x[:, :, i])
    end
    display(fig)
end

function plot_mnist_samples(x::T, y::T, n=6) where T
    fig = Figure(size = (300 * 2, 300 * n))
    #ax = [[Axis(fig[i, j], aspect = 1) for j in 1:2] for i in 1:n]
    for i in 1:n
        plot_mnist_sample(Axis(fig[i, 1], aspect = 1, title="Ground Truth") , x[:, :, i])
        plot_mnist_sample(Axis(fig[i, 2], aspect = 1, title="Reconstruction") , y[:, :, i])
    end
    display(fig)
    fig
end

function plot_mnist_samples(x::T, y::T, exist_logit, n=6) where T
    exist_prob = sigmoid.(vec(exist_logit))
    n = min(n, size(x, 3), size(y, 3), length(exist_prob))
    fig = Figure(size = (300 * 3, 300 * n))
    for i in 1:n
        plot_mnist_sample(Axis(fig[i, 1], aspect = 1, title = "Ground Truth"), x[:, :, i])
        plot_mnist_sample(Axis(fig[i, 2], aspect = 1, title = "Reconstruction"), y[:, :, i])
        ax = Axis(fig[i, 3], title = "Existence", limits = (0.5, 1.5, 0, 1), xticks = ([1], ["Prediction"]))
        barplot!(ax, [1], [exist_prob[i]])
    end
    display(fig)
    fig
end

function plot_mnist_samples_with_exist_title(x::T, y::T, exist_logit, n=6) where T
    n = min(n, size(x, 3), size(y, 3), length(exist_logit))
    fig = Figure(size = (300 * 2, 300 * n))
    for i in 1:n
        plot_mnist_sample(Axis(fig[i, 1], aspect = 1, title = "Ground Truth"), x[:, :, i])
        exist_color = exist_logit[1, i] > 0.5 ? :green : :red
        plot_mnist_sample(Axis(fig[i, 2], aspect = 1, title = "Reconstruction: $(exist_logit[1, i])", titlecolor = exist_color), y[:, :, i])
    end
    display(fig)
    fig
end

function plot_mnist_samples_with_exist_title(x::T, y::T, exist_logit, gt_label, n=6) where T
    n = min(n, size(x, 3), size(y, 3), length(exist_logit), length(gt_label))
    fig = Figure(size = (300 * 2, 300 * n))
    for i in 1:n
        plot_mnist_sample(Axis(fig[i, 1], aspect = 1, title = "Ground Truth: $(gt_label[i])"), x[:, :, i])
        exist_color = exist_logit[1, i] > 0.5 ? :green : :red
        plot_mnist_sample(Axis(fig[i, 2], aspect = 1, title = "Reconstruction: $(exist_logit[1, i])", titlecolor = exist_color), y[:, :, i])
    end
    display(fig)
    fig
end

function plot_pointcloud_sample(x)
    fig = Figure(size = (450, 450))
    ax = Axis3(fig[1, 1], aspect = :data)
    scatter!(ax, x[1, :], x[2, :], x[3, :])
    display(fig)
    fig
end

function plot_pointcloud_sample(ax::Axis3, x)
    scatter!(ax, x[1, :], x[3, :], x[2, :])
end

function plot_pointcloud_samples(x, n=6)
    fig = Figure(size = (450, 450 * n))
    ax = [Axis3(fig[i, 1], aspect = :data) for i in 1:n]
    for i in 1:n
        plot_pointcloud_sample(ax[i], x[:, :, i])
    end
    display(fig)
    fig
end

function plot_pointcloud_samples(x::T, y::T, n=6) where T
    fig = Figure(size = (450 * 2, 450 * n))
    for i in 1:n
        plot_pointcloud_sample(Axis3(fig[i, 1], aspect = :data, title = "Ground Truth"), x[:, :, i])
        plot_pointcloud_sample(Axis3(fig[i, 2], aspect = :data, title = "Reconstruction"), y[:, :, i])
    end
    display(fig)
    fig
end