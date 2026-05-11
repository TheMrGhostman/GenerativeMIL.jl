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

function plot_mnist_sample(x)
    θ = π/2  # 90 stupňů v radiánech
    R = [cos(θ) -sin(θ); sin(θ)  cos(θ)]
    x = R * x
    scatter(x[2, :], x[1, :])
end

function plot_mnist_sample(ax::Axis, x)
    θ = π/2  # 90 stupňů v radiánech
    R = [cos(θ) -sin(θ); sin(θ)  cos(θ)]
    x = R * x
    scatter!(ax, x[2, :], x[1, :])
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