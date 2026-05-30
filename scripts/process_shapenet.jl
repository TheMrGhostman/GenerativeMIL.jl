using Random
using LinearAlgebra
using Statistics
using ProgressBars
using StatsBase
using Serialization
using CairoMakie

function load_obj_triangles(path::String)
    vertices = Vector{NTuple{3, Float64}}()
    faces = Vector{NTuple{3, Int}}()

    open(path, "r") do io
        for line in eachline(io)
            line = strip(line)

            isempty(line) && continue
            startswith(line, "#") && continue

            parts = split(line)

            if parts[1] == "v"
                x = parse(Float64, parts[2])
                y = parse(Float64, parts[3])
                z = parse(Float64, parts[4])
                push!(vertices, (x, y, z))

            elseif parts[1] == "f"
                # OBJ face can look like:
                # f 1 2 3
                # f 1/1/1 2/2/2 3/3/3
                # f 1//1 2//2 3//3
                idxs = Int[]
                for p in parts[2:end]
                    vertex_id = split(p, "/")[1]
                    push!(idxs, parse(Int, vertex_id))
                end

                # triangulate polygon fan if face has more than 3 vertices
                for i in 2:(length(idxs)-1)
                    push!(faces, (idxs[1], idxs[i], idxs[i+1]))
                end
            end
        end
    end

    return vertices, faces
end

function triangle_area(a, b, c)
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c

    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az

    cross_x = uy * vz - uz * vy
    cross_y = uz * vx - ux * vz
    cross_z = ux * vy - uy * vx

    return 0.5 * sqrt(cross_x^2 + cross_y^2 + cross_z^2)
end

function sample_point_on_triangle(a, b, c)
    u = rand()
    v = rand()

    r1 = sqrt(u)
    r2 = v

    w1 = 1.0 - r1
    w2 = r1 * (1.0 - r2)
    w3 = r1 * r2

    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c

    return (
        w1 * ax + w2 * bx + w3 * cx,
        w1 * ay + w2 * by + w3 * cy,
        w1 * az + w2 * bz + w3 * cz,
    )
end

function sample_pointcloud_from_obj(path::String, n_points::Int=2048; normalize::Bool=true)
    vertices, faces = load_obj_triangles(path)

    areas = Float64[]
    valid_faces = NTuple{3, Int}[]

    for f in faces
        a = vertices[f[1]]
        b = vertices[f[2]]
        c = vertices[f[3]]

        area = triangle_area(a, b, c)

        if area > 0
            push!(areas, area)
            push!(valid_faces, f)
        end
    end

    total_area = sum(areas)
    probs = areas ./ total_area
    cdf = cumsum(probs)

    points = Matrix{Float32}(undef, 3, n_points)

    for i in 1:n_points
        r = rand()
        face_id = searchsortedfirst(cdf, r)

        f = valid_faces[face_id]

        p = sample_point_on_triangle(
            vertices[f[1]],
            vertices[f[2]],
            vertices[f[3]],
        )

        points[:, i] .= Float32.(p)
    end

    if normalize
        # center
        μ = mean(points; dims=2)
        points .-= μ

        # unit sphere normalization
        max_norm = maximum(sqrt.(sum(points .^ 2; dims=1)))
        points ./= max_norm
    end

    return points
end


function plot_pointcloud_sample(x)
    fig = Figure(size = (450, 450))
    ax = Axis3(fig[1, 1], aspect = :data)
    scatter!(ax, x[3, :], x[1, :], x[1, :])
    display(fig)
    fig
end



pth = "/mnt/personal/zorekmat/Datasets/ShapeNetCore/"

function process_airplanes(npoints=8192; normalize::Bool=true)
    airplane_id = "02691156"
    airplane_folders = readdir(joinpath(pth, airplane_id), join=true)
    airplanes = map(x->joinpath(x, "models", "model_normalized.obj"), airplane_folders)
    #airplane_folders[1]|>readdir
    #test
    #a = sample_pointcloud_from_obj(airplanes[4], 8192);
    #plot_pointcloud_sample(a)

    PClouds = zeros(Float32, 3, npoints, length(airplanes))
    for i in tqdm(axes(airplanes,1))
        PClouds[:,:,i] .= sample_pointcloud_from_obj(airplanes[i], npoints; normalize=normalize)
    end

    res_dict = Dict(
            "features" => PClouds,
            "targets" => zeros(length(airplanes)),
            "classes" => fill("airplane", length(airplanes)),
            "folder_id" => airplane_id
        )
    return res_dict
end
#serialize("/home/zorekmat/MIL/GenerativeMIL/data/datasets/shapenetcore/airplanes_normalized_$(npoints).jls", res_dict)