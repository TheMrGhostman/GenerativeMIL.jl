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


function process_shapenet_folder(path, folder_id, classname=nothing; npoints=8192, normalize::Bool=true)
    subfolders = readdir(joinpath(path, folder_id), join=true)
    objects = map(x->joinpath(x, "models", "model_normalized.obj"), subfolders)
    objects = filter(ispath, objects)
    @info "found $(length(subfolders) - length(objects)) invalid files or files without `.obj` files"
    #airplane_folders[1]|>readdir
    #test
    #a = sample_pointcloud_from_obj(airplanes[4], 8192);
    #plot_pointcloud_sample(a)

    PClouds = zeros(Float32, 3, npoints, length(objects))
    for i in tqdm(axes(objects,1))
        PClouds[:,:,i] .= sample_pointcloud_from_obj(objects[i], npoints; normalize=normalize)
    end

    res_dict = Dict(
            "features" => PClouds,
            "targets" => zeros(length(objects)),
            "classes" => fill((classname === nothing) ? folder_id : classname, length(objects)),
            "folder_id" => folder_id
        )
    return res_dict
end

function load_class_from_15K_shapenet(class_id)
    tr_files = readdir("/mnt/personal/zorekmat/Datasets/ShapeNetCore.v2.PC15k/$(class_id)/train", join=true); 
    val_files = readdir("/mnt/personal/zorekmat/Datasets/ShapeNetCore.v2.PC15k/$(class_id)/val", join=true); 
    tst_files = readdir("/mnt/personal/zorekmat/Datasets/ShapeNetCore.v2.PC15k/$(class_id)/test", join=true); 

    @info "loading train split"
    x_train = zeros(Float32, 3, 15000, length(tr_files))
    for (i, file) in enumerate(tqdm(tr_files))
        x = npzread(file)
        x_train[:,:,i] .= x'
    end

    @info "loading train split"
    x_val = zeros(Float32, 3, 15000, length(val_files))
    for (i, file) in enumerate(tqdm(val_files))
        x = npzread(file)
        x_val[:,:,i] .= x'
    end

    @info "loading train split"
    x_test = zeros(Float32, 3, 15000, length(tst_files))
    for (i, file) in enumerate(tqdm(tst_files))
        x = npzread(file)
        x_test[:,:,i] .= x'
    end

    return (train = x_train, valid = x_val, test = x_test)
end


function load_and_save_15K(classes)
    for (class_id, class_name) in classes
        out = load_class_from_15K_shapenet(class_id)
        serialize("/home/zorekmat/MIL/GenerativeMIL/data/datasets/shapenetcore/$(class_name)_15000.jls", out)
    end
end

"""
load_and_save_15K(
    [
        ("02691156", "airplane"),
        ("02958343", "car"),
        ("04379243", "table"),
        ("03001627", "chair"),
        ("04256520", "sofa"),
        ("04090263", "rifle"),
        ("03636649", "lamp"),
        ("04530566", "watercraft"),
        ("03691459", "loudspeaker"),
        ("03211117", "display")
    ]
)
"""





pth = "/mnt/personal/zorekmat/Datasets/ShapeNetCore/"


res = process_shapenet_folder(pth, "02691156", "airplane"; npoints=8192)
#serialize("/home/zorekmat/MIL/GenerativeMIL/data/datasets/shapenetcore/airplane_normalized_8192.jls", res)

res = process_shapenet_folder(pth, "02691156", "airplane"; npoints=8192, normalize=false)
#serialize("/home/zorekmat/MIL/GenerativeMIL/data/datasets/shapenetcore/airplane_8192.jls", res)

res = process_shapenet_folder(pth, "02958343", "car"; npoints=8192)
#serialize("/home/zorekmat/MIL/GenerativeMIL/data/datasets/shapenetcore/car_normalized_8192.jls", res)

res = process_shapenet_folder(pth, "02958343", "car"; npoints=8192, normalize=false)
#serialize("/home/zorekmat/MIL/GenerativeMIL/data/datasets/shapenetcore/car_8192.jls", res)

res = process_shapenet_folder(pth, "04379243", "table"; npoints=8192)
#serialize("/home/zorekmat/MIL/GenerativeMIL/data/datasets/shapenetcore/table_normalized_8192.jls", res)

res = process_shapenet_folder(pth, "04379243", "table"; npoints=8192, normalize=false)
#serialize("/home/zorekmat/MIL/GenerativeMIL/data/datasets/shapenetcore/table_8192.jls", res)

res = process_shapenet_folder(pth, "03001627", "chair"; npoints=8192, normalize=true)
#serialize("/home/zorekmat/MIL/GenerativeMIL/data/datasets/shapenetcore/chair_normalized_8192.jls", res)

res = process_shapenet_folder(pth, "03001627", "chair"; npoints=8192, normalize=false)
#serialize("/home/zorekmat/MIL/GenerativeMIL/data/datasets/shapenetcore/chair_8192.jls", res)


res = process_shapenet_folder(pth, "04256520", "sofa"; npoints=8192, normalize=true)
serialize("/home/zorekmat/MIL/GenerativeMIL/data/datasets/shapenetcore/sofa_normalized_8192.jls", res)

res = process_shapenet_folder(pth, "04256520", "sofa"; npoints=8192, normalize=false)
serialize("/home/zorekmat/MIL/GenerativeMIL/data/datasets/shapenetcore/sofa_8192.jls", res)




using NPZ

x = npzread("/mnt/personal/zorekmat/Datasets/ShapeNetCore.v2.PC15k/02691156/train/816935cac027310d5e9e2656aff7dd5b.npy")  # typicky 15000 × 3

println(size(x))
println(eltype(x))

println("min per axis: ", minimum(x, dims=1))
println("max per axis: ", maximum(x, dims=1))
println("mean per axis: ", mean(x, dims=1))

center = mean(x, dims=1)
radii = sqrt.(sum((x .- center).^2, dims=2))
println("max radius: ", maximum(radii))
println("mean radius: ", mean(radii))


plot_pointcloud_sample(permutedims(x, (2,1)))

files = readdir("/mnt/personal/zorekmat/Datasets/ShapeNetCore.v2.PC15k/02691156/train", join=true); 

μ_s = zeros(Float32, 3, length(files));
Σ_s = zeros(Float32, length(files));

for i in tqdm(axes(files,1))
    x = npzread(files[i]);
    center = mean(x, dims=1)
    radii = sqrt.(sum((x .- center).^2, dims=2))
    μ_s[:,i] .= center[1,:]
    Σ_s[i] = maximum(radii)
end

mean(μ_s, dims=2)
median(μ_s, dims=2)

mean(Σ_s)
median(Σ_s)
maximum(Σ_s)

files = readdir("/mnt/personal/zorekmat/Datasets/ShapeNetCore.v2.PC15k/02958343/train", join=true); 

μ_s = zeros(Float32, 3, length(files));
Σ_s = zeros(Float32, length(files));

for i in tqdm(axes(files,1))
    x = npzread(files[i]);
    center = mean(x, dims=1)
    radii = sqrt.(sum((x .- center).^2, dims=2))
    μ_s[:,i] .= center[1,:]
    Σ_s[i] = maximum(radii)
end

mean(μ_s, dims=2)
median(μ_s, dims=2)

mean(Σ_s)
median(Σ_s)
maximum(Σ_s)



files = readdir("/mnt/personal/zorekmat/Datasets/ShapeNetCore.v2.PC15k/03001627/train", join=true); 

μ_s = zeros(Float32, 3, length(files));
Σ_s = zeros(Float32, length(files));

for i in tqdm(axes(files,1))
    x = npzread(files[i]);
    center = mean(x, dims=1)
    radii = sqrt.(sum((x .- center).^2, dims=2))
    μ_s[:,i] .= center[1,:]
    Σ_s[i] = maximum(radii)
end

mean(μ_s, dims=2)
median(μ_s, dims=2)

mean(Σ_s)
median(Σ_s)
maximum(Σ_s)


μ_s = zeros(Float32, 3, length(v));
Σ_s = zeros(Float32, length(v));

for i in tqdm(axes(v,1))
    x = v[i]
    center = mean(x, dims=2)
    radii = sqrt.(sum((x .- center).^2, dims=1))
    μ_s[:,i] .= center[:,1]
    Σ_s[i] = maximum(radii)
end

mean(μ_s, dims=2)
median(μ_s, dims=2)

mean(Σ_s)
median(Σ_s)
maximum(Σ_s)