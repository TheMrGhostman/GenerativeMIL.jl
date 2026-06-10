using Random
using LinearAlgebra
using Statistics
using DelimitedFiles
using ProgressBars
using StatsBase
using Serialization



function read_nonempty_line(io)
    while !eof(io)
        line = strip(readline(io))
        if !isempty(line)
            return line
        end
    end
    return nothing
end

function load_off(path::AbstractString; verbose::Bool = false)
    open(path, "r") do io
        header = read_nonempty_line(io)
        header === nothing && error("$path is not a valid OFF file")

        counts_line = if header == "OFF"
            read_nonempty_line(io)
        elseif startswith(header, "OFF")
            header[4:end]
        else
            error("$path is not a valid OFF file")
        end

        #counts_line = read_nonempty_line(io)
        verbose && @info "OFF header: $header, counts line: $counts_line"

        counts_line === nothing && error("Missing vertex/face counts in $path")
        counts = parse.(Int, split(counts_line))
        length(counts) >= 2 || error("Invalid OFF counts line in $path")

        vertex_count = counts[1]
        face_count = counts[2]

        vertices = Array{Float32}(undef, vertex_count, 3)
        for i in 1:vertex_count
            parts = split(strip(readline(io)))
            length(parts) >= 3 || error("Invalid vertex line $i in $path")
            vertices[i, 1] = parse(Float32, parts[1])
            vertices[i, 2] = parse(Float32, parts[2])
            vertices[i, 3] = parse(Float32, parts[3])
        end

        faces = Vector{NTuple{3, Int}}()
        for _ in 1:face_count
            line = strip(readline(io))
            isempty(line) && continue

            parts = split(line)
            polygon_size = parse(Int, parts[1])
            indices = parse.(Int, parts[2:(1 + polygon_size)])

            if polygon_size == 3
                push!(faces, (indices[1] + 1, indices[2] + 1, indices[3] + 1))
            elseif polygon_size > 3
                for i in 2:(polygon_size - 1)
                    push!(faces, (indices[1] + 1, indices[i] + 1, indices[i + 1] + 1))
                end
            end
        end

        return vertices, faces
    end
end

function triangle_area(a, b, c)
    edge_a = b .- a
    edge_b = c .- a
    return 0.5f0 * norm(cross(edge_a, edge_b))
end

function sample_point_cloud(vertices, faces; num_points::Int = 2048, seed::Union{Nothing, Int} = 42)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    areas = Float32[]
    for face in faces
        a = @view vertices[face[1], :]
        b = @view vertices[face[2], :]
        c = @view vertices[face[3], :]
        push!(areas, triangle_area(a, b, c))
    end

    total_area = sum(areas)
    total_area > 0 || error("Mesh has zero-area faces only")

    probabilities = areas ./ total_area
    cumulative = cumsum(probabilities)

    points = Array{Float32}(undef, num_points, 3)
    for i in 1:num_points
        r = rand(rng)
        face_index = searchsortedfirst(cumulative, r)
        face_index = min(face_index, length(faces))
        face = faces[face_index]

        a = @view vertices[face[1], :]
        b = @view vertices[face[2], :]
        c = @view vertices[face[3], :]

        u = rand(rng)
        v = rand(rng)
        sqrt_u = sqrt(u)

        points[i, :] = (1 - sqrt_u) .* a .+ (sqrt_u * (1 - v)) .* b .+ (sqrt_u * v) .* c
    end

    return points
end


function main(PTH::String; NPOINTS = 8196, MODE="train", MIN_SAMPLES_PER_CLASS=600, SEED = 123)
    #MIN_SAMPLES_PER_CLASS = 600
    SELECTED_CLASSES = ["airplane", "car", "chair", "bed", "table", "sofa", "monitor", "lamp", "plant", "tent"]

    classes = readdir(PTH, join=true);
    names = Dict()
    pcs = Dict()

    for class in classes
        if !(basename(class) in SELECTED_CLASSES)
            continue
        end
        println("Processing class: ", basename(class))
        cls = basename(class)
        mode_path = joinpath(class, MODE)
        pcs[cls] = []
        names[cls] = []
        off_files = filter(f -> endswith(f, ".off"), readdir(mode_path, join=true))
        if isempty(off_files)
            @warn "No OFF files found for class $(cls) in $(MODE) mode at path: $(mode_path)"
            continue
        end
        if length(off_files) < MIN_SAMPLES_PER_CLASS
            additional_files = sample(off_files, MIN_SAMPLES_PER_CLASS - length(off_files); replace = true)
            off_files = vcat(off_files, additional_files)
        end
        @info  "length = $(length(off_files)) |  OFF files found for class $(cls) in $(MODE) mode"
        #break
        for off_file in tqdm(off_files)
            #println("  Loading file: ", basename(off_file))
            vertices, faces = load_off(off_file)
            point_cloud = sample_point_cloud(vertices, faces; num_points = NPOINTS, seed = SEED)
            push!(names[cls], basename(off_file))
            push!(pcs[cls], point_cloud)
        end
    end

    
    M = 0
    for cls in SELECTED_CLASSES
        println("Class: ", cls, ", Number of samples: ", length(pcs[cls]))
        M += length(pcs[cls])
    end
    
    X = zeros(Float32, M, NPOINTS, 3)
    CLS = string.(zeros(M))
    NAMES = string.(zeros(M))
    Y = zeros(Int, M)

    idx = 0
    for (i, cls) in enumerate(SELECTED_CLASSES) #SELECTED_CLASSES
        for (j, pc) in enumerate(pcs[cls])
            X[idx + j, :, :] = pc
            CLS[idx + j] = cls
            NAMES[idx + j] = names[cls][j]
            Y[idx + j] = i
        end
        idx += length(pcs[cls])
    end

    res_dict = Dict(
        "features" => permutedims(X, (3,2,1)), # (3, NPOINTS, M)
        "targets" => Y,
        "classes" => CLS,
        "names" => NAMES
    )
    return res_dict
end

#pth = "/home/zorekmat/Datasets/ModelNet40/modelnet40-princeton-3d-object-dataset/versions/1/ModelNet40/"