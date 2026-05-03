using DrWatson
using Random 
using MLUtils
using Serialization
using Test
using StatsBase
using HDF5

root = joinpath(@__DIR__, "..")
# load the loader implementation directly (avoid GenerativeMIL module to prevent name clashes)
include(joinpath(root, "src", "data_processing", "load_datasets.jl"))

# determine dataset paths from implementation helpers
balanced_path = datadir("datasets/testing/balanced_test.jls")
natural_path = datadir("datasets/testing/natural_test.jls")

_mnist_balanced_path() = balanced_path # temporary
_mnist_natural_path() = natural_path # temporary

mkpath(dirname(balanced_path))

@info "Writing small synthetic datasets to $(dirname(balanced_path)) for tests"

balanced_features = rand(Float32, 3, 10, 20)
balanced_targets = collect(1:20)
serialize(balanced_path, Dict("features" => balanced_features, "targets" => balanced_targets))

natural_features = [rand(Float32, 3, rand(3:7)) for _ in 1:20]
natural_targets = collect(1:20)
serialize(natural_path, Dict("features" => natural_features, "targets" => natural_targets))
#natural_targets = collect(1:6)
#Serialization.serialize(natural_path, Dict("features" => natural_features, "targets" => natural_targets))

@testset "load_mnist helpers and modes" begin
    # Balanced static sampling
    (train, val, test) = load_mnist(4; validation=true, cardinality_count=:balanced, sample_on_fly=false, seed=1)
    x_train, y_train = train
    @test size(x_train, 1) == 3
    @test size(x_train, 2) == 4
    @test size(x_train, 3) == length(y_train)

    x_val, y_val = val
    @test size(x_val, 1) == 3
    @test size(x_val, 2) == 4
    @test size(x_val, 3) == length(y_val)

    x_test, y_test = test
    @test size(x_test, 1) == 3
    @test size(x_test, 2) == 4
    @test size(x_test, 3) == length(y_test)

    # Balanced on-the-fly -> mapobs
    (train_of, val_of, test_of) = load_mnist(4; validation=true, cardinality_count=:balanced, sample_on_fly=true, seed=1)
    x_train_of, y_train_of = train_of
    @test isa(x_train_of, MLUtils.MappedData)
    obs1 = MLUtils.getobs(x_train_of, 1)
    @test size(obs1) == (3, 4, 1)
    obs2 = MLUtils.getobs(x_train_of, 1:2)
    @test size(obs2) == (3, 4, 2)


    # Natural on-the-fly
    (train_nat, val_nat, test_nat) = load_mnist(3; validation=true, cardinality_count=:natural, sample_on_fly=true, seed=1)
    x_train_nat, y_train_nat = train_nat
    @test isa(x_train_nat, MLUtils.MappedData)
    obsn = MLUtils.getobs(x_train_nat, 1)
    @test size(obsn) == (3, 3, 1)


    (train_nat, val_nat, test_nat) = load_mnist(2; validation=true, cardinality_count=:natural, sample_on_fly=false, seed=1)
    x_train_nat, y_train_nat = train_nat
    @test size(x_train_nat, 1) == 3
    @test size(x_train_nat, 2) == 2
    @test size(x_train_nat, 3) == length(y_train_nat)
    obsn = MLUtils.getobs(x_train_nat, 1:1)
    @test size(obsn) == (3, 2, 1)
    
end

@testset "create_dataloaders and collate" begin
    cfg = Dict(:dataset => "mnist", :npoints => 4, :validation => true, :cardinality_count => :balanced, :sample_on_fly => true, :seed => 1)

    collate = batch -> (
        cat((b[1] for b in batch)...; dims=3),
        collect(b[2] for b in batch)
    )

    dls = create_dataloaders(cfg; batch_size=2, train_collate_fn=collate, valid_collate_fn=collate, test_collate_fn=collate)
    @test isa(dls.train, MLUtils.DataLoader)
    batch = first(dls.train)
    x, y = batch
    @test size(x, 1) == 3
    @test size(x, 2) == 4
    @test size(x, 3) <= 2
    @test length(y) <= 2
end

@testset "create_dataloaders and collate" begin
    cfg = Dict(:dataset => "mnist", :npoints => 3, :validation => true, :cardinality_count => :natural, :sample_on_fly => true, :seed => 1)

    collate = batch -> (
        cat((b[1] for b in batch)...; dims=3),
        collect(b[2] for b in batch)
    )

    dls = create_dataloaders(cfg; batch_size=2, train_collate_fn=collate, valid_collate_fn=collate, test_collate_fn=collate)
    @test isa(dls.train, MLUtils.DataLoader)
    batch = first(dls.train)
    x, y = batch
    @test size(x, 1) == 3
    @test size(x, 2) == 3
    @test size(x, 3) <= 2
    @test length(y) <= 2
end


@testset "create_dataloaders modelnet10" begin
   
    cfg = Dict(:dataset => "modelnet10", :npoints => 1024, :type => "all", :validation => true, :seed => 1)
    dls = create_dataloaders(cfg; batch_size=2)
    @test !isnothing(dls.train)
    @test !isnothing(dls.valid)
    @test !isnothing(dls.test)

    batch_train = first(dls.train)
    x_batch, y_batch = batch_train
    @test size(x_batch, 1) == 3
    @test size(x_batch, 2) == 1024
    @test size(x_batch, 3) <= 2
    @test length(y_batch) <= 2

end
