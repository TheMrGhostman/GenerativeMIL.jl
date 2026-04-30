# structure of the data processing module
# this module will contain functions to process the data, such as building point clouds from images,
# and also functions to load and save the datasets, and to split the datasets into train and test sets


_mnist_balanced_path() = datadir("datasets/mnist_pc/mnist_4x_point_clouds_3x900_matrix.jls")
_mnist_natural_path() = datadir("datasets/mnist_pc/mnist_4x_point_clouds_all_vec.jls")
_modelnet10_path(npoints) = datadir("datasets/modelnet10/modelnet10_$(npoints).h5")

include("grayscale2pointcloud.jl")
export build_point_cloud_from_grayscale_image

include("data_utils.jl")
export normalize_point_cloud, on_fly_collate_fn
export sample_fixed_n_from_matrix, sample_fixed_n_unsqueeze, sample_fixed_n, _stack_point_clouds

include("load_datasets.jl")
export load_modelnet10, load_mnist, create_dataloaders