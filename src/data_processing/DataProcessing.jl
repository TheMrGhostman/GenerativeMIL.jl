# structure of the data processing module
# this module will contain functions to process the data, such as building point clouds from images,
# and also functions to load and save the datasets, and to split the datasets into train and test sets

include("grayscale2pointcloud.jl")
export build_point_cloud_from_grayscale_image

include("load_datasets.jl")
export load_modelnet10, load_mnist, create_dataloaders, on_fly_collate_fn