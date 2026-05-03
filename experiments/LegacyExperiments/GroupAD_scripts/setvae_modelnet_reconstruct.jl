using DrWatson
@quickactivate
using ArgParse
using StatsBase
using BSON
using Random
using ValueHistories
#generative MIL
using GenerativeMIL
using Flux
using Zygote
using CUDA
using GenerativeMIL: transform_batch
using GenerativeMIL.Models: check, loss, unpack_mill, train_test_split
using MLDataPattern

#using FileIO #for loading of data
using HDF5
using Plots
using Dates

seed=1


data = HDF5.h5open("/home/zorekmat/MIL/GenerativeMIL/experiments/modelnet/data/modelnet10_2048.h5")
#data = FileIO.load("/home/zorekmat/MIL/GenerativeMIL/experiments/modelnet/data/modelnet10_$(npoints).h5")
X_train, X_test, Y_train, Y_test = data["X_train"]|>read, data["X_test"]|>read, data["Y_train"]|>read, data["Y_test"]|>read

#(X_train,Y_train), (X_val,Y_val) = train_test_split(X_train, Y_train, 0.2, seed=seed)
#data = ((X_train,Y_train), (X_val,Y_val), (X_test,Y_test)) 

df = BSON.load("/home/zorekmat/MIL/GenerativeMIL/data/experiments/setvae/ModelNet10-2048/seed=1/model_activation=swish_batchsize=100_beta=0.1_beta_anealing=500.0_epochs=8000_hdim=64_heads=4_init_seed=36819803_levels=7_lr=0.0001_lr_decay=WarmupCosine_prior=mog_prior_dim=32_vb_depth=2_vb_hdim=64.bson")
#df = BSON.load("/home/zorekmat/MIL/GenerativeMIL/data/experiments/setvae/ModelNet10-2048/seed=1/model_activation=swish_batchsize=100_beta=1.0_beta_anealing=500.0_epochs=8000_hdim=64_heads=4_init_seed=22414751_levels=7_lr=0.001_lr_decay=false_prior=mog_prior_dim=32_vb_depth=2_vb_hdim=64.bson")

model = df["model"];


x = X_train[:,:,1:10];
x_mask = ones(Bool, size(x[1:1, :, :])...);


#x, x_mask = transform_batch(X_train[:,:,1:9], true);
#x, x_mask = transform_batch(test_x, true);
#x = cat([X_test[:,:,Y_test.==i][:,:,1:1] for i=1:10]..., dims=3);

ri = rand(1:50,10)
#x = cat([X_train[:,:,Y_train.==i][:,:,ri[i]:ri[i]] for i=1:10]..., dims=3);
#x = cat([X_test[:,:,Y_test.==i][:,:,ri[i]:ri[i]] for i=1:10]..., dims=3);
x = cat([X_test[:,:,Y_test.==i][:,:,ri[i]:ri[i]] for i=1:10]..., dims=3);
y = GenerativeMIL.Models.reconstruct(model, x, x_mask);

titles = [ "bathtub","bed","chair","desk","dresser","monitor","night_stand","sofa","table","toilet"]

l = @layout [a b c d e f g h ch i ; j k l m n o p q r s]
plots_x = [scatter3d(x[1,:,i], x[2,:,i], x[3,:,i], alpha=1, legend=false, showaxis = false, ticks=false, title=titles[i]) for i=1:10];
plots_y = [scatter3d(y[1,:,i], y[2,:,i], y[3,:,i], alpha=1, legend=false, showaxis = false, ticks=false, title=string(chamfer_distance(x[:,:,i],y[:,:,i]))) for i=1:10];
plot(plots_x..., plots_y..., layout = l, size=(2500,500));

savefig("X_test-CD-recon-$(now()).png");
