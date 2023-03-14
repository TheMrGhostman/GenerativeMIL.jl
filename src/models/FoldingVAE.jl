"""
M. Masuda, R. Hachiuma, R. Fujii, H. Saito and Y. Sekikawa, 
"Toward Unsupervised 3d Point Cloud Anomaly Detection Using Variational Autoencoder,"
2021 IEEE International Conference on Image Processing (ICIP), 2021, pp. 3118-3122, 
doi: 10.1109/ICIP42928.2021.9506795.
"""

"""
def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]            # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx
"""

#y = permutedims(Flux.unsqueeze([5 6 7; 1 2 3],3), (2,1,3));

function knn(x::AbstractArray{<:Real, 2}, k::Int)
    # Input x ~ (Dim, N, BS)
    # x_t = permutedims(x, (2,1,3))
    # Input transposed to x_t ~ (N, Dim, BS)
    inner = -2 .* transpose(x) *  x 
    # inner product between transposed and normal ~ (N, N, BS)
    xx = Flux.sum(x.^2,dims=1)
    # expectation of x^2 .... ~ (1,N,BS) 
    pairwise_distance = -xx .- inner .- transpose(xx)
    # pairwise_distance ~ (N,N,BS)
    idx = mapslices(z->sortperm(z, rev=true)[1:k], pairwise_distance, dims=1)
end

function knn(x::AbstractArray{<:Real, 3}, k::Int)
    # Input x ~ (Dim, N, BS)
    # x_t = permutedims(x, (2,1,3))
    # Input transposed to x_t ~ (N, Dim, BS)
    inner = -2*Flux.batched_mul(permutedims(x, (2,1,3)), x) 
    # inner product between transposed and normal ~ (N, N, BS)
    xx = Flux.sum(x.^2,dims=1)
    # expectation of x^2 .... ~ (1,N,BS) 
    pairwise_distance = -xx .- inner .- permutedims(xx, (2,1,3))
    # pairwise_distance ~ (N,N,BS)
    idx = mapslices(z->sortperm(z, rev=true)[1:k], pairwise_distance, dims=(1))
end


function local_maxpool(x::AbstractArray{<:Real, 2},kx::AbstractArray{<:Real, 2})
    """ ORIGINAL IMPLEMENTATION
    def local_maxpool(x, idx):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
    
        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()                      # (batch_size, num_points, num_dims)
        x = x.view(batch_size*num_points, -1)[idx, :]           # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
        x = x.view(batch_size, num_points, -1, num_dims)        # (batch_size, num_points, k, num_dims)
        x, _ = torch.max(x, dim=2)                              # (batch_size, num_points, num_dims)

        return x
    """

    d,bs = size(x)
    x = dropdims(maximum(x[:,kx],dims=2),dims=2)
    return x
end

function local_maxpool(x::AbstractArray{<:Real, 3},kx::AbstractArray{<:Real, 3})
    # works on CPU and GPU
    cat(map(y->local_maxpool(y[1], y[2]), zip(eachslice(x, dims=3), eachslice(kx, dims=3)))..., dims=3)
end

function local_covariance(pts::AbstractArray{<:Real, 2}, idx::AbstractArray{<:Real, 2})
    """ ORIGINAL IMPLEMENTATION
    batch_size = pts.size(0)
    num_points = pts.size(2)
    pts = pts.view(batch_size, -1, num_points)              # (batch_size, 3, num_points)
 
    _, num_dims, _ = pts.size()

    x = pts.transpose(2, 1).contiguous()                    # (batch_size, num_points, 3)
    x = x.view(batch_size*num_points, -1)[idx, :]           # (batch_size*num_points*2, 3)
    x = x.view(batch_size, num_points, -1, num_dims)        # (batch_size, num_points, k, 3)

    x = torch.matmul(x[:,:,0].unsqueeze(3), x[:,:,1].unsqueeze(2))  # (batch_size, num_points, 3, 1) * (batch_size, num_points, 1, 3) -> (batch_size, num_points, 3, 3)
    # x = torch.matmul(x[:,:,1:].transpose(3, 2), x[:,:,1:])
    x = x.view(batch_size, num_points, 9).transpose(2, 1)   # (batch_size, 9, num_points)

    x = torch.cat((pts, x), dim=1)                          # (batch_size, 12, num_points)
    """
    bs = size(pts, 2)
    x = pts[:, idx]
    x = x[:,2:end,:] # the closest one is original point => filter it out 
    x = batched_mul(x, permutedims(x, (2,1,3))) # x @ x^t
    x = reshape(x, (:, bs))
    return cat(pts, x, dims=1)
end

function local_covariance(pts::AbstractArray{<:Real, 3},idx::AbstractArray{<:Real, 3})
    # works on CPU and GPU; slower on GPU but it works
    cat(map(y->local_covariance(y[1], y[2]), zip(eachslice(pts, dims=3), eachslice(idx, dims=3)))..., dims=3)
end

struct FoldingNet_encoder
    mlp1
    graph_layer
    mlp2
    n_neighbors
end

Flux.@functor FoldingNet_encoder

function Base.show(io::IO, m::FoldingNet_encoder)
    print(io, "FoldingNet_encoder(")
    print(io, "\n\t mlp1 -> $(m.mlp1)")
    print(io, "\n\t graph layer -> $(m.graph_layer)")
    print(io, "\n\t mlp2 -> $(m.mlp2)")
    print(io, "\n\t n_neighbors = $(m.n_neighbors) \n )")
end

function (enc::FoldingNet_encoder)(x::AbstractArray{<:Real}, kidx::AbstractArray{<:Real}; local_cov::Bool=false, skip::Bool=true)
    # 1) local covariance
    # 2) mlp1
    # 3) graph max-pooling and mlp
    # 4) mlp2
    # 5) to latent space
    if local_cov
        x = local_covariance(x, kidx);
    end
    x = enc.mlp1(x);
    # graph layer
    h = local_maxpool(x, kidx);
    h = enc.graph_layer[1](h);
    h = local_maxpool(h, kidx);
    h = enc.graph_layer[2](h);
    if skip
        h = cat(h,x, dims=1)
    end
    h = maximum(h, dims=2)
    μ, Σ = enc.mlp2(h)
end

function (enc::FoldingNet_encoder)(x::AbstractArray{<:Real}; local_cov::Bool=false, skip::Bool=true)
    kidx = nothing
    Zygote.ignore() do
        to_cuda = typeof(x) <: CUDA.CuArray
        x_ = (to_cuda) ? x |> cpu : x
        kidx = knn(x_, enc.n_neighbors); # i don't think i need to differentiate knn, it is just another input
        kidx = (to_cuda) ? kidx |> gpu : kidx
    end
    enc(x, kidx; local_cov=local_cov, skip=skip)
end




function FoldingNet_encoder(
    idim::Int=3, n_neighbors::Int=16, mlp1_hdim::Int=64, graph_hdim::Int=128, 
    hdim::Int=1024, zdim::Int=512, activation::Function=Flux.relu, 
    skip::Bool=true, local_cov::Bool=false

)
    """ ORIGINAL IMPLEMENTATION
    self.mlp1 = nn.Sequential(
        nn.Conv1d(12, 64, 1),
        nn.ReLU(),
        nn.Conv1d(64, 64, 1),
        nn.ReLU(),
        nn.Conv1d(64, 64, 1),
        nn.ReLU(),
    )

    self.linear1 = nn.Linear(64, 64)
    self.conv1 = nn.Conv1d(64, 128, 1)
    self.linear2 = nn.Linear(128, 128)
    self.conv2 = nn.Conv1d(128, 1024, 1)
    self.mlp2 = nn.Sequential(
        nn.Conv1d(1024, args.feat_dims, 1),
        nn.ReLU(),
        nn.Conv1d(args.feat_dims, args.feat_dims, 1),
    )

    Conv1d with kernel size = 1 is equivalent Dense layer
    """
    mlp1 = Flux.Chain(
        Flux.Dense(idim+local_cov*idim^2, mlp1_hdim, activation),
        Flux.Dense(mlp1_hdim, mlp1_hdim, activation),
        Flux.Dense(mlp1_hdim, mlp1_hdim, activation)
    )

    """
    graph_layer(self, x, idx):           
        x = local_maxpool(x, idx)    
        x = self.linear1(x)  
        x = x.transpose(2, 1)                                     
        x = F.relu(self.conv1(x))                            
        x = local_maxpool(x, idx)  
        x = self.linear2(x) 
        x = x.transpose(2, 1)                                   
        x = self.conv2(x)                       
        return x
    """
    graph_layers = [
        # local_maxpool
        Flux.Chain(
            Flux.Dense(mlp1_hdim, mlp1_hdim), # no activation 
            Flux.Dense(mlp1_hdim, graph_hdim, activation) # relu
        ),
        # local maxpool
        Flux.Chain(
            Flux.Dense(graph_hdim, graph_hdim), # no activation
            Flux.Dense(graph_hdim, hdim) # no activation 
        )
    ]

    mlp2 = SplitLayer(hdim+skip*mlp1_hdim, (zdim, zdim), (identity, softplus))
    return FoldingNet_encoder(mlp1, graph_layers, mlp2, n_neighbors)
end



struct FoldingNet_decoder
    sphere # 3D shpere / nD sphere 
    folding_1
    folding_2
    n_samples
end

Flux.@functor FoldingNet_decoder

function (dec::FoldingNet_decoder)(x::AbstractArray{<:Real, 2})
    x = repeat(x, 1, dec.n_samples)
    h = cat(x, dec.sphere, dims=1)
    h = dec.folding_1(h)
    h = cat(x, h, dims=1)
    h = dec.folding_2(h)
    return h
end

function (dec::FoldingNet_decoder)(x::AbstractArray{<:Real, 3})
    device = get_device(dec)
    bs = size(x)[end]
    mainfold_ = device.ones(Float32, 1, dec.n_samples, bs)
    x = mainfold_ .* x
    sphere = mainfold_ .* Flux.unsqueeze(dec.sphere, 3)
    h = cat(x, sphere, dims=1) 
    #println("h", h|>size)
    h = dec.folding_1(h)
    h = cat(x, h, dims=1)
    h = dec.folding_2(h)
    return h
end

function FoldingNet_decoder(
    idim::Int=512, odim::Int=3, n_samples::Int=200, pdim::Int=3, hdim::Int=512,
    activation::Function=Flux.relu
)
    """
    self.folding1 = nn.Sequential(
        nn.Conv1d(args.feat_dims+3, args.feat_dims, 1),
        nn.ReLU(),
        nn.Conv1d(args.feat_dims, args.feat_dims, 1),
        nn.ReLU(),
        nn.Conv1d(args.feat_dims, 3, 1),
    )  

    self.folding2 = nn.Sequential(
        nn.Conv1d(args.feat_dims+3, args.feat_dims, 1),
        nn.ReLU(),
        nn.Conv1d(args.feat_dims, args.feat_dims, 1),
        nn.ReLU(),
        nn.Conv1d(args.feat_dims, 3, 1),
    )
    """
    folding_1 = Flux.Chain(
        Flux.Dense(pdim+idim, hdim, activation),
        Flux.Dense(hdim, hdim, activation),
        Flux.Dense(hdim, odim) # no activation
    )

    folding_2 = Flux.Chain(
        Flux.Dense(odim+idim, hdim, activation),
        Flux.Dense(hdim, hdim, activation),
        Flux.Dense(hdim, odim) # no activation
    )

    sphere = randn(Float32, pdim, n_samples)
    sphere = sphere ./ sqrt.(sum(abs2, sphere, dims=1))
    # normal sampling from shpere

    return FoldingNet_decoder(sphere, folding_1, folding_2, n_samples)
end

function Base.show(io::IO, m::FoldingNet_decoder)
    print(io, "FoldingNet_decoder(")
    print(io, "\n\t folding 1 -> $(m.folding_1)")
    print(io, "\n\t folding 2 -> $(m.folding_2)")
    print(io, "\n\t n_samples = $(m.n_samples)")
    print(io, "\n\t shpere = $(m.sphere |> size) \n ) ")
end

struct FoldingNet_VAE
    encoder::FoldingNet_encoder
    decoder::FoldingNet_decoder
    local_cov::Bool
    skip::Bool
end

Flux.@functor FoldingNet_VAE

function (model::FoldingNet_VAE)(x::AbstractArray{<:Real, 3})
    μ, Σ = model.encoder(x; local_cov=model.local_cov, skip=model.skip);
    z = μ .+ Σ .* MLUtils.randn_like(μ); 
    return model.decoder(z)
end

function loss(model::FoldingNet_VAE, x::AbstractArray{<:Real, 3}; β=1f0, γ=1f0)
    μₒ, Σₒ = model.encoder(x; local_cov=model.local_cov, skip=model.skip);
    z = μₒ .+ Σₒ .* MLUtils.randn_like(μₒ);
    x̂ = model.decoder(z)
    μᵣ, Σᵣ = model.encoder(x̂; local_cov=model.local_cov, skip=model.skip);
    # 𝓛ᵣₑ = reconstruction error
    # 𝓛ₖₗₒᵣᵢ = KL divergence for original input
    # 𝓛ₖₗᵣₑ = KL divergence for reconstructed input
    𝓛ᵣₑ = Flux3D.chamfer_distance(x̂, x) 
    𝓛ₖₗₒᵣᵢ = - Flux.mean(0.5f0 * sum(1f0 .+ log.(Σₒ.^2) - μₒ.^2  - Σₒ.^2, dims=1)) 
    𝓛ₖₗᵣₑ = - Flux.mean(0.5f0 * sum(1f0 .+ log.(Σᵣ.^2) - μᵣ.^2  - Σᵣ.^2, dims=1))
    #println( 𝓛ᵣₑ, "|", 𝓛ₖₗₒᵣᵢ, "|" , 𝓛ₖₗᵣₑ)
    𝓛 = 𝓛ᵣₑ .+ β .* 𝓛ₖₗₒᵣᵢ .+ γ .* 𝓛ₖₗᵣₑ # default β = 1  & γ = 1 | β .* (𝓛ₖₗₒᵣᵢ + 𝓛ₖₗᵣₑ)
end


function simple_loss(model::FoldingNet_VAE, x::AbstractArray{<:Real, 3}; β=1f0, kwargs...)
    μₒ, Σₒ = model.encoder(x; local_cov=model.local_cov, skip=model.skip);
    z = μₒ .+ Σₒ .* MLUtils.randn_like(μₒ);#randn(Float32, size(μₒ)...)
    x̂ = model.decoder(z)
    #μᵣ, Σᵣ = model.encoder(x̂; local_cov=model.local_cov, skip=model.skip);
    # 𝓛ᵣₑ = reconstruction error
    # 𝓛ₖₗₒᵣᵢ = KL divergence for original input
    # 𝓛ₖₗᵣₑ = KL divergence for reconstructed input
    𝓛ᵣₑ = Flux3D.chamfer_distance(x̂, x) 
    𝓛ₖₗₒᵣᵢ = - Flux.mean(0.5f0 * sum(1f0 .+ log.(Σₒ.^2) - μₒ.^2  - Σₒ.^2, dims=1)) 
    #𝓛ₖₗᵣₑ = - Flux.mean(0.5f0 * sum(1f0 .+ log.(Σᵣ.^2) - μᵣ.^2  - Σᵣ.^2, dims=1))
    𝓛 = 𝓛ᵣₑ .+ β .* 𝓛ₖₗₒᵣᵢ # default β = 1  & γ = 1
end

function simple_loss(model::FoldingNet_VAE, x::AbstractArray{<:Real, 3}, kidx::AbstractArray{<:Real}; β=1f0, kwargs...)
    μₒ, Σₒ = model.encoder(x, kidx; local_cov=model.local_cov, skip=model.skip);
    z = μₒ .+ Σₒ .* MLUtils.randn_like(μₒ);#randn(Float32, size(μₒ)...)
    x̂ = model.decoder(z)
    #μᵣ, Σᵣ = model.encoder(x̂; local_cov=model.local_cov, skip=model.skip);
    # 𝓛ᵣₑ = reconstruction error
    # 𝓛ₖₗₒᵣᵢ = KL divergence for original input
    # 𝓛ₖₗᵣₑ = KL divergence for reconstructed input
    𝓛ᵣₑ = Flux3D.chamfer_distance(x̂, x) 
    𝓛ₖₗₒᵣᵢ = - Flux.mean(0.5f0 * sum(1f0 .+ log.(Σₒ.^2) - μₒ.^2  - Σₒ.^2, dims=1)) 
    #𝓛ₖₗᵣₑ = - Flux.mean(0.5f0 * sum(1f0 .+ log.(Σᵣ.^2) - μᵣ.^2  - Σᵣ.^2, dims=1))
    𝓛 = 𝓛ᵣₑ .+ β .* 𝓛ₖₗₒᵣᵢ # default β = 1  & γ = 1
end

function logging_loss(model::FoldingNet_VAE, x::AbstractArray{<:Real, 2}; β=1f0)
    μₒ, Σₒ = model.encoder(x; local_cov=model.local_cov, skip=model.skip);
    z = μₒ .+ Σₒ .* randn(Float32, size(μₒ)...);
    x̂ = model.decoder(z)
    μᵣ, Σᵣ = model.encoder(x̂; local_cov=model.local_cov, skip=model.skip);
    # 𝓛ᵣₑ = reconstruction error
    # 𝓛ₖₗₒᵣᵢ = KL divergence for original input
    # 𝓛ₖₗᵣₑ = KL divergence for reconstructed input
    𝓛ᵣₑ = Flux3D.chamfer_distance(x̂, x) 
    𝓛ₖₗₒᵣᵢ = - Flux.mean(0.5f0 * sum(1f0 .+ log.(Σₒ.^2) - μₒ.^2  - Σₒ.^2, dims=1)) 
    𝓛ₖₗᵣₑ = - Flux.mean(0.5f0 * sum(1f0 .+ log.(Σᵣ.^2) - μᵣ.^2  - Σᵣ.^2, dims=1))
    𝓛 = 𝓛ᵣₑ .+ β .* (𝓛ₖₗₒᵣᵢ + 𝓛ₖₗᵣₑ) # default β = 1 
    return 𝓛, 𝓛ᵣₑ, 𝓛ₖₗₒᵣᵢ, 𝓛ₖₗᵣₑ
end


function foldingnet_constructor_from_named_tuple(
    ;idim::Int, local_cov::Bool=false, skip::Bool=true, n_neighbors::Int=20, edim::Int=64,
    zdim::Int=512, activation::String="relu", ddim::Int=512, n_samples::Int=200, pdim::Int=3,
    kwargs...
)
    activation = eval(:($(Symbol(activation))))
    """
    encoder = FoldingNet_encoder(
        idim=idim, n_neighbors=n_neighbors, mlp1_hdim=edim, graph_hdim=2*edim, 
        hdim=8*edim, zdim=zdim, activation=activation, 
        skip=skip, local_cov=local_cov
    )
    decoder = FoldingNet_decoder(
        idim=zdim, odim=idim, n_samples=n_samples, pdim=pdim, hdim=ddim,
        activation=activation
    )
    """
    encoder = FoldingNet_encoder(
        idim, n_neighbors, edim, 2*edim, 
        8*edim, zdim, activation, 
        skip, local_cov
    )

    decoder = FoldingNet_decoder(
        zdim, idim, n_samples, pdim, ddim,
        activation
    )

    return FoldingNet_VAE(encoder, decoder, local_cov, skip)

end



function test_enc_backward(x)
    e = FoldingNet_encoder()
    ps = Flux.params(e);
    opt = ADAM(1f-3)

    loss_, back = Flux.pullback(ps) do 
        Flux.mean(e(x)[1]) 
    end;
    grad = back(1f0);
    Flux.Optimise.update!(opt, ps, grad)
end


function test_backward_1(x)
    model = FoldingNet_VAE(
        FoldingNet_encoder(),
        FoldingNet_decoder(),
        false,
        true
    )
    #loss(model, x)
    #e = FoldingNet_encoder()
    #d = FoldingNet_decoder()

    #m, s = e(x);
    #z = m .+ s .* randn(Float32, size(m)...);
    #d(z);
    ps = Flux.params(model);
    opt = ADAM(1f-3)
    loss_, back = Flux.pullback(ps) do 
        loss(model, x)
    end;
    grad = back(1f0);
    Flux.Optimise.update!(opt, ps, grad)
end

function test_backward_batch(x)
    model = FoldingNet_VAE(
        FoldingNet_encoder(),
        FoldingNet_decoder(),
        false,
        true
    )
    #loss(model, x)
    #e = FoldingNet_encoder()
    #d = FoldingNet_decoder()

    #m, s = e(x);
    #z = m .+ s .* randn(Float32, size(m)...);
    #d(z);
    ps = Flux.params(model);
    opt = ADAM(1f-3)
    loss_f(x) = loss(model, x)

    loss_, back = Flux.pullback(ps) do 
        Flux.mean(loss_f.(x))#loss(model, x)
    end;
    grad = back(1f0);
    Flux.Optimise.update!(opt, ps, grad)
    println(loss_)
end

function test_backward_batch_train(x)
    model = FoldingNet_VAE(
        FoldingNet_encoder(),
        FoldingNet_decoder(),
        false,
        true
    )
    ps = Flux.params(model);
    opt = ADAM(1f-3)
    loss_f(x) = loss(model, x)

    Flux.train!(loss_f, ps, xx, opt)
end