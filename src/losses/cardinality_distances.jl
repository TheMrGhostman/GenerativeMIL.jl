function pairwise_lin_card_distance(n̂::AbstractArray{T, 4}, x_mask::AbstractArray{Bool,4}) where T <: AbstractFloat
    _, n_max, _ , bs = size(x_mask);
    true_card = T.(sum(x_mask, dims=2)) ./ n_max;# (1, 1, l, bs)
    pred_card = Flux.softplus.(n̂);  # (1 , 1, n_slots, bs)
    ℒ_card = _pairwise_sqdist_batched( reshape(pred_card, 1, :, bs) , reshape(true_card, 1, :, bs))
    return ℒ_card
end