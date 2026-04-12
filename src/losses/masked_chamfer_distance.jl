function masked_chamfer_distance(x, y, x_mask, y_mask)
    return Flux.mean([
            chamfer_distance(
                unmask(x[:,:,i:i],x_mask[:,:,i:i]), 
                unmask(y[:,:,i:i],y_mask[:,:,i:i])
            ) for i=1:size(x,3)])
end

function masked_chamfer_distance_cpu(x, y, x_mask, y_mask)
    x, x_mask = x|>cpu, x_mask|>cpu
    y, y_mask = y|>cpu, y_mask|>cpu
    return Flux.mean([
            chamfer_distance(
                unmask(x[:,:,i:i],x_mask[:,:,i:i]), 
                unmask(y[:,:,i:i],y_mask[:,:,i:i])
            ) for i=1:size(x,3)])
end