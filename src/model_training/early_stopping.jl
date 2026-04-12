mutable struct EarlyStopping
    best_model
    best_loss
    patience
    curr_patience
end

function EarlyStopping(model, patience::Real)
    return EarlyStopping(deepcopy(model), Inf, copy(patience), copy(patience))
end

function (es::EarlyStopping)(loss::Real, model)
    if loss < es.best_loss
        es.best_loss = loss
        es.curr_patience = deepcopy(es.patience)
        es.best_model = deepcopy(model)
    else
        es.curr_patience -= 1 
    end
    (es.curr_patience == 0) ? true : false # to stop: true/false
end