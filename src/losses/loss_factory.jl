"""
Build a loss function callable from configuration.

Supported inputs:
- Function: returned as-is.
- String/Symbol: loss name.
- Dict with field `:type` and optional parameters.

Supported loss names:
- `chamfer_distance`
- `maximum_mean_discrepancy` (also accepts legacy typo `maximum_mean_discrepency`)
"""
function create_loss_function(cfg)
    cfg isa Function && return cfg

    if cfg isa AbstractString || cfg isa Symbol
        return _resolve_named_loss(cfg)
    end

    if cfg isa Dict
        loss_type = get(cfg, :type, get(cfg, "type", "chamfer_distance"))
        type_norm = _normalize_loss_name(loss_type)

        if type_norm == "chamfer_distance"
            return chamfer_distance
        elseif type_norm in ("maximum_mean_discrepancy", "maximum_mean_discrepency")
            sigma = get(cfg, :sigma, get(cfg, "sigma", 1f0))
            kernel = get(cfg, :kernel, get(cfg, "kernel", "rbf"))
            distance_kernel = _resolve_mmd_distance_kernel(kernel)
            return (x, y) -> maximum_mean_discrepancy(x, y; sigma=sigma, distance_kernel=distance_kernel)
        else
            error("Unsupported loss_function type: $(loss_type)")
        end
    end

    error("Unsupported loss_function config type: $(typeof(cfg))")
end

function _resolve_named_loss(name)
    name_norm = _normalize_loss_name(name)
    if name_norm == "chamfer_distance"
        return chamfer_distance
    elseif name_norm in ("maximum_mean_discrepancy", "maximum_mean_discrepency")
        return (x, y) -> maximum_mean_discrepancy(x, y)
    else
        error("Unknown loss function name: $(name)")
    end
end

function _normalize_loss_name(name)
    s = name isa Symbol ? String(name) : String(name)
    return lowercase(strip(s))
end

function _resolve_mmd_distance_kernel(kernel)
    if kernel isa Function
        return kernel
    end

    k = lowercase(String(kernel isa Symbol ? String(kernel) : kernel))
    if k == "rbf"
        return nothing
    elseif k == "imq"
        return d -> 1f0 ./ (1f0 .+ d)
    else
        error("Unsupported MMD kernel: $(kernel). Supported: rbf, imq")
    end
end
