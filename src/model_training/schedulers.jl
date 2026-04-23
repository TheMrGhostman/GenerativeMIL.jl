
function CreateAnealer(max_value, milestone)
    new_value = it->max_value * min(1f0, it/milestone)
end

function CreateLrScheduler(sch_name, lr, max_iters; milestones=[0.02, 0.8], scale=5, kwargs...)
    @assert sch_name in [false, "false", "Linear2ndHalf", "WarmupCosine"] # this can be expanded later
    if sch_name == "WarmupCosine"
        scheduler = WarmupCosine(1e-7, lr*scale, lr, Int(milestones[1] * max_iters), Int(milestones[2] * max_iters))
        # from 0 to milestones[1]% iters there is linear increase of learing rate with "scale"
        # from milestones[1]% to milestones[2]% there is cosine decay of learing rate 
        # from milestones[2]% to 100% iters there is constant learing rate 
    elseif sch_name == "Linear2ndHalf"
        scheduler = it -> lr .* min.(1.0, 2.0 - it/(0.5*max_iters))
        #lr .* min.(1.0, map(x -> 2.0 - x/(0.5*max_iters), 1:max_iters)) 
        # learning rate decay (0%,50%) -> 1 , (50%, 100%) -> linear(1->0)
    else
        scheduler = x -> lr
        # constant learning rate 
    end
    return scheduler
end


"""
scheduler with warmup
using ParameterSchedulers
x = [1:1200...]
s = WarmupLinear(0, 0.1, 0.001, 200, 1000, CosAnneal(λ0=0.001, λ1=0.1, period=1000))

lineplot(x, s.(x); border= :none)
    ┌─────────────────────────────────────────────┐ 
0.1 │⠀⠀⠀⣸⠉⠉⠓⠢⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
    │⠀⠀⢀⡇⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
    │⠀⠀⡼⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
    │⠀⢠⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
    │⠀⡞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠳⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
    │⢰⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠓⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  0 │⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠲⢤⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
    └─────────────────────────────────────────────┘ 
    0                                          2000 
"""
WarmupLinear(startlr, initlr, warmup, total_iters, schedule) =
    ParameterSchedulers.Sequence(
        ParameterSchedulers.Triangle(λ0 = startlr, λ1 = initlr, period = 2 * warmup) => warmup,
        schedule => total_iters
    )

WarmupCosine(startlr, initlr, finallr, warmup, total_iters) =
    ParameterSchedulers.Sequence(
        ParameterSchedulers.Triangle(λ0 = startlr, λ1 = initlr, period = 2 * warmup) => warmup,
        ParameterSchedulers.CosAnneal(λ0 = finallr, λ1 = initlr, period=total_iters) => total_iters,
        finallr => Inf # to prevent periodicity of cosine
    )


function CreateAnealer(max_value, milestone)
    new_value = it->max_value * min(1f0, it/milestone)
end
