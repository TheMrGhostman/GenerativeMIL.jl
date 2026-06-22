"""
    create_beta_scheduler(beta_cfg::Union{Number, Dict})

Vytvoří β_scheduler z konfigurace (číslo nebo Dict).

Pokud je vstup číslo, vrátí konstantní funkci.
Pokud je Dict, vytvoří anealer podle typu.

# Typ konfigurací

- `beta_cfg = 1.0` → konstanta
- `Dict(:type => "constant", :value => 1.0)` → konstanta
- `Dict(:type => "linear", :max_value => 1.0, :milestone => 500)` → lineárně roste od 0 v prvních 500 epoch
- `Dict(:type => "exponential", :initial => 0.0, :final => 1.0, :decay_rate => 0.95)` → exponenciální
- `Dict(:type => "cosine", :max_value => 1.0, :total_epochs => 1000)` → cosine annealing
- `Dict(:type => "sigmoidal", :max_value => 1.0, :milestone => 800, :slope_factor => 0.015)` → sigmoida
- `Dict(:type => "sigmoidal_cyclical", :max_value => 1.0, :beta_warmup => 1e-4, :warmup_epochs => 180, :rise_epochs => 100, :hold_epochs => 100, :cycles => 4)` → warmup + cykly

# Vrací

Funkci `epoch -> β_value` (Float32) kterou lze volat každou epochu.
"""
function create_beta_scheduler(beta_cfg::Union{Number, Dict})
    # Pokud je to jen číslo, vrátit konstantu
    beta_cfg isa Number && return _ -> Float32(beta_cfg)
    
    type = get(beta_cfg, :type, "constant")
    
    if type == "constant"
        value = get(beta_cfg, :value, get(beta_cfg, :max_value, 1f0))
        return _ -> Float32(value)
    elseif type == "step_linear"
        # Use ParameterSchedulers.Sequence:
        #  - hold `initial` for `m1` epochs
        #  - Triangle from `initial` -> `max_value` over (m2-m1) epochs
        #  - hold `max_value` afterwards
        initial = get(beta_cfg, :initial, 0.0)
        max_value = get(beta_cfg, :max_value, 1.0)
        milestones = get(beta_cfg, :milestones, [100, 450])
        @assert length(milestones) == 2 && milestones[1] < milestones[2] "milestones for step_linear must contain exactly 2 values and be in ascending order"
        m1, m2 = Int.(floor.(milestones))
        ramp_len = m2 - m1
        seq = ParameterSchedulers.Sequence(
            Float32(initial) => m1,
            ParameterSchedulers.Triangle(λ0 = Float32(initial), λ1 = Float32(max_value), period = 2 * ramp_len) => ramp_len,
            Float32(max_value) => Inf
        )
        return epoch -> Float32(seq(epoch))
    elseif type == "linear"
        max_value = get(beta_cfg, :max_value, 1f0)
        milestone = get(beta_cfg, :milestone, 500)
        return epoch -> Float32(max_value * min(epoch / milestone, 1f0))
    
    elseif type == "exponential"
        initial = get(beta_cfg, :initial, 0f0)
        final = get(beta_cfg, :final, 1f0)
        decay_rate = get(beta_cfg, :decay_rate, 0.95)
        return epoch -> Float32(final + (initial - final) * decay_rate^epoch)
    
    elseif type == "cosine"
        max_value = get(beta_cfg, :max_value, 1f0)
        total_epochs = get(beta_cfg, :total_epochs, 1000)
        return epoch -> Float32(max_value * (1 + cos(π * epoch / total_epochs)) / 2)

    elseif type == "sigmoidal"
        max_value = get(beta_cfg, :max_value, 1f0)
        slope_factor = get(beta_cfg, :slope_factor, 0.015f0)
        milestone = get(beta_cfg, :milestone, 800) # centre of sigmoid of sigmoid
        return SigmoidSchedule(max_value, milestone, slope_factor)

    elseif type in ("sigmoidal_cyclical", "cyclical_sigmoidal", "sigmoid_cyclical")
        max_value = get(beta_cfg, :max_value, 1f0)
        beta_warmup = get(beta_cfg, :beta_warmup, 0f0)
        warmup_epochs = Int(get(beta_cfg, :warmup_epochs, 180))
        rise_epochs = Int(get(beta_cfg, :rise_epochs, 100))
        hold_epochs = Int(get(beta_cfg, :hold_epochs, 100))
        cycles = Int(get(beta_cfg, :cycles, 4))
        slope_factor = get(beta_cfg, :slope_factor, 12f0/rise_epochs)
        return CyclicalSigmoidSchedule(max_value, beta_warmup, warmup_epochs, rise_epochs, hold_epochs, cycles; slope_factor=slope_factor)

    else
        @warn "Unknown β scheduler type: $type, using constant 1.0"
        return _ -> 1f0
    end
end


function CreateLrScheduler(sch_name, lr, max_iters; milestones=[0.02, 0.8], scale=5, kwargs...)
    @assert max_iters > 0 "max_iters must be positive"
    @assert length(milestones) == 2 "milestones must contain exactly 2 values"
    @assert 0.0 <= milestones[1] <= milestones[2] <= 1.0 "milestones must satisfy 0 <= m1 <= m2 <= 1"

    sch = sch_name isa Symbol ? String(sch_name) : sch_name
    sch_norm = sch isa AbstractString ? lowercase(replace(sch, "_" => "", "-" => "")) : sch

    if sch_norm in [false, "false", "none", "constant"]
        scheduler = _ -> Float32(lr)
        # constant learning rate
    elseif sch_norm == "warmupcosine"
        scheduler = WarmupCosine(1e-7, lr * scale, lr, Int(milestones[1] * max_iters), Int(milestones[2] * max_iters))
        # from 0 to milestones[1] there is linear LR warmup to lr*scale
        # from milestones[1] to milestones[2] there is cosine decay back to lr
        # from milestones[2] to the end there is constant LR
    elseif sch_norm == "linear2ndhalf"
        scheduler = it -> Float32(lr * min(1.0, 2.0 - it / (0.5 * max_iters)))
        # learning rate decay: (0%,50%) -> 1, (50%,100%) -> linear(1->0)
    else
        error("Unknown lr scheduler name: $(sch_name). Supported: false, \"Linear2ndHalf\", \"WarmupCosine\"")
    end

    return scheduler
end


"""
    create_lr_scheduler(cfg, lr, max_iters)

Vytvori learning-rate scheduler z konfigurace.

Podporovane vstupy:
- `nothing`/`false`/`"false"`/`"none"`: bez scheduleru (vraci `nothing`)
- `"Linear2ndHalf"` nebo `"WarmupCosine"`: pouzije `CreateLrScheduler`
- `Dict` s klicem `:type` (+ volitelne `:milestones`, `:scale`)

Vraci `nothing` nebo scheduler volany jako `scheduler(epoch)`.
"""
function create_lr_scheduler(cfg, lr, max_iters)
    if isnothing(cfg)
        return nothing
    end

    if cfg in (false, "false", "none")
        return nothing
    end

    if cfg isa AbstractString || cfg isa Symbol
        return CreateLrScheduler(cfg, lr, max_iters)
    end

    if cfg isa Dict
        sch_type = get(cfg, :type, false)
        sch_type in (false, "false", "none") && return nothing
        milestones = get(cfg, :milestones, [0.02, 0.8])
        scale = get(cfg, :scale, 5)
        return CreateLrScheduler(sch_type, lr, max_iters; milestones=milestones, scale=scale)
    end

    error("Unsupported lr scheduler config type: $(typeof(cfg))")
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



struct SigmoidSchedule{T<:AbstractFloat, C<:Int, S<:AbstractFloat} <: AbstractScheduler
    max_value::T
    center::C
    slope::S
end

function (s::SigmoidSchedule)(step::Integer)
    x = s.slope * (step - s.center)
    return s.max_value / (1 + exp(-x))
end

function CyclicalSigmoidSchedule(max_value, beta_warmup, warmup_epochs, rise_epochs, hold_epochs, cycles; slope_factor=12f0/rise_epochs)#0.08f0
    @assert warmup_epochs >= 0 "warmup_epochs must be non-negative"
    @assert rise_epochs > 0 "rise_epochs must be positive"
    @assert hold_epochs >= 0 "hold_epochs must be non-negative"
    @assert cycles >= 1 "cycles must be at least 1"

    max_value = Float32(max_value)
    beta_warmup = Float32(beta_warmup)
    slope_factor = Float32(slope_factor)
    warmup_center = max(1, warmup_epochs ÷ 2)
    rise_center = max(1, rise_epochs ÷ 2)

    schedules = Any[beta_warmup => warmup_epochs]

    for _ in 1:cycles
        push!(schedules, SigmoidSchedule(max_value, rise_center, slope_factor) => rise_epochs)
        push!(schedules, max_value => hold_epochs)
    end

    push!(schedules, max_value => Inf)
    return ParameterSchedulers.Sequence(schedules...)
end

#cs = CyclicalSigmoidSchedule(1, 1e-5, 150, 200, 50, 4; slope_factor=13/200f0)