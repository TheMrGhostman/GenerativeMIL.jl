"""
Jednoduchý příklad použití sinkhorn_divergence_loss s rrule
"""

using Flux, Zygote, CUDA, Statistics

# Přidej src do path
push!(LOAD_PATH, "/home/zorekmat/MIL/GenerativeMIL/src")
using GenerativeMIL

println("=" ^ 70)
println("PŘÍKLAD: Sinkhorn Divergence Loss s Custom RRule")
println("=" ^ 70)

# =============================================================================
# PŘÍKLAD 1: Základní forward pass
# =============================================================================
println("\n[PŘÍKLAD 1] Forward pass - spočítáme loss")
println("-" ^ 70)

# Vytvoř test data
x = rand(Float32, 3, 64, 2)  # (dim=3, n_points=64, batch_size=2)
y = rand(Float32, 3, 128, 2) # (dim=3, n_points=128, batch_size=2)

# Loss hyperparameter
ε = 0.1f0

# Spočítej loss
loss = sinkhorn_divergence_loss(x, y, ε; maxiter=50)
println("✓ Loss spočítána")
println("  Loss value: $loss")
println("  x shape: $(size(x))")
println("  y shape: $(size(y))")

# =============================================================================
# PŘÍKLAD 2: Gradienty přes Zygote
# =============================================================================
println("\n[PŘÍKLAD 2] Gradienty přes Zygote.jl")
println("-" ^ 70)

loss_fn(x_in, y_in) = sinkhorn_divergence_loss(x_in, y_in, ε; maxiter=50)

# Spočítej gradienty
∇x, ∇y = gradient(loss_fn, x, y)

println("✓ Gradienty spočítány")
println("  ∇x shape: $(size(∇x))")
println("  ∇y shape: $(size(∇y))")
println("  ∇x mean magnitude: $(mean(abs, ∇x))")
println("  ∇y mean magnitude: $(mean(abs, ∇y))")

# =============================================================================
# PŘÍKLAD 3: Použití s GPU
# =============================================================================
println("\n[PŘÍKLAD 3] GPU verze (CUDA)")
println("-" ^ 70)

# Přesun dat na GPU (pokud je dostupný)
if CUDA.functional()
    println("✓ CUDA je dostupný")
    x_gpu = x |> cu
    y_gpu = y |> cu
    
    # Spočítej loss na GPU
    loss_gpu = sinkhorn_divergence_loss(x_gpu, y_gpu, ε; maxiter=50)
    println("  Loss na GPU: $loss_gpu")
    
    # Gradienty na GPU
    ∇x_gpu, ∇y_gpu = gradient(loss_fn, x_gpu, y_gpu)
    println("  ✓ GPU gradienty spočítány")
else
    println("⚠ CUDA není dostupný, vynechávám GPU příklad")
end

# =============================================================================
# PŘÍKLAD 4: Optimizační cyklus
# =============================================================================
println("\n[PŘÍKLAD 4] Jednoduchý optimizační cyklus")
println("-" ^ 70)

# Parametry
params = (x = copy(x), y = copy(y))
opt = Flux.Adam(0.001)
opt_state = Flux.setup(opt, params)

# Trénovací cyklus
num_epochs = 3
for epoch in 1:num_epochs
    loss_val, grads = Flux.withgradient(
        (p) -> sinkhorn_divergence_loss(p.x, p.y, ε; maxiter=30), 
        params
    )
    
    # Update parametrů
    Flux.update!(opt_state, params, grads[1])
    
    println("Epoch $epoch | Loss: $(round(loss_val, digits=6))")
end

println("✓ Optimizační cyklus hotov")

# =============================================================================
# PŘÍKLAD 5: Numerická verifikace gradientů
# =============================================================================
println("\n[PŘÍKLAD 5] Numerická verifikace gradientů (malé data)")
println("-" ^ 70)

# Menší data pro rychlejší výpočet
x_small = rand(Float32, 2, 8, 1)
y_small = rand(Float32, 2, 8, 1)

# Analytické gradienty
∇x_ana, ∇y_ana = gradient(
    (x_in, y_in) -> sinkhorn_divergence_loss(x_in, y_in, ε; maxiter=30),
    x_small, y_small
)

# Numerické gradienty (finite differences) - pouze pro jeden parametr
eps_fd = 1e-4f0
x_pert = copy(x_small)
x_pert[1, 1, 1] += eps_fd

loss_orig = sinkhorn_divergence_loss(x_small, y_small, ε; maxiter=30)
loss_pert = sinkhorn_divergence_loss(x_pert, y_small, ε; maxiter=30)
∇x_num = (loss_pert - loss_orig) / eps_fd

# Porovnání
rel_error = abs(∇x_num - ∇x_ana[1, 1, 1]) / (abs(∇x_num) + eps_fd)
println("Numerical gradient check:")
println("  Position [1,1,1]:")
println("    Numerické ∇x: $∇x_num")
println("    Analytické ∇x: $(∇x_ana[1, 1, 1])")
println("    Relativní chyba: $rel_error")

if rel_error < 0.01f0
    println("  ✓ Gradienty jsou správné (chyba < 1%)")
else
    println("  ⚠ Gradienty se liší (chyba > 1%)")
end

println("\n" * "=" ^ 70)
println("Všechny příklady hotovy! ✓")
println("=" ^ 70)
