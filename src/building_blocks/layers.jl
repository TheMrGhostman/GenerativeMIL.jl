struct SplitLayer # TODO unite with MaskedGaussian
    μ::Flux.Dense
    σ::Flux.Dense
end

Flux.@functor SplitLayer

function SplitLayer(in::Int, out::NTuple{2, Int}, acts::NTuple{2, Function})
	SplitLayer(
		Flux.Dense(in, out[1], acts[1]),
		Flux.Dense(in, out[2], acts[2])
	)
end

function (m::SplitLayer)(x)
	return (m.μ(x), m.σ(x))
end

