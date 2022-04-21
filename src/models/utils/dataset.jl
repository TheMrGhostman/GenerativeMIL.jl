using Flux

#a = randn(3,1)
#b = rpad(a, 10, 0) # expand array to length 10 with zeros
# b ~ (10,1)
#x = [randn(3) for i =1:3]
#y = cat([reshape(rpad(xx, len, 0), (len,1)) for xx in x])

a = [randn(Float32, 3,j) for j in rand(100:300, 32)]
a_mask = [ones(size(x)) for x in a]
max_set = maximum(size.(a))[end]
b = map(x->Array(PaddedView(0, x, (3, max_set))), a)
b_mask = map(x->Array(PaddedView(0, x, (3, max_set))), a_mask)
c = cat(b..., dims=3)
c_mask = cat(b_mask..., dims=3) .> 0 # mask as BitArray
#Array(PaddedView(0, a, (3, 300)))
#@time b = reduce(vcat,map(i->Array(PaddedView(0, i, (3, 300))), a))