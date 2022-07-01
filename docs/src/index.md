

# WeightedOnlineStats.jl

A version onf OnlineStats.jl allowing observations with custom weights.


## Basics

### Creating

```@repl index
using WeightedOnlineStats
m = WeightedMean()
x = randn(100);
w = randn(100);
```

### Updating

```@repl index
fit!(m, x, w)
```

### Merging

```@repl index
m2 = WeightedMean()
x2 = rand(100);
w2 = rand(100);
fit!(m2, x2, w2)
merge!(m, m2)
```
