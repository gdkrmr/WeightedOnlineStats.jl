using Documenter, WeightedOnlineStats

makedocs(sitename = "WeightedOnlineStats.jl")

cname = open("docs/build/CNAME", "w")
write(cname, "www.guido-kraemer.com")
close(cname)

deploydocs(repo = "github.com/gdkrmr/WeightedOnlineStats.jl.git",
           push_preview = true)
