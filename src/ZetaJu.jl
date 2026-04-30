module ZetaJu

using Statistics
using Distributions
using HypothesisTests
using Logging

include("dependencies.jl")
include("ifr_dependencies.jl")
include("ts_dependencies.jl")
include("main.jl")

export zetatest, zetatest2, zetatstest, zetatstest2, ifr

end # module ZetaJu
