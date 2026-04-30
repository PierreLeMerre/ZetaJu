# Prompt: Port `zetapy` Python Package to Julia (`ZetaJu`)

## Context and Goal

Port the Python package `zetapy` (https://github.com/JorritMontijn/zetapy) to a proper Julia package called `ZetaJu`. Full credit to the original authors: Jorrit Montijn, Guido Meijer, Alexander Heimel, and Jan Alexej Heimel.

The package implements the **ZETA test** (Zenith of Event-based Time-locked Anomalies), a parameter-free statistical test for neuronal responsiveness used in systems neuroscience. It detects whether a neuron's firing is time-locked to experimental events (e.g., stimulus onsets).

The source of truth is the Python package at `/Users/pierre/Documents/Python/zetapy/zetapy/`. You have direct file access to it. **Read every source file before writing any Julia code.**

---

## Source File Map

Read all of these before starting:

| Python file | Role |
|---|---|
| `main.py` | Public API: 5 top-level functions |
| `dependencies.py` | Core spike-train zeta math |
| `ifr_dependencies.py` | Instantaneous firing rate (IFR) computation |
| `ts_dependencies.py` | Time-series zeta variant |
| `__init__.py` | Package exports |
| `test_zetatest.py` | Test cases — use to verify correctness |
| `example.py` | Usage examples |

---

## Julia Package Structure to Create

```
ZetaJu/
├── Project.toml
├── src/
│   ├── ZetaJu.jl            # module entry, exports
│   ├── main.jl              # public API (zetatest, zetatest2, zetatstest, zetatstest2, ifr)
│   ├── dependencies.jl      # core spike-train zeta internals
│   ├── ifr_dependencies.jl  # multi-scale derivative / IFR
│   └── ts_dependencies.jl   # time-series zeta internals
└── test/
    └── runtests.jl          # port of test_zetatest.py
```

Work in `/Users/pierre/Documents/Julia/zetaju/`.

---

## Public API to Implement

### 1. `zetatest(vecSpikeTimes, arrEventTimes; kwargs...) → (dblZetaP, dZETA, dRate)`

One-sample spike-train ZETA test.

**Required inputs:**
- `vecSpikeTimes::Vector{Float64}` — spike times in seconds
- `arrEventTimes::Union{Vector{Float64}, Matrix{Float64}}` — event onset times (s), or [T×2] matrix with [onset, offset]

**Keyword arguments (all have defaults):**
- `dblUseMaxDur=nothing` — analysis window length (default: minimum inter-event interval)
- `intResampNum=100` — number of jitter resamplings
- `boolPlot=false` — plotting (skip in Julia port, always false)
- `dblJitterSize=2.0` — jitter window size relative to `dblUseMaxDur`
- `tplRestrictRange=(-Inf, Inf)` — restrict latency search range
- `boolStitch=true` — whether to stitch spike data across trials
- `boolDirectQuantile=false` — use empirical quantiles instead of Gumbel approximation
- `boolReturnRate=false` — whether to return IFR dict

**Returns:**
- `dblZetaP::Float64` — p-value
- `dZETA::Dict{String,Any}` — full result dict (keep all field names identical to Python)
- `dRate::Dict{String,Any}` — IFR fields (empty if `boolReturnRate=false`)

### 2. `zetatest2(vecSpikeTimes1, arrEventTimes1, vecSpikeTimes2, arrEventTimes2; kwargs...) → (dblZetaP, dZETA)`

Two-sample spike-train ZETA test.

**Required inputs:** two sets of spike times + event times (same format as zetatest)

**Keyword arguments:**
- `dblUseMaxDur=nothing`
- `intResampNum=250`
- `boolPlot=false`
- `boolDirectQuantile=false`

### 3. `zetatstest(vecTime, vecValue, arrEventTimes; kwargs...) → (dblZetaP, dZETA)`

One-sample time-series ZETA test (e.g., for calcium imaging data).

**Required inputs:**
- `vecTime::Vector{Float64}` — timestamps of the time series
- `vecValue::Vector{Float64}` — values (e.g., dF/F₀)
- `arrEventTimes` — same format as above

**Keyword arguments:**
- `dblUseMaxDur=nothing`
- `intResampNum=100`
- `boolPlot=false`
- `dblJitterSize=2.0`
- `boolDirectQuantile=false`
- `boolStitch=true`

### 4. `zetatstest2(vecTime1, vecValue1, arrEventTimes1, vecTime2, vecValue2, arrEventTimes2; kwargs...) → (dblZetaP, dZETA)`

Two-sample time-series ZETA test.

**Keyword arguments:**
- `dblUseMaxDur=nothing`
- `intResampNum=250`
- `boolPlot=false`
- `boolDirectQuantile=false`
- `dblSuperResFactor=100`

### 5. `ifr(vecSpikeTimes, vecEventTimes; kwargs...) → (vecTime, vecRate, dIFR)`

Returns instantaneous firing rates without running the ZETA test.

**Keyword arguments:**
- `dblUseMaxDur=nothing`
- `dblSmoothSd=2.0`
- `dblMinScale=nothing`
- `dblBase=1.5`

---

## Internal Functions to Port

### `dependencies.jl`

- `calcZetaOne(vecSpikeTimes, arrEventTimes, dblUseMaxDur, intResampNum, boolDirectQuantile, dblJitterSize, boolStitch, boolParallel)` → `Dict`
- `calcZetaTwo(vecSpikeTimes1, arrEventTimes1, vecSpikeTimes2, arrEventTimes2, dblUseMaxDur, intResampNum, boolDirectQuantile)` → `Dict`
- `getTempOffsetOne(vecSpikeTimes, vecEventTimes, dblUseMaxDur)` → `(vecDeviation, vecFrac, vecFracLinear, vecSpikeT)`
- `getTempOffsetTwo(cellTimePerSpike1, cellTimePerSpike2, dblUseMaxDur; boolFastInterp=false, vecSpikeT=nothing)` → `(vecSpikeT, vecDiff, vecFrac1, vecSpikeTimes1, vecFrac2, vecSpikeTimes2)`
- `getSpikesInTrial(vecSpikes, vecTrialStarts, dblMaxDur)` → `(cellTrialPerSpike, cellTimePerSpike)`
- `getZetaP(arrMaxD, vecMaxRandD, boolDirectQuantile)` → `(arrZetaP, arrZETA)`
- `getGumbel(dblE, dblV, arrX)` → `(arrP, arrZ)`
- `getSpikeT(vecSpikeTimes, vecEventTimes, dblUseMaxDur)` → `Vector{Float64}`
- `getUniqueSpikes(vecSpikeTimes)` → `Vector{Float64}`
- `getPseudoSpikeVectors(vecSpikeTimes, vecEventTimes, dblWindowDur; boolDiscardEdges=false)` → `(vecPseudoSpikeTimes, vecPseudoEventT)`
- `findfirst_zeta(condition_array)` → `Union{Int, Nothing}` — renamed to avoid clash with Julia's built-in `findfirst`
- `my_randint(high; size=nothing)` — MATLAB `randi` equivalent
- `my_randperm(n, k)` → `Vector{Int}`

### `ifr_dependencies.jl`

- `getMultiScaleDeriv(vecT, vecV; dblSmoothSd=2.0, dblMinScale=nothing, dblBase=1.5, dblMeanRate=1.0, dblUseMaxDur=nothing, boolParallel=false)` → `(vecRate, dMSD::Dict)`
- `calcSingleMSD(dblScale, vecT, vecV)` → `Vector{Float64}`
- `getPeak(vecData, vecT; tplRestrictRange=(-Inf, Inf), intSwitchZ=1)` → `Dict`
- `getOnset(vecData, vecT; dblLatencyPeak=nothing, tplRestrictRange=nothing)` → `Dict`

### `ts_dependencies.jl`

- `calcTsZetaOne(vecTimestamps, vecData, arrEventTimes, dblUseMaxDur, intResampNum, boolDirectQuantile, dblJitterSize, boolStitch)` → `Dict`
- `calcTsZetaTwo(vecTimestamps1, vecData1, arrEventTimes1, vecTimestamps2, vecData2, arrEventTimes2, dblSuperResFactor, dblUseMaxDur, intResampNum, boolDirectQuantile)` → `Dict`
- `getTimeseriesOffsetOne(vecTimestamps, vecData, vecEventStartT, dblUseMaxDur; dblSuperResFactor=100)` → `(vecDeviation, vecFrac, vecFracLinear, vecTime)`
- `getTimeseriesOffsetTwo(matTracePerTrial1, matTracePerTrial2)` → `(vecDiff, vecFrac1, vecFrac2)`
- `getPseudoTimeSeries(vecTimestamps, vecData, vecEventTimes, dblWindowDur)` → `(vecPseudoTime, vecPseudoData, vecPseudoEventT)`
- `getTsRefT(vecTimestamps, vecEventStartT, dblUseMaxDur; dblSuperResFactor=1)` → `Vector{Float64}`
- `getInterpolatedTimeSeries(vecTimestamps, vecData, vecEventStartT, vecRefTime)` → `(vecRefTime, matTracePerTrial)`
- `uniquetol(array_in, dblTol)` → `Vector{Float64}`

---

## Critical Python → Julia Translation Notes

### 1. INDEXING — the most important source of bugs

Python is **0-indexed**; Julia is **1-indexed**.

- `x[0]` → `x[1]`
- `x[-1]` → `x[end]`
- `np.arange(0, n)` can map to either `0:n-1` or `1:n` — trace the intent carefully
- `argmax` in Python returns a 0-based index; Julia's `argmax` returns a 1-based index — same value, different meaning, always check downstream use
- Any code that does `vecT[intIdx]` where `intIdx` comes from Python's `argmax`/`findfirst` needs +1 offset review

### 2. `findfirst_zeta`

The Python helper `findfirst(indArray)` returns `None` or the **0-based** index of the first `True`. Port it as:

```julia
function findfirst_zeta(condition::AbstractArray{Bool})
    idx = findfirst(condition)
    return idx  # Julia's findfirst already returns nothing or 1-based Int
end
```

But note: all call sites in Python do `findfirst(vecT > x)` which is a boolean array — Julia must broadcast: `findfirst(vecT .> x)`.

### 3. NumPy → Julia equivalents

| Python/NumPy | Julia |
|---|---|
| `np.sort(x)` | `sort(x)` |
| `np.argsort(x)` | `sortperm(x)` |
| `np.concatenate([a, b])` | `vcat(a, b)` |
| `np.zeros(n)` | `zeros(n)` |
| `np.ones(n)` | `ones(n)` |
| `fill(nan)` → `np.empty(n); x.fill(np.nan)` | `fill(NaN, n)` |
| `np.linspace(a, b, n)` | `LinRange(a, b, n)` or `collect(range(a, b, length=n))` |
| `np.diff(x)` | `diff(x)` |
| `abs.(x)` for arrays | `abs.(x)` |
| `np.argmax(x)` | `argmax(x)` (1-based!) |
| `np.max(x)` | `maximum(x)` |
| `np.min(x)` | `minimum(x)` |
| `np.mean(x)` | `mean(x)` (Statistics) |
| `np.median(x)` | `median(x)` (Statistics) |
| `np.std(x)` | `std(x)` (Statistics) |
| `np.var(x, ddof=1)` | `var(x)` (Julia's default is n-1, same as ddof=1) |
| `np.sum(x)` | `sum(x)` |
| `np.cumsum(x)` | `cumsum(x)` |
| `np.ptp(x)` | `maximum(x) - minimum(x)` |
| `np.unique(x)` | `unique(sort(x))` |
| `np.interp(xnew, x, y)` | `LinearInterpolation(x, y)(xnew)` from Interpolations.jl |
| `np.histogram(x, bins=b)` | `fit(Histogram, x, b)` from StatsBase |
| `np.logical_and(a, b)` | `a .& b` |
| `np.logical_or(a, b)` | `a .| b` |
| `~mask` | `.!mask` |
| `x.flatten()` | `vec(x)` |
| `np.reshape(x, (-1, 1))` | `reshape(x, :, 1)` |
| `np.reshape(x, -1)` | `vec(x)` |
| `x.shape` | `size(x)` |
| `x.shape[0]` | `size(x, 1)` |
| `x.size` | `length(x)` |
| `x.T` | `transpose(x)` or `permutedims(x)` |
| `np.isinf(x)` | `isinf.(x)` |
| `np.isnan(x)` | `isnan.(x)` |
| `np.finfo(float).eps` | `eps(Float64)` |
| `np.append(a, b)` | `vcat(a, b)` |
| `np.pad(m, pad, 'edge')` | manual: `vcat(repeat(m[1:1, :], pad), m, repeat(m[end:end, :], pad))` |
| `x[0::2]` (stride-2 from 0) | `x[1:2:end]` |
| `x[1::2]` (stride-2 from 1) | `x[2:2:end]` |
| `x[start:stop]` (exclusive stop) | `x[start+1:stop]` (Julia end is inclusive) |

### 4. SciPy → Julia equivalents

| Python/SciPy | Julia |
|---|---|
| `scipy.stats.norm.ppf(p)` | `quantile(Normal(0,1), p)` from Distributions.jl |
| `scipy.stats.zscore(x)` | `(x .- mean(x)) ./ std(x)` |
| `scipy.stats.ttest_rel(a, b)[1]` | `pvalue(OneSampleTTest(a .- b))` from HypothesisTests.jl |
| `scipy.stats.ttest_ind(a, b)[1]` | `pvalue(EqualVarianceTTest(a, b))` from HypothesisTests.jl |
| `scipy.signal.find_peaks(x, ...)` | Use `Peaks.jl` or implement manually |
| `scipy.signal.convolve2d(m, k, 'valid')` | `DSP.conv` or manual implementation |
| `scipy.interpolate.interp1d` | `LinearInterpolation` from Interpolations.jl |

### 5. Python `dict` → Julia `Dict{String, Any}`

Use `Dict{String, Any}` throughout to match the Python code exactly and keep all field name strings identical. This is the cleanest approach for a first port.

```julia
dZETA = Dict{String, Any}(
    "dblZetaP" => 1.0,
    "dblZETA"  => nothing,
    "dblMeanZ" => nothing,
    # ...
)
```

Access with `dZETA["dblZetaP"]`.

### 6. `None` → `nothing`

All Python `None` values map to Julia's `nothing`. Use `isnothing(x)` to check.

### 7. Python `assert` → Julia

Replace `assert condition, "message"` with `@assert condition "message"` or `condition || throw(ArgumentError("message"))`.

### 8. Python `logging.warning` → Julia

Use `@warn "message"`.

### 9. Python `isinstance(x, bool)` → Julia

Use `isa(x, Bool)`.

### 10. Broadcasting

Python numpy operations on arrays often broadcast automatically. In Julia, **all element-wise operations on arrays require the dot syntax**: `a .* b`, `a .+ b`, `f.(x)`, etc. Be thorough — missing a dot is a common bug.

### 11. `issubclass(x.dtype.type, np.floating)` → Julia

Use `eltype(x) <: AbstractFloat`.

### 12. List/cell arrays (`cell*` variables)

Python uses lists of arrays for things like `cellTimePerSpike`. In Julia, use `Vector{Vector{Float64}}`.

### 13. `math.sqrt`, `math.exp`, `math.factorial`

Just use Julia's built-in `sqrt`, `exp`, `factorial`.

---

## Key Algorithmic Details to Preserve Exactly

### The ZETA statistic (core of `calcZetaOne`)

1. Optionally build pseudo-spike times via data stitching (`getPseudoSpikeVectors`)
2. Compute `getTempOffsetOne`: collect all spikes from all trials into a single vector of trial-relative times, sort, compute cumulative fraction vs. linear (uniform) fraction → deviation vector, mean-subtract
3. `dblMaxD = maximum(abs.(vecRealDeviation))`
4. For `intResampNum` resamplings: jitter event times by `±dblJitterSize * dblUseMaxDur * uniform[-1,1]`, recompute deviation, mean-subtract, record max
5. Fit Gumbel (or empirical quantiles) to `vecMaxRandD` → p-value

### Gumbel distribution (`getGumbel`)

```julia
const EULER_MASCHERONI = 0.5772156649015328606065120900824

function getGumbel(dblE, dblV, arrX)
    dblBeta = sqrt(6) * sqrt(dblV) / π
    dblMode = dblE - dblBeta * EULER_MASCHERONI
    fGumbelCDF(x) = exp(-exp(-(x - dblMode) / dblBeta))
    arrGumbelCDF = fGumbelCDF.(arrX)
    arrP = 1.0 .- arrGumbelCDF
    arrZ = -quantile.(Normal(), arrP ./ 2)
    # handle Inf: approximate for large X
    for i in eachindex(arrZ)
        if isinf(arrZ[i])
            arrP[i] = exp((dblMode - arrX[i]) / dblBeta)
            arrZ[i] = -quantile(Normal(), arrP[i] / 2)
        end
    end
    return arrP, arrZ
end
```

### `getUniqueSpikes`

Adds tiny jitter to duplicate spike times iteratively. Do not simplify — the iterative loop is needed for edge cases:

```julia
function getUniqueSpikes(vecSpikeTimes::Vector{Float64})
    vecSpikeTimes = sort(vecSpikeTimes)
    dblUniqueOffset = eps(eltype(vecSpikeTimes))
    dblShift = dblUniqueOffset
    indDuplicates = vcat(false, diff(vecSpikeTimes) .< dblUniqueOffset)
    while any(indDuplicates)
        vecNotUnique = vecSpikeTimes[indDuplicates]
        n = length(vecNotUnique)
        vecJitter = vcat(1 .+ 9 .* rand(n), -1 .- 9 .* rand(n))
        vecJitter = dblShift .* vecJitter[my_randperm(length(vecJitter), n)]
        vecSpikeTimes[indDuplicates] .+= vecJitter
        vecSpikeTimes = sort(vecSpikeTimes)
        indDuplicates = vcat(false, diff(vecSpikeTimes) .< dblUniqueOffset)
        dblShift *= 2  # avoid infinite loop
    end
    return vecSpikeTimes
end
```

### `getPseudoSpikeVectors`

Data stitching creates a pseudo-continuous spike train from trial windows. The boundary handling for the first and last trial is non-trivial. Study the Python code carefully, trace through the logic, and pay attention to:
- `intLastUsedSample` tracking
- The "add beginning" and "add end" sections that extend the pseudo-spike vector beyond the trial windows

### Multi-scale derivative (`calcSingleMSD`)

O(N²) loop — expected, do not optimize:
```julia
function calcSingleMSD(dblScale, vecT, vecV)
    intN = length(vecT)
    vecMSD = zeros(intN)
    for intS in 1:intN
        dblT = vecT[intS]
        dblMinEdge = dblT - dblScale/2
        dblMaxEdge = dblT + dblScale/2
        intIdxMinT = findfirst_zeta(vecT .> dblMinEdge)
        isnothing(intIdxMinT) && (intIdxMinT = 1)
        intIdxMaxT_raw = findfirst_zeta(vecT .> dblMaxEdge)
        intIdxMaxT = isnothing(intIdxMaxT_raw) ? intN : intIdxMaxT_raw - 1
        if intIdxMinT > intIdxMaxT
            dblD = 0.0
        else
            if intIdxMinT == intIdxMaxT && intIdxMinT > 1 && intIdxMinT < intN
                intIdxMaxT = intIdxMinT + 1
                intIdxMinT = intIdxMinT - 1
            end
            dbl_dT = max(dblScale, vecT[intIdxMaxT] - vecT[intIdxMinT])
            dblD = (vecV[intIdxMaxT] - vecV[intIdxMinT]) / dbl_dT
        end
        vecMSD[intS] = dblD
    end
    return vecMSD
end
```

### 2D smoothing in `getMultiScaleDeriv`

The Python code pads `matMSD` and applies `convolve2d` with a Gaussian kernel columnwise. In Julia:
1. Build the Gaussian kernel vector `vecFilt`
2. Pad `matMSD` vertically (edge-padding, `intPadSize` rows top and bottom)
3. Convolve each column of `matMSD` with `vecFilt` using `DSP.conv` (take the 'valid' part)

---

## Required Julia Packages

```toml
[deps]
Distributions = "..."
Statistics = "..."
LinearAlgebra = "..."
Interpolations = "..."
StatsBase = "..."
HypothesisTests = "..."
DSP = "..."
Peaks = "..."
Logging = "..."
```

Run `Pkg.add(["Distributions", "Interpolations", "StatsBase", "HypothesisTests", "DSP", "Peaks"])` to install non-stdlib packages.

---

## Naming Conventions

**Keep all variable names identical to the Python source**, including the Hungarian notation prefixes:
- `dbl` → Float64 scalars
- `int` → Int scalars  
- `vec` → 1D arrays
- `mat` → 2D arrays
- `bool` → Bool
- `cell` → Vector of arrays
- `str` → String

This makes it easy to cross-reference with the original during debugging.

---

## What NOT to Port

- `plot_dependencies.py` — matplotlib plotting. Skip entirely.
- `legacy/` folder — old versions, skip.
- `boolParallel=true` paths — not implemented even in Python, always pass `false`.
- The `boolTest = True` debug paths in `calcZetaOne` and `calcTsZetaOne` — skip.

---

## Testing Strategy

Write `test/runtests.jl` that:

1. **Basic smoke test** — call `zetatest` on minimal synthetic data and confirm it returns without error
2. **Significance test** — generate a Poisson spike train with a clear event-locked rate increase, call `zetatest`, assert `dblZetaP < 0.05`
3. **Non-significance test** — pure Poisson noise, assert `dblZetaP` is not always < 0.05
4. **Dict fields** — verify all expected keys exist in `dZETA` return dict
5. **`zetatest2`** — two conditions with different rates, assert result is significant
6. **`zetatstest`** — synthetic calcium-like trace with event-locked responses
7. **`ifr`** — verify it returns `vecTime`, `vecRate` of matching length

---

## Suggested Implementation Order

1. **`dependencies.jl`**: `findfirst_zeta` → `my_randint` → `my_randperm` → `getSpikeT` → `getUniqueSpikes` → `getPseudoSpikeVectors` → `getTempOffsetOne` → `getSpikesInTrial` → `getTempOffsetTwo` → `getGumbel` → `getZetaP` → `calcZetaOne` → `calcZetaTwo`
2. **`ifr_dependencies.jl`**: `calcSingleMSD` → `getMultiScaleDeriv` → `getPeak` → `getOnset`
3. **`ts_dependencies.jl`**: `uniquetol` → `getTsRefT` → `getInterpolatedTimeSeries` → `getTimeseriesOffsetOne` → `getTimeseriesOffsetTwo` → `getPseudoTimeSeries` → `calcTsZetaOne` → `calcTsZetaTwo`
4. **`main.jl`**: `zetatest` → `zetatest2` → `zetatstest` → `zetatstest2` → `ifr`
5. **`ZetaJu.jl`**: module wrapper and exports
6. **`Project.toml`** + **`test/runtests.jl`**

---

## Final Checklist Before Declaring Done

- [ ] All 5 public functions implemented with correct signatures and keyword argument defaults
- [ ] All internal functions ported in the correct files
- [ ] Every 0-indexed Python array access converted to 1-indexed Julia correctly
- [ ] `nothing` used instead of `None` everywhere; `isnothing()` used for checks
- [ ] Broadcasting (`.`) applied to all element-wise array operations
- [ ] `Dict{String, Any}` used for all output dicts; field name strings are identical to Python
- [ ] Gumbel p-value formula verified with Euler-Mascheroni constant
- [ ] `getUniqueSpikes` handles duplicates iteratively
- [ ] `getPseudoSpikeVectors` boundary logic is correct (beginning and end sections)
- [ ] `calcSingleMSD` O(N²) loop is correct
- [ ] 2D smoothing convolution in `getMultiScaleDeriv` uses edge-padded columns
- [ ] No plotting code included
- [ ] Tests pass on synthetic data
- [ ] `Project.toml` lists all dependencies with UUIDs
