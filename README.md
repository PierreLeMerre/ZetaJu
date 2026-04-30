# ZetaJu

Julia port of the ZETA-test family for neuronal responsiveness.

The original code is written in MATLAB by Jorrit Montijn (https://github.com/Herseninstituut/zetatest). The most up-to-date Python port is at https://github.com/Herseninstituut/zetapy, maintained by Guido Meijer and Alexander Heimel. This Julia port was produced with Claude; details of the porting process are in [PORTING_PROMPT.md](PORTING_PROMPT.md).

The pre-print describing data-stitching, the time-series ZETA-test, and the two-sample tests:
https://www.biorxiv.org/content/10.1101/2023.10.30.564780v1

The original ZETA-test article (peer-reviewed):
https://elifesciences.org/articles/71969

> The ZETA-test for spiking data has been extensively tested on real and artificial data and is peer-reviewed. The time-series and two-sample variants are described in the pre-print and are not yet peer-reviewed. Please report bugs on the Issues page.

![zeta_image](https://user-images.githubusercontent.com/15422591/135059690-2d7f216a-726e-4080-a4ec-2b3fae78e10c.png)

---

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/PierreLeMerre/ZetaJu")
```

Or for local development:

```julia
Pkg.develop(path="/path/to/ZetaJu")
```

Then load with:

```julia
using ZetaJu
```

---

## Functions

### `zetatest` — one-sample spike-train ZETA test

Tests whether a neuron's spike timing is locked to experimental events.

```julia
dblZetaP, dZETA, dRate = zetatest(vecSpikeTimes, arrEventTimes;
    dblUseMaxDur      = nothing,   # analysis window (default: min inter-event interval)
    intResampNum      = 100,       # number of jitter resamplings
    dblJitterSize     = 2.0,       # jitter window relative to dblUseMaxDur
    tplRestrictRange  = (-Inf, Inf),
    boolStitch        = true,
    boolDirectQuantile= false,
    boolReturnRate    = false)
```

**Inputs**
- `vecSpikeTimes` — `Vector{Float64}` of spike times in seconds
- `arrEventTimes` — event onset times (`Vector{Float64}`), or `T×2` matrix of `[onset offset]`

**Returns**
- `dblZetaP` — p-value
- `dZETA` — `Dict{String,Any}` with fields including:
  - `"dblZETA"` — ZETA statistic
  - `"dblZetaP"` — p-value (same as first return)
  - `"dblLatencyZETA"` — latency of peak deviation
  - `"vecLatencies"` — `[latencyZETA, latencyInvZETA, peakTime, onsetTime]`
  - `"vecRealDeviation"` — deviation vector
  - `"vecSpikeT"` — trial-relative spike times
  - `"dblUseMaxDur"` — window used
  - `"dblMeanP"`, `"dblMeanZ"` — mean-rate t-test (only when stop times are provided)
- `dRate` — `Dict{String,Any}` with instantaneous firing rate (populated when `boolReturnRate=true`):
  - `"vecT"` — time vector
  - `"vecRate"` — firing rate vector
  - `"dblLatencyPeak"`, `"dblLatencyPeakOnset"` — peak and onset latencies

**Example**

```julia
using ZetaJu

# Basic test (onset times only)
p, dZETA, dRate = zetatest(spike_times, event_times)
println("p = $p")

# With stop times (enables mean-rate t-test)
arr = hcat(event_onsets, event_offsets)   # T×2 matrix
p, dZETA, dRate = zetatest(spike_times, arr; dblUseMaxDur=0.5, boolReturnRate=true)

# Access latencies
lats = dZETA["vecLatencies"]   # [ZETA latency, inv-sign latency, peak, onset]

# Access instantaneous firing rate
if !isnothing(dRate["vecT"])
    using Plots
    plot(dRate["vecT"], dRate["vecRate"])
end
```

---

### `zetatest2` — two-sample spike-train ZETA test

Tests whether two neurons (or one neuron under two conditions) respond differently.

```julia
dblZetaP, dZETA = zetatest2(vecSpikeTimes1, arrEventTimes1,
                             vecSpikeTimes2, arrEventTimes2;
    dblUseMaxDur       = nothing,
    intResampNum       = 250,
    boolDirectQuantile = false)
```

---

### `zetatstest` — one-sample time-series ZETA test

For calcium imaging, EEG, LFP, or any continuous signal.

```julia
dblZetaP, dZETA = zetatstest(vecTime, vecValue, arrEventTimes;
    dblUseMaxDur       = nothing,
    intResampNum       = 100,
    dblJitterSize      = 2.0,
    boolStitch         = true,
    boolDirectQuantile = false)
```

**Inputs**
- `vecTime` — `Vector{Float64}` of timestamps
- `vecValue` — `Vector{Float64}` of signal values (same length as `vecTime`)
- `arrEventTimes` — same as `zetatest`

**Returns** `(dblZetaP, dZETA)` where `dZETA` contains `"vecRealTime"` and `"vecRealDeviation"` instead of spike-train equivalents.

---

### `zetatstest2` — two-sample time-series ZETA test

```julia
dblZetaP, dZETA = zetatstest2(vecTime1, vecValue1, arrEventTimes1,
                               vecTime2, vecValue2, arrEventTimes2;
    dblUseMaxDur       = nothing,
    intResampNum       = 250,
    dblSuperResFactor  = 100.0,
    boolDirectQuantile = false)
```

---

### `ifr` — instantaneous firing rate

Computes the multi-scale derivative firing rate without running the ZETA test. Use this as you would a PSTH function.

```julia
vecTime, vecRate, dIFR = ifr(vecSpikeTimes, vecEventTimes;
    dblUseMaxDur = nothing,
    dblSmoothSd  = 2.0,
    dblMinScale  = nothing,
    dblBase      = 1.5)
```

**Example**

```julia
t, r, _ = ifr(spike_times, event_times)
plot(t, r, xlabel="Time (s)", ylabel="Firing rate (Hz)")
```

---

## Rationale for ZETA

Neurophysiological studies depend on reliable quantification of whether and when a neuron responds to stimulation. Current methods require arbitrary parameter choices (e.g. bin size) that can change results and reduce replicability. Many methods also only detect mean-rate modulated cells. The ZETA-test family is parameter-free, sensitive to a broad class of response types, and works for both spike trains and continuous signals.

As shown in the papers, ZETA outperforms optimally-binned ANOVAs, t-tests, and model-based approaches: it includes more cells in real neurophysiological data at a similar false-positive rate.

---

## Latency estimation

For computing response latency, consider the LatenZy test (https://github.com/Herseninstituut/latenZy), which is based on the ZETA-test. The `vecLatencies` field returned by `zetatest` provides four latency estimates when `boolReturnRate=true`: ZETA peak, inverse-sign peak, IFR peak, and IFR onset.

---

## Differences from the Python version

| Python (`zetapy`) | Julia (`ZetaJu`) |
|---|---|
| `dblUseMaxDur` is a positional argument | `dblUseMaxDur` is a **keyword** argument |
| Returns `(dblZetaP, dZETA, dRate)` | Same |
| `dZETA['vecLatencies']` is a numpy array | `dZETA["vecLatencies"]` is a `Vector{Float64}` |
| `intLatencyPeaks` parameter | Not needed — always returns 4 latencies |
| `boolVerbose` / `boolPlot` | `boolPlot` accepted but does nothing (no plotting) |

---

## Dependencies

- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
- [HypothesisTests.jl](https://github.com/JuliaStats/HypothesisTests.jl)
- Statistics (stdlib)
- Logging (stdlib)
