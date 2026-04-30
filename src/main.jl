# main.jl - Public API
# Ported from zetapy/main.py (Montijn, Meijer, Heimel)

using Statistics
using Distributions
using HypothesisTests
using Logging

# ──────────────────────────────────────────────────────────────────────────────
# zetatest
# ──────────────────────────────────────────────────────────────────────────────

"""
    zetatest(vecSpikeTimes, arrEventTimes; kwargs...) -> (dblZetaP, dZETA, dRate)

Calculates neuronal responsiveness index ZETA (one-sample, spike-train version).

Reference: Montijn et al. (2021) eLife 10, e71969.

# Arguments
- `vecSpikeTimes::Vector{Float64}` — spike times in seconds
- `arrEventTimes` — event on-times (s), or [T×2] matrix with [onset, offset]

# Keyword arguments
- `dblUseMaxDur=nothing` — window length (default: min inter-event interval)
- `intResampNum=100` — number of jitter resamplings
- `dblJitterSize=2.0` — jitter window size relative to dblUseMaxDur
- `tplRestrictRange=(-Inf, Inf)` — restrict latency search to this range
- `boolStitch=true` — stitch trial data
- `boolDirectQuantile=false` — use empirical quantiles (default: Gumbel)
- `boolReturnRate=false` — whether to compute and return IFR
"""
function zetatest(vecSpikeTimes::Vector{Float64},
                  arrEventTimes;
                  dblUseMaxDur=nothing,
                  intResampNum::Int=100,
                  boolPlot::Bool=false,
                  dblJitterSize::Float64=2.0,
                  tplRestrictRange::Tuple{Float64,Float64}=(-Inf, Inf),
                  boolStitch::Bool=true,
                  boolDirectQuantile::Bool=false,
                  boolReturnRate::Bool=false)

    # placeholder outputs
    dblZetaP_out = 1.0
    vecLatencies    = fill(NaN, 4)
    vecLatencyVals  = fill(NaN, 4)

    dZETA = Dict{String,Any}(
        "dblZetaP"          => 1.0,
        "dblZETA"           => nothing,
        "dblMeanZ"          => nothing,
        "dblMeanP"          => nothing,
        "dblZETADeviation"  => nothing,
        "dblLatencyZETA"    => nothing,
        "intZETAIdx"        => nothing,
        "vecMu_Dur"         => nothing,
        "vecMu_Pre"         => nothing,
        "dblD_InvSign"      => nothing,
        "dblLatencyInvZETA" => nothing,
        "intIdx_InvSign"    => nothing,
        "vecSpikeT"         => nothing,
        "vecRealDeviation"  => nothing,
        "vecRealFrac"       => nothing,
        "vecRealFracLinear" => nothing,
        "cellRandTime"      => nothing,
        "cellRandDeviation" => nothing,
        "dblUseMaxDur"      => nothing,
        "vecLatencies"      => vecLatencies,
        "vecLatencyVals"    => vecLatencyVals,
    )
    dRate = Dict{String,Any}(
        "vecRate"             => nothing,
        "vecT"                => nothing,
        "vecM"                => nothing,
        "vecScale"            => nothing,
        "matMSD"              => nothing,
        "vecV"                => nothing,
        "dblLatencyPeak"      => nothing,
        "dblPeakWidth"        => nothing,
        "vecPeakStartStop"    => nothing,
        "intPeakLoc"          => nothing,
        "vecPeakStartStopIdx" => nothing,
        "dblLatencyPeakOnset" => nothing,
    )

    # validate and prepare spike times
    vecSpikeTimes = sort(vec(vecSpikeTimes))

    # normalise event times
    arrEventTimes_m = _prep_event_times(arrEventTimes)
    vecEventStarts  = arrEventTimes_m[:, 1]

    # check minimum data
    if length(vecSpikeTimes) < 3 || length(vecEventStarts) < 3
        @warn "zetatest: too few spikes or events, returning p=1.0"
        return dblZetaP_out, dZETA, dRate
    end

    # stop supplied?
    boolStopSupplied = size(arrEventTimes_m, 2) > 1
    if boolStopSupplied
        arrEventOnDur = arrEventTimes_m[:, 2] .- arrEventTimes_m[:, 1]
        @assert all(arrEventOnDur .> 0) "at least one event has non-positive duration"
    else
        dblMeanZ = NaN; dblMeanP = NaN
    end

    # window
    if isnothing(dblUseMaxDur)
        dblUseMaxDur = Float64(minimum(diff(vecEventStarts)))
    else
        dblUseMaxDur = Float64(dblUseMaxDur)
        @assert dblUseMaxDur > 0 "dblUseMaxDur must be positive"
    end

    if boolPlot && !boolReturnRate
        boolReturnRate = true
        @warn "zetatest: boolReturnRate set to true because boolPlot=true"
    end

    # ── core ZETA ──
    dZETA_One = calcZetaOne(vecSpikeTimes, vecEventStarts, dblUseMaxDur,
                            intResampNum, boolDirectQuantile,
                            dblJitterSize, boolStitch, false)
    merge!(dZETA, dZETA_One)

    vecSpikeT        = dZETA["vecSpikeT"]
    vecRealDeviation = dZETA["vecRealDeviation"]
    dblZetaP_out     = dZETA["dblZetaP"]
    intZETAIdx       = dZETA["intZETAIdx"]

    if isnothing(intZETAIdx)
        @warn "zetatest: calculation failed, returning p=1.0"
        return dblZetaP_out, dZETA, dRate
    end

    dblLatencyZETA   = vecSpikeT[intZETAIdx]
    dblZETADeviation = vecRealDeviation[intZETAIdx]

    intIdx_InvSign    = argmax(-sign(dblZETADeviation) .* vecRealDeviation)
    dblLatencyInvZETA = vecSpikeT[intIdx_InvSign]
    dblD_InvSign      = vecRealDeviation[intIdx_InvSign]

    # ── mean-rate t-test (only if stop times provided) ──
    if boolStopSupplied
        vecRespBinsDur = sort(vec(arrEventTimes_m))
        # histogram: count spikes in each bin
        vecR = _histogram_bins(vecSpikeTimes, vecRespBinsDur)
        vecD = diff(vecRespBinsDur)

        vecMu_Dur = Float64.(vecR[1:2:end]) ./ vecD[1:2:end]

        dblStart1    = minimum(vecRespBinsDur)
        med_off      = median(vecD[2:2:end])
        dblFirstPreDur = dblStart1 - max(dblStart1 - med_off, 0.0) + eps(Float64)
        dblR1 = Float64(sum((vecSpikeTimes .> (dblStart1 - dblFirstPreDur)) .&
                            (vecSpikeTimes .< dblStart1)))
        vecCounts = vcat([dblR1], Float64.(vecR[2:2:end]))
        vecDurs   = vcat([dblFirstPreDur], vecD[2:2:end])
        vecMu_Pre = vecCounts ./ vecDurs

        dblMeanP = pvalue(OneSampleTTest(vecMu_Dur .- vecMu_Pre))
        dblMeanZ = -quantile(Normal(), dblMeanP / 2.0)

        dZETA["dblMeanZ"]   = dblMeanZ
        dZETA["dblMeanP"]   = dblMeanP
        dZETA["vecMu_Dur"]  = vecMu_Dur
        dZETA["vecMu_Pre"]  = vecMu_Pre
    end

    # ── instantaneous firing rate ──
    if boolReturnRate
        dblMeanRate = length(vecSpikeT) / (dblUseMaxDur * length(vecEventStarts))
        vecRate, dRate_inner = getMultiScaleDeriv(vecSpikeT, vecRealDeviation;
                                                  dblMeanRate=dblMeanRate,
                                                  dblUseMaxDur=dblUseMaxDur)
        merge!(dRate, dRate_inner)

        if !isnothing(vecRate)
            dPeak = getPeak(vecRate, dRate["vecT"]; tplRestrictRange=tplRestrictRange)
            merge!(dRate, dPeak)

            if !isnothing(dRate["dblLatencyPeak"]) && !isnan(dRate["dblLatencyPeak"])
                intZetaIdxRate    = clamp(intZETAIdx - 1,    1, length(vecRate))
                intZetaIdxInvRate = clamp(intIdx_InvSign - 1, 1, length(vecRate))

                dOnset = getOnset(vecRate, dRate["vecT"],
                                  dblLatencyPeak=dRate["dblLatencyPeak"],
                                  tplRestrictRange=tplRestrictRange)
                dRate["dblLatencyPeakOnset"] = dOnset["dblLatencyPeakOnset"]

                vecLatencies   = [dblLatencyZETA, dblLatencyInvZETA,
                                   dRate["dblLatencyPeak"], dOnset["dblLatencyPeakOnset"]]
                vecLatencyVals = [vecRate[intZetaIdxRate], vecRate[intZetaIdxInvRate],
                                   vecRate[dRate["intPeakLoc"]], dOnset["dblValue"]]
            end
        end
    end

    # fill remaining dZETA fields
    dZETA["dblZETADeviation"]  = dblZETADeviation
    dZETA["dblLatencyZETA"]    = dblLatencyZETA
    dZETA["dblD_InvSign"]      = dblD_InvSign
    dZETA["dblLatencyInvZETA"] = dblLatencyInvZETA
    dZETA["intIdx_InvSign"]    = intIdx_InvSign
    dZETA["dblUseMaxDur"]      = dblUseMaxDur
    dZETA["vecLatencies"]      = vecLatencies
    dZETA["vecLatencyVals"]    = vecLatencyVals

    return dblZetaP_out, dZETA, dRate
end

# ──────────────────────────────────────────────────────────────────────────────
# zetatest2
# ──────────────────────────────────────────────────────────────────────────────

"""
    zetatest2(vecSpikeTimes1, arrEventTimes1, vecSpikeTimes2, arrEventTimes2; kwargs...)
        -> (dblZetaP, dZETA)

Two-sample spike-train ZETA test.
"""
function zetatest2(vecSpikeTimes1::Vector{Float64},
                   arrEventTimes1,
                   vecSpikeTimes2::Vector{Float64},
                   arrEventTimes2;
                   dblUseMaxDur=nothing,
                   intResampNum::Int=250,
                   boolPlot::Bool=false,
                   boolDirectQuantile::Bool=false)

    dZETA = Dict{String,Any}(
        "dblZetaP"          => 1.0,
        "dblZETA"           => nothing,
        "dblMeanZ"          => nothing,
        "dblMeanP"          => nothing,
        "dblZETADeviation"  => nothing,
        "dblZetaT"          => nothing,
        "intZetaIdx"        => nothing,
        "vecMu1"            => nothing,
        "vecMu2"            => nothing,
        "dblD_InvSign"      => nothing,
        "dblZetaT_InvSign"  => nothing,
        "intZetaIdx_InvSign"=> nothing,
        "vecSpikeT"         => nothing,
        "vecRealDiff"       => nothing,
        "vecRealFrac1"      => nothing,
        "vecRealFrac2"      => nothing,
        "cellRandTime"      => nothing,
        "cellRandDiff"      => nothing,
        "dblUseMaxDur"      => nothing,
    )

    vecSpikeTimes1 = sort(vec(vecSpikeTimes1))
    vecSpikeTimes2 = sort(vec(vecSpikeTimes2))

    arrET1 = _prep_event_times(arrEventTimes1)
    arrET2 = _prep_event_times(arrEventTimes2)
    vecEventStarts1 = arrET1[:, 1]
    vecEventStarts2 = arrET2[:, 1]

    boolStopSupplied = size(arrET1, 2) > 1 && size(arrET2, 2) > 1

    if boolStopSupplied
        arrEventOnDur1 = arrET1[:, 2] .- arrET1[:, 1]
        arrEventOnDur2 = arrET2[:, 2] .- arrET2[:, 1]
        @assert all(arrEventOnDur1 .> 0) && all(arrEventOnDur2 .> 0) "non-positive event duration"

        if isnothing(dblUseMaxDur)
            dblUseMaxDur = Float64(min(minimum(arrEventOnDur1), minimum(arrEventOnDur2)))
        end
    else
        dblMeanZ = NaN; dblMeanP = NaN
    end

    if isnothing(dblUseMaxDur)
        dblUseMaxDur = Float64(min(minimum(diff(vecEventStarts1)),
                                  minimum(diff(vecEventStarts2))))
    else
        dblUseMaxDur = Float64(dblUseMaxDur)
    end

    dZETA_Two = calcZetaTwo(vecSpikeTimes1, arrET1, vecSpikeTimes2, arrET2,
                            dblUseMaxDur, intResampNum, boolDirectQuantile)
    merge!(dZETA, dZETA_Two)

    vecSpikeT   = dZETA["vecSpikeT"]
    vecRealDiff = dZETA["vecRealDiff"]
    dblZetaP    = dZETA["dblZetaP"]
    intZetaIdx  = dZETA["intZETAIdx"]

    if isnothing(intZetaIdx)
        @warn "zetatest2: calculation failed, returning p=1.0"
        return dZETA["dblZetaP"], dZETA
    end

    dblZetaT        = vecSpikeT[intZetaIdx]
    dblZETADeviation = vecRealDiff[intZetaIdx]

    intZetaIdx_InvSign  = argmax(-sign(dblZETADeviation) .* vecRealDiff)
    dblZetaT_InvSign    = vecSpikeT[intZetaIdx_InvSign]
    dblD_InvSign        = vecRealDiff[intZetaIdx_InvSign]

    if boolStopSupplied
        vecRespBins1 = sort(vec(arrET1))
        vecR1 = _histogram_bins(vecSpikeTimes1, vecRespBins1)
        vecD1 = diff(vecRespBins1)
        vecMu1 = Float64.(vecR1[1:2:end]) ./ vecD1[1:2:end]

        vecRespBins2 = sort(vec(arrET2))
        vecR2 = _histogram_bins(vecSpikeTimes2, vecRespBins2)
        vecD2 = diff(vecRespBins2)
        vecMu2 = Float64.(vecR2[1:2:end]) ./ vecD2[1:2:end]

        dblMeanP = pvalue(EqualVarianceTTest(vecMu1, vecMu2))
        dblMeanZ = -quantile(Normal(), dblMeanP / 2.0)

        dZETA["dblMeanZ"] = dblMeanZ
        dZETA["dblMeanP"] = dblMeanP
        dZETA["vecMu1"]   = vecMu1
        dZETA["vecMu2"]   = vecMu2
    end

    dZETA["dblZETADeviation"]   = dblZETADeviation
    dZETA["dblZetaT"]           = dblZetaT
    dZETA["dblD_InvSign"]       = dblD_InvSign
    dZETA["dblZetaT_InvSign"]   = dblZetaT_InvSign
    dZETA["intZetaIdx_InvSign"] = intZetaIdx_InvSign
    dZETA["dblUseMaxDur"]       = dblUseMaxDur

    return dZETA["dblZetaP"], dZETA
end

# ──────────────────────────────────────────────────────────────────────────────
# zetatstest
# ──────────────────────────────────────────────────────────────────────────────

"""
    zetatstest(vecTime, vecValue, arrEventTimes; kwargs...) -> (dblZetaP, dZETA)

One-sample time-series ZETA test (e.g., calcium imaging).
"""
function zetatstest(vecTime::Vector{Float64},
                    vecValue::Vector{Float64},
                    arrEventTimes;
                    dblUseMaxDur=nothing,
                    intResampNum::Int=100,
                    boolPlot::Bool=false,
                    dblJitterSize::Float64=2.0,
                    boolDirectQuantile::Bool=false,
                    boolStitch::Bool=true)

    dZETA = Dict{String,Any}(
        "dblZetaP"          => 1.0,
        "dblZETA"           => nothing,
        "dblMeanZ"          => nothing,
        "dblMeanP"          => nothing,
        "dblZETADeviation"  => nothing,
        "dblLatencyZETA"    => nothing,
        "intZETAIdx"        => nothing,
        "vecMu_Dur"         => nothing,
        "vecMu_Base"        => nothing,
        "dblD_InvSign"      => nothing,
        "dblLatencyInvZETA" => nothing,
        "intIdx_InvSign"    => nothing,
        "vecRealTime"       => nothing,
        "vecRealDeviation"  => nothing,
        "vecRealFrac"       => nothing,
        "vecRealFracLinear" => nothing,
        "cellRandTime"      => nothing,
        "cellRandDeviation" => nothing,
        "dblUseMaxDur"      => nothing,
    )

    # sort time series
    perm = sortperm(vecTime)
    vecTime  = vecTime[perm]
    vecValue = vecValue[perm]

    arrET = _prep_event_times(arrEventTimes)
    vecEventStarts = arrET[:, 1]

    if length(vecTime) < 3 || length(vecEventStarts) < 3
        @warn "zetatstest: too few entries, returning p=1.0"
        return 1.0, dZETA
    end

    boolStopSupplied = size(arrET, 2) > 1
    if boolStopSupplied
        vecEventStops = arrET[:, 2]
        vecEventOnDur = vecEventStops .- vecEventStarts
        @assert all(vecEventOnDur .> 0) "non-positive event duration"
        vecMu_Dur  = zeros(length(vecEventStops))
        vecMu_Base = zeros(length(vecEventStops))
    else
        dblMeanZ = NaN; dblMeanP = NaN
    end

    if isnothing(dblUseMaxDur)
        dblUseMaxDur = Float64(minimum(diff(vecEventStarts)))
    else
        dblUseMaxDur = Float64(dblUseMaxDur)
    end

    # data extent warnings
    dblDataT0  = minimum(vecTime)
    dblReqT0   = minimum(vecEventStarts) - dblJitterSize * dblUseMaxDur
    if dblDataT0 > dblReqT0
        @warn "zetatstest: leading data before first event is insufficient for maximal jittering"
    end
    dblDataEnd = maximum(vecTime)
    dblReqEnd  = maximum(vecEventStarts) + dblJitterSize * dblUseMaxDur + dblUseMaxDur
    if dblDataEnd < dblReqEnd
        @warn "zetatstest: lagging data after last event is insufficient for maximal jittering"
    end

    dZETA_One = calcTsZetaOne(vecTime, vecValue, arrET, dblUseMaxDur,
                              intResampNum, boolDirectQuantile, dblJitterSize, boolStitch)
    merge!(dZETA, dZETA_One)

    vecRealTime      = dZETA["vecRealTime"]
    vecRealDeviation = dZETA["vecRealDeviation"]
    dblZetaP         = dZETA["dblZetaP"]
    intZETAIdx       = dZETA["intZETAIdx"]

    if isnothing(intZETAIdx)
        @warn "zetatstest: calculation failed, returning p=1.0"
        return dZETA["dblZetaP"], dZETA
    end

    dblLatencyZETA   = vecRealTime[intZETAIdx]
    dblZETADeviation = vecRealDeviation[intZETAIdx]
    intIdx_InvSign    = argmax(-sign(dblZETADeviation) .* vecRealDeviation)
    dblLatencyInvZETA = vecRealTime[intIdx_InvSign]
    dblD_InvSign      = vecRealDeviation[intIdx_InvSign]

    if boolStopSupplied
        intTimeNum = length(vecTime) - 1
        for (intEvent, dblStimStartT) in enumerate(vecEventStarts)
            dblStimStopT = vecEventStops[intEvent]
            dblBaseStopT = dblStimStartT + dblUseMaxDur
            @assert (dblBaseStopT - dblStimStopT) > 0 "event stop times must precede next stimulus start"

            intStartT = max(1, something(findfirst_zeta(vecTime .> dblStimStartT), 2) - 1)
            intStopT  = min(intTimeNum, something(findfirst_zeta(vecTime .> dblStimStopT), intTimeNum+1) + 1)
            intEndT   = min(intTimeNum, something(findfirst_zeta(vecTime .> dblBaseStopT), intTimeNum+1) + 1)

            vecSelectFramesStim = intStartT:intStopT
            vecSelectFramesBase = intStopT:intEndT

            vecMu_Base[intEvent] = mean(vecValue[vecSelectFramesBase])
            vecMu_Dur[intEvent]  = mean(vecValue[vecSelectFramesStim])
        end

        indUseTrials = .!isnan.(vecMu_Dur) .& .!isnan.(vecMu_Base)
        vecMu_Dur  = vecMu_Dur[indUseTrials]
        vecMu_Base = vecMu_Base[indUseTrials]

        dblMeanP = pvalue(OneSampleTTest(vecMu_Dur .- vecMu_Base))
        dblMeanZ = -quantile(Normal(), dblMeanP / 2.0)

        dZETA["dblMeanZ"]   = dblMeanZ
        dZETA["dblMeanP"]   = dblMeanP
        dZETA["vecMu_Dur"]  = vecMu_Dur
        dZETA["vecMu_Base"] = vecMu_Base
    end

    dZETA["dblZETADeviation"]  = dblZETADeviation
    dZETA["dblLatencyZETA"]    = dblLatencyZETA
    dZETA["dblD_InvSign"]      = dblD_InvSign
    dZETA["dblLatencyInvZETA"] = dblLatencyInvZETA
    dZETA["intIdx_InvSign"]    = intIdx_InvSign
    dZETA["dblUseMaxDur"]      = dblUseMaxDur

    return dZETA["dblZetaP"], dZETA
end

# ──────────────────────────────────────────────────────────────────────────────
# zetatstest2
# ──────────────────────────────────────────────────────────────────────────────

"""
    zetatstest2(vecTime1, vecValue1, arrEventTimes1,
                vecTime2, vecValue2, arrEventTimes2; kwargs...)
        -> (dblZetaP, dZETA)

Two-sample time-series ZETA test.
"""
function zetatstest2(vecTime1::Vector{Float64},
                     vecValue1::Vector{Float64},
                     arrEventTimes1,
                     vecTime2::Vector{Float64},
                     vecValue2::Vector{Float64},
                     arrEventTimes2;
                     dblUseMaxDur=nothing,
                     intResampNum::Int=250,
                     boolPlot::Bool=false,
                     boolDirectQuantile::Bool=false,
                     dblSuperResFactor::Float64=100.0)

    dZETA = Dict{String,Any}(
        "dblZetaP"               => 1.0,
        "dblZETA"                => nothing,
        "dblMeanZ"               => nothing,
        "dblMeanP"               => nothing,
        "dblZETADeviation"       => nothing,
        "dblZETATime"            => nothing,
        "intZETAIdx"             => nothing,
        "vecMu1"                 => nothing,
        "vecMu2"                 => nothing,
        "dblZETADeviation_InvSign"=> nothing,
        "dblZETATime_InvSign"    => nothing,
        "intZETAIdx_InvSign"     => nothing,
        "vecRefTime"             => nothing,
        "vecRealDiff"            => nothing,
        "matRandDiff"            => nothing,
        "vecRealFrac1"           => nothing,
        "vecRealFrac2"           => nothing,
        "matTracePerTrial1"      => nothing,
        "matTracePerTrial2"      => nothing,
        "dblUseMaxDur"           => nothing,
    )

    # sort time series
    p1 = sortperm(vecTime1); vecTime1 = vecTime1[p1]; vecValue1 = vecValue1[p1]
    p2 = sortperm(vecTime2); vecTime2 = vecTime2[p2]; vecValue2 = vecValue2[p2]

    arrET1 = _prep_event_times(arrEventTimes1)
    arrET2 = _prep_event_times(arrEventTimes2)
    vecEventStarts1 = arrET1[:, 1]
    vecEventStarts2 = arrET2[:, 1]

    if length(vecTime1) < 3 || length(vecEventStarts1) < 3 ||
       length(vecTime2) < 3 || length(vecEventStarts2) < 3
        @warn "zetatstest2: too few entries, returning p=1.0"
        return 1.0, dZETA
    end

    if isnothing(dblUseMaxDur)
        dblUseMaxDur = Float64(min(minimum(diff(vecEventStarts1)),
                                  minimum(diff(vecEventStarts2))))
    else
        dblUseMaxDur = Float64(dblUseMaxDur)
    end

    @assert (minimum(vecEventStarts1) > minimum(vecTime1) &&
             maximum(vecEventStarts1) < (maximum(vecTime1) - dblUseMaxDur) &&
             minimum(vecEventStarts2) > minimum(vecTime2) &&
             maximum(vecEventStarts2) < (maximum(vecTime2) - dblUseMaxDur)) "events exist outside of data period"

    dZETA_Two = calcTsZetaTwo(vecTime1, vecValue1, arrET1,
                              vecTime2, vecValue2, arrET2,
                              dblSuperResFactor, dblUseMaxDur,
                              intResampNum, boolDirectQuantile)
    merge!(dZETA, dZETA_Two)

    vecRefTime  = dZETA["vecRefTime"]
    vecRealDiff = dZETA["vecRealDiff"]
    intZETAIdx  = dZETA["intZETAIdx"]

    if isnothing(intZETAIdx)
        @warn "zetatstest2: calculation failed, returning p=1.0"
        return dZETA["dblZetaP"], dZETA
    end

    dblZETATime      = vecRefTime[intZETAIdx]
    dblZETADeviation = vecRealDiff[intZETAIdx]

    intZETAIdx_InvSign       = argmax(-sign(dblZETADeviation) .* vecRealDiff)
    dblZETATime_InvSign      = vecRefTime[intZETAIdx_InvSign]
    dblZETADeviation_InvSign = vecRealDiff[intZETAIdx_InvSign]

    # mean-rate test when stop times provided
    boolStopSupplied = size(arrET1, 2) > 1 && size(arrET2, 2) > 1
    if boolStopSupplied
        vecMu1 = _ts_mean_diff(vecTime1, vecValue1, arrET1, dblUseMaxDur)
        vecMu2 = _ts_mean_diff(vecTime2, vecValue2, arrET2, dblUseMaxDur)

        dblMeanP = pvalue(EqualVarianceTTest(vecMu1, vecMu2))
        dblMeanZ = -quantile(Normal(), dblMeanP / 2.0)

        dZETA["dblMeanZ"] = dblMeanZ
        dZETA["dblMeanP"] = dblMeanP
        dZETA["vecMu1"]   = vecMu1
        dZETA["vecMu2"]   = vecMu2
    end

    dZETA["dblZETADeviation"]        = dblZETADeviation
    dZETA["dblZETATime"]             = dblZETATime
    dZETA["dblZETADeviation_InvSign"]= dblZETADeviation_InvSign
    dZETA["dblZETATime_InvSign"]     = dblZETATime_InvSign
    dZETA["intZETAIdx_InvSign"]      = intZETAIdx_InvSign
    dZETA["dblUseMaxDur"]            = dblUseMaxDur

    return dZETA["dblZetaP"], dZETA
end

# ──────────────────────────────────────────────────────────────────────────────
# ifr
# ──────────────────────────────────────────────────────────────────────────────

"""
    ifr(vecSpikeTimes, vecEventTimes; kwargs...) -> (vecTime, vecRate, dIFR)

Returns instantaneous firing rates without running the full ZETA test.
"""
function ifr(vecSpikeTimes::Vector{Float64},
             vecEventTimes;
             dblUseMaxDur=nothing,
             dblSmoothSd::Float64=2.0,
             dblMinScale::Union{Float64,Nothing}=nothing,
             dblBase::Float64=1.5,
             boolParallel::Bool=false)

    vecTime = Float64[]
    vecRate = Float64[]
    dIFR = Dict{String,Any}(
        "vecTime"      => vecTime,
        "vecRate"      => vecRate,
        "vecDeviation" => Float64[],
        "vecScale"     => Float64[],
    )

    vecSpikeTimes = sort(vec(vecSpikeTimes))
    arrET = _prep_event_times(vecEventTimes)
    vecEventStarts = arrET[:, 1]

    if length(vecSpikeTimes) < 3 || length(vecEventStarts) < 3
        @warn "ifr: too few spikes or events"
        return vecTime, vecRate, dIFR
    end

    if isnothing(dblUseMaxDur)
        dblUseMaxDur = Float64(minimum(diff(vecEventStarts)))
    else
        dblUseMaxDur = Float64(dblUseMaxDur)
    end

    vecThisDeviation, _, _, vecThisSpikeTimes =
        getTempOffsetOne(vecSpikeTimes, vecEventStarts, dblUseMaxDur)

    if length(vecThisDeviation) < 3
        @warn "ifr: too few spikes, returning empty"
        return vecTime, vecRate, dIFR
    end

    intSpikeNum = length(vecThisSpikeTimes)
    dblMeanRate = intSpikeNum / (dblUseMaxDur * length(vecEventStarts))

    vecRate_out, dMSD = getMultiScaleDeriv(vecThisSpikeTimes, vecThisDeviation;
                                           dblSmoothSd=dblSmoothSd,
                                           dblMinScale=dblMinScale,
                                           dblBase=dblBase,
                                           dblMeanRate=dblMeanRate,
                                           dblUseMaxDur=dblUseMaxDur)

    vecTime_out = dMSD["vecT"]
    dIFR = Dict{String,Any}(
        "vecTime"      => vecTime_out,
        "vecRate"      => vecRate_out,
        "vecDeviation" => vecThisDeviation,
        "vecScale"     => dMSD["vecScale"],
    )

    return vecTime_out, vecRate_out, dIFR
end

# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

"""Normalise arrEventTimes to a Matrix{Float64} with at least 1 column."""
function _prep_event_times(arrEventTimes)
    if arrEventTimes isa AbstractVector
        return reshape(Float64.(arrEventTimes), :, 1)
    else
        m = Float64.(arrEventTimes)
        r, c = size(m)
        if c >= 3 && r < 3
            m = permutedims(m)
        end
        return m
    end
end

"""Count spikes in each bin (like np.histogram with explicit bin edges)."""
function _histogram_bins(vecSpikeTimes::Vector{Float64}, binEdges::AbstractVector{Float64})
    n = length(binEdges) - 1
    counts = zeros(Int, n)
    for s in vecSpikeTimes
        idx = searchsortedlast(binEdges, s)
        if 1 <= idx <= n
            counts[idx] += 1
        end
    end
    return counts
end

"""Compute mean activity diff (stim minus base) per trial for zetatstest2."""
function _ts_mean_diff(vecT, vecV, arrET, dblUseMaxDur)
    intMaxRep  = size(arrET, 1)
    intTimeNum = length(vecT)
    vecMu_Base = fill(NaN, intMaxRep)
    vecMu_Dur  = fill(NaN, intMaxRep)

    for intEvent in 1:intMaxRep
        dblStimStartT = arrET[intEvent, 1]
        dblStimStopT  = arrET[intEvent, 2]
        dblBaseStopT  = dblStimStartT + dblUseMaxDur

        intStartT = max(1, something(findfirst_zeta(vecT .> dblStimStartT), 2) - 1)
        intStopT  = min(intTimeNum, something(findfirst_zeta(vecT .> dblStimStopT), intTimeNum+1) + 1)
        intEndT   = min(intTimeNum, something(findfirst_zeta(vecT .> dblBaseStopT), intTimeNum+1) + 1)

        vecSelectFramesStim = intStartT:intStopT
        vecSelectFramesBase = intStopT:intEndT

        vecMu_Base[intEvent] = mean(vecV[vecSelectFramesBase])
        vecMu_Dur[intEvent]  = mean(vecV[vecSelectFramesStim])
    end

    return vecMu_Dur .- vecMu_Base
end
