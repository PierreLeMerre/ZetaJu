# ts_dependencies.jl - Time-series ZETA internals
# Ported from zetapy/ts_dependencies.py (Montijn, Heimel)

using Statistics
using Logging

"""
    uniquetol(array_in, dblTol) -> Vector{Float64}

Return unique values within tolerance (equivalent to MATLAB uniquetol).
"""
function uniquetol(array_in::AbstractArray, dblTol::Float64)
    floored = unique(floor.(Int64, vec(array_in) ./ dblTol))
    return Float64.(floored) .* dblTol
end

"""
    getTsRefT(vecTimestamps, vecEventStartT, dblUseMaxDur; dblSuperResFactor=1)
        -> Vector{Float64}

Build a common reference time vector (trial-relative timestamps).
"""
function getTsRefT(vecTimestamps::AbstractVector{Float64},
                   vecEventStartT::AbstractVector{Float64},
                   dblUseMaxDur::Float64;
                   dblSuperResFactor::Float64=1.0)

    vecEventStartT = sort(vec(Float64.(vecEventStartT)))
    intTimeNum = length(vecTimestamps) - 1

    cellRefT = Vector{Vector{Float64}}()

    for dblStartT in vecEventStartT
        intBegin_raw = findfirst_zeta(vecTimestamps .> dblStartT)
        intStartT = isnothing(intBegin_raw) ? 1 : max(1, intBegin_raw - 1)

        dblStopT = dblStartT + dblUseMaxDur
        intEnd_raw = findfirst_zeta(vecTimestamps .> dblStopT)
        intStopT = isnothing(intEnd_raw) ? intTimeNum : min(intTimeNum, intEnd_raw)

        vecSelectSamples = intStartT:intStopT
        push!(cellRefT, vecTimestamps[vecSelectSamples] .- dblStartT)
    end

    if dblSuperResFactor == 1.0
        # use the longest trial's ref times, rounded
        intUseEntry = argmax(length.(cellRefT))
        vecRefT = vec(cellRefT[intUseEntry])
        dblMedDiff = median(diff(vecRefT))
        vecTime = round.(vecRefT ./ dblMedDiff; digits=1) .* dblMedDiff
    else
        dblSampInterval = median(diff(vecTimestamps))
        dblTol = dblSampInterval / 100.0
        all_vals = sort(vcat(cellRefT...))
        vecTime = uniquetol(all_vals, dblTol)
    end

    return vecTime
end

"""
    getInterpolatedTimeSeries(vecTimestamps, vecData, vecEventStartT, vecRefTime)
        -> (vecRefTime, matTracePerTrial)

Interpolate time-series data onto a common reference timeframe for each trial.
"""
function getInterpolatedTimeSeries(vecTimestamps::AbstractVector{Float64},
                                   vecData::AbstractVector{Float64},
                                   vecEventStartT::AbstractVector{Float64},
                                   vecRefTime::AbstractVector{Float64})

    vecRefTime_out = Float64.(vec(vecRefTime))
    vecTimestamps  = Float64.(vec(vecTimestamps))
    vecData        = Float64.(vec(vecData))

    intTrials = length(vecEventStartT)
    intRefLen = length(vecRefTime_out)
    matTracePerTrial = zeros(Float64, intTrials, intRefLen)

    for (intTrial, dblStartT) in enumerate(vecEventStartT)
        intBegin_raw = findfirst_zeta(vecTimestamps .> (dblStartT + vecRefTime_out[1]))
        if isnothing(intBegin_raw)
            error("getInterpolatedTimeSeries: no timestamps after trial start")
        end
        intStartT = max(1, intBegin_raw - 1)

        intEnd_raw = findfirst_zeta(vecTimestamps .> (dblStartT + vecRefTime_out[end]))
        intStopT = isnothing(intEnd_raw) ? length(vecTimestamps) : min(length(vecTimestamps), intEnd_raw + 1)

        vecSelectSamples = intStartT:intStopT
        vecUseTimes = vecTimestamps[vecSelectSamples]
        vecUseData  = vecData[vecSelectSamples]

        vecUseInterpT = vecRefTime_out .+ dblStartT
        matTracePerTrial[intTrial, :] = _interp1_vec(vecUseTimes, vecUseData, vecUseInterpT)
    end

    return vecRefTime_out, matTracePerTrial
end

"""
    getTimeseriesOffsetOne(vecTimestamps, vecData, vecEventStartT, dblUseMaxDur;
                           dblSuperResFactor=100)
        -> (vecDeviation, vecFrac, vecFracLinear, vecTime)
"""
function getTimeseriesOffsetOne(vecTimestamps::AbstractVector{Float64},
                                vecData::AbstractVector{Float64},
                                vecEventStartT::AbstractVector{Float64},
                                dblUseMaxDur::Float64;
                                dblSuperResFactor::Float64=100.0)

    vecTime = getTsRefT(vecTimestamps, vecEventStartT, dblUseMaxDur;
                        dblSuperResFactor=dblSuperResFactor)

    vecTime_out, matTracePerTrial = getInterpolatedTimeSeries(
        vecTimestamps, vecData, vecEventStartT, vecTime)

    indKeepPoints = (vecTime_out .>= 0.0) .& (vecTime_out .<= dblUseMaxDur)
    vecTime_out      = vecTime_out[indKeepPoints]
    matTracePerTrial = matTracePerTrial[:, indKeepPoints]

    vecMeanTrace = vec(mean(matTracePerTrial, dims=1))
    s = sum(vecMeanTrace)
    vecThisFrac = cumsum(vecMeanTrace) ./ (s == 0.0 ? 1.0 : s)

    dblMeanVal = mean(vecMeanTrace)
    vecThisFracLinear = range(dblMeanVal, sum(vecMeanTrace); length=length(vecMeanTrace)) ./
                        (s == 0.0 ? 1.0 : s)
    vecThisFracLinear = collect(vecThisFracLinear)

    vecDeviation  = vecThisFrac .- vecThisFracLinear
    vecDeviation .-= mean(vecDeviation)

    return vecDeviation, vecThisFrac, vecThisFracLinear, vecTime_out
end

"""
    getTimeseriesOffsetTwo(matTracePerTrial1, matTracePerTrial2)
        -> (vecDiff, vecFrac1, vecFrac2)
"""
function getTimeseriesOffsetTwo(matTracePerTrial1::AbstractMatrix{Float64},
                                matTracePerTrial2::AbstractMatrix{Float64})

    # mean across trials (nanmean)
    vecMeanTrace1 = [mean(filter(!isnan, matTracePerTrial1[:, j])) for j in axes(matTracePerTrial1, 2)]
    vecMeanTrace2 = [mean(filter(!isnan, matTracePerTrial2[:, j])) for j in axes(matTracePerTrial2, 2)]

    vecThisFrac1 = cumsum(vecMeanTrace1)
    vecThisFrac2 = cumsum(vecMeanTrace2)

    vecDeviation = vecThisFrac1 .- vecThisFrac2
    vecThisDiff  = vecDeviation .- mean(vecDeviation)

    return vecThisDiff, vecThisFrac1, vecThisFrac2
end

"""
    getPseudoTimeSeries(vecTimestamps, vecData, vecEventTimes, dblWindowDur)
        -> (vecPseudoTime, vecPseudoData, vecPseudoEventT)

Stitch trial windows into pseudo-continuous time series.
"""
function getPseudoTimeSeries(vecTimestamps::AbstractVector{Float64},
                             vecData::AbstractVector{Float64},
                             vecEventTimes::AbstractVector{Float64},
                             dblWindowDur::Float64)

    perm = sortperm(vecTimestamps)
    vecTimestamps = Float64.(vec(vecTimestamps[perm]))
    vecData       = Float64.(vec(vecData[perm]))
    vecEventTimes = sort(Float64.(vec(vecEventTimes)))

    intSamples = length(vecTimestamps)
    intTrials  = length(vecEventTimes)
    dblMedianDur = median(diff(vecTimestamps))

    cellPseudoTime = Vector{Union{Nothing,Vector{Float64}}}()
    cellPseudoData = Vector{Union{Nothing,Vector{Float64}}}()
    vecPseudoEventT = fill(NaN, intTrials)
    dblPseudoEventT = 0.0
    dblStartNextAtT = 0.0
    intLastUsedSample = 0    # 0 means none used
    intFirstSample = nothing
    dblPseudoT0 = 0.0

    for (intTrial, dblEventT) in enumerate(vecEventTimes)
        # match MATLAB 2024-09-11
        raw_start = findfirst_zeta(vecTimestamps .> dblEventT)
        intStartSample = isnothing(raw_start) ? 1 : max(1, raw_start - 1)
        intEndSample_raw = findfirst_zeta(vecTimestamps .> (dblEventT + dblWindowDur))
        intEndSample = isnothing(intEndSample_raw) ? intStartSample : intEndSample_raw

        vecElig = collect(intStartSample:intEndSample)
        indUse  = (vecElig .>= 1) .& (vecElig .<= intSamples)
        vecUseSamples = vecElig[indUse]

        # first/last trial: extend to edge
        if intTrial == 1
            vecUseSamples = isempty(vecUseSamples) ? Int[] : collect(1:vecUseSamples[end])
        end
        if intTrial == intTrials
            vecUseSamples = isempty(vecUseSamples) ? Int[] : collect(vecUseSamples[1]:intSamples)
        end

        vecUseT = isempty(vecUseSamples) ? Float64[] : vecTimestamps[vecUseSamples]

        # remove overlap
        if !isempty(vecUseSamples)
            indOverlap = vecUseSamples .<= intLastUsedSample
            if any(indOverlap)
                vecUseSamples = vecUseSamples[.!indOverlap]
                vecUseT = isempty(vecUseSamples) ? Float64[] : vecTimestamps[vecUseSamples]
            end
        end

        if isempty(vecUseSamples)
            vecLocalPseudoT = nothing
            vecLocalPseudoV = nothing
            # advance pseudo event time by gap
            if intLastUsedSample > 0
                dblPseudoEventT = dblEventT - vecTimestamps[intLastUsedSample] + dblStartNextAtT
            end
        else
            intLastUsedSample = vecUseSamples[end]
            vecLocalPseudoV = vecData[vecUseSamples]
            vecLocalPseudoT = vecUseT .- vecUseT[1] .+ dblStartNextAtT
            dblPseudoEventT = dblEventT - vecUseT[1] + dblStartNextAtT

            if length(vecTimestamps) > intLastUsedSample
                dblStepEnd = vecTimestamps[intLastUsedSample+1] - vecTimestamps[intLastUsedSample]
            else
                dblStepEnd = dblMedianDur
            end
            dblStartNextAtT = vecLocalPseudoT[end] + dblStepEnd
        end

        if intTrial == 1 && !isempty(vecUseSamples)
            intFirstSample = vecUseSamples[1]
            dblPseudoT0    = isnothing(vecLocalPseudoT) ? 0.0 : vecLocalPseudoT[1]
        end

        push!(cellPseudoTime, vecLocalPseudoT)
        push!(cellPseudoData, vecLocalPseudoV)
        vecPseudoEventT[intTrial] = dblPseudoEventT
    end

    # add beginning
    if !isnothing(intFirstSample)
        dblT1 = vecTimestamps[intFirstSample]
        intT0_raw = findfirst_zeta(vecTimestamps .> (dblT1 - dblWindowDur))
        if !isnothing(intT0_raw) && intFirstSample > 1
            dblStepBegin = vecTimestamps[intFirstSample] - vecTimestamps[intFirstSample-1]
            vecSampAddBeginning = max(1, intT0_raw-1):intFirstSample-1
            vsub = vecTimestamps[vecSampAddBeginning]
            rng = maximum(vsub) - minimum(vsub)
            vecAddT = vsub .- vsub[1] .+ dblPseudoT0 .- dblStepBegin .- rng
            push!(cellPseudoTime, vecAddT)
            push!(cellPseudoData, vecData[vecSampAddBeginning])
        end
    end

    # add end
    dblLastEnd = vecEventTimes[end] + dblWindowDur
    intFindTail = findfirst_zeta(vecTimestamps .> dblLastEnd)
    if isnothing(intFindTail)
        error("getPseudoTimeSeries: dblMaxDur is too large — tail of final event exceeds data. Include more data, shorten dblMaxDur, or remove the last event.")
    else
        dblTn  = vecTimestamps[intLastUsedSample]
        intTn_raw = findfirst_zeta(vecTimestamps .> dblTn)
        if !isnothing(intTn_raw) && (intTn_raw - 1) > intLastUsedSample
            vecSampAddEnd = (intLastUsedSample+1):intTn_raw
            push!(cellPseudoTime,
                vecTimestamps[vecSampAddEnd] .- vecTimestamps[vecSampAddEnd[1]] .+ dblStartNextAtT)
            push!(cellPseudoData, vecData[vecSampAddEnd])
        end
    end

    # combine
    allT = Float64[]
    allV = Float64[]
    for (t, v) in zip(cellPseudoTime, cellPseudoData)
        if !isnothing(t) && !isempty(t)
            append!(allT, t)
            append!(allV, v)
        end
    end

    perm2 = sortperm(allT)
    vecPseudoTime = reshape(allT[perm2], :, 1)
    vecPseudoData = reshape(allV[perm2], :, 1)

    return vecPseudoTime, vecPseudoData, vecPseudoEventT
end

# ──────────────────────────────────────────────────────────────────────────────
# Core TS ZETA calculators
# ──────────────────────────────────────────────────────────────────────────────

"""
    calcTsZetaOne(vecTimestamps, vecData, arrEventTimes, dblUseMaxDur,
                  intResampNum, boolDirectQuantile, dblJitterSize, boolStitch)
        -> Dict{String,Any}
"""
function calcTsZetaOne(vecTimestamps::AbstractVector{Float64},
                       vecData::AbstractVector{Float64},
                       arrEventTimes::AbstractArray,
                       dblUseMaxDur::Float64,
                       intResampNum::Int,
                       boolDirectQuantile::Bool,
                       dblJitterSize::Float64,
                       boolStitch::Bool)

    dZETA = _empty_ts_zeta_one()
    vecEventT = _normalise_event_times(arrEventTimes)

    # trim timestamps
    dblPreUse  = -dblUseMaxDur * dblJitterSize
    dblPostUse =  dblUseMaxDur * (dblJitterSize + 1.0)
    dblStartT  = minimum(vecEventT) + dblPreUse * 2.0
    dblStopT   = maximum(vecEventT) + dblPostUse * 2.0

    indKeep = (vecTimestamps .>= dblStartT) .& (vecTimestamps .<= dblStopT)
    vecTimestamps = vecTimestamps[indKeep]
    vecData       = vecData[indKeep]

    # normalise data
    dblMin = minimum(vecData); dblMax = maximum(vecData)
    dblRange = dblMax - dblMin
    if dblRange == 0.0
        dblRange = 1.0
        @warn "calcTsZetaOne: input data has zero variance"
    end
    vecData = (vecData .- dblMin) ./ dblRange

    # stitch
    if boolStitch
        vecPseudoT, vecPseudoV, vecPseudoEventT = getPseudoTimeSeries(
            vecTimestamps, vecData, vecEventT, dblUseMaxDur)
    else
        vecPseudoT      = reshape(vecTimestamps, :, 1)
        vecPseudoV      = reshape(vecData, :, 1)
        vecPseudoEventT = vecEventT
    end

    vecPseudoV = vec(vecPseudoV) .- minimum(vec(vecPseudoV))
    vecPseudoT_v = vec(Float64.(vecPseudoT))
    vecPseudoV_v = vec(Float64.(vecPseudoV))
    vecPseudoEventT_v = vec(Float64.(vecPseudoEventT))

    if length(vecTimestamps) < 3
        @warn "calcTsZetaOne: too few entries around events"
        return dZETA
    end

    # real deviation
    dblSuperResFactor = 100.0
    vecRealDeviation, vecRealFrac, vecRealFracLinear, vecRealTime =
        getTimeseriesOffsetOne(vecPseudoT_v, vecPseudoV_v, vecPseudoEventT_v,
                               dblUseMaxDur; dblSuperResFactor=dblSuperResFactor)

    if length(vecRealDeviation) < 3
        @warn "calcTsZetaOne: too few entries after stitching"
        return dZETA
    end

    vecRealDeviation .-= mean(vecRealDeviation)
    intZETAIdx = argmax(abs.(vecRealDeviation))
    dblMaxD    = abs(vecRealDeviation[intZETAIdx])

    # jitter resamplings
    intTrials = length(vecPseudoEventT_v)
    matJitterPerTrial = dblJitterSize .* dblUseMaxDur .*
                        ((rand(intTrials, intResampNum) .- 0.5) .* 2.0)

    cellRandTime      = Vector{Vector{Float64}}(undef, intResampNum)
    cellRandDeviation = Vector{Vector{Float64}}(undef, intResampNum)
    vecMaxRandD       = fill(NaN, intResampNum)

    for intResampling in 1:intResampNum
        vecStimUseOnTime = vecPseudoEventT_v .+ matJitterPerTrial[:, intResampling]

        vecRandDeviation, _, _, vecRandT =
            getTimeseriesOffsetOne(vecPseudoT_v, vecPseudoV_v, vecStimUseOnTime, dblUseMaxDur)

        cellRandTime[intResampling]      = vecRandT
        cellRandDeviation[intResampling] = vecRandDeviation .- mean(vecRandDeviation)
        vecMaxRandD[intResampling]       = maximum(abs.(cellRandDeviation[intResampling]))
    end

    dblZetaP, dblZETA = getZetaP(dblMaxD, vecMaxRandD, boolDirectQuantile)

    return Dict{String,Any}(
        "vecRealTime"       => vecRealTime,
        "vecRealDeviation"  => vecRealDeviation,
        "vecRealFrac"       => vecRealFrac,
        "vecRealFracLinear" => vecRealFracLinear,
        "cellRandTime"      => cellRandTime,
        "cellRandDeviation" => cellRandDeviation,
        "dblZetaP"          => dblZetaP,
        "dblZETA"           => dblZETA,
        "intZETAIdx"        => intZETAIdx,
    )
end

"""
    calcTsZetaTwo(vecTimestamps1, vecData1, arrEventTimes1,
                  vecTimestamps2, vecData2, arrEventTimes2,
                  dblSuperResFactor, dblUseMaxDur, intResampNum, boolDirectQuantile)
        -> Dict{String,Any}
"""
function calcTsZetaTwo(vecTimestamps1::AbstractVector{Float64},
                       vecData1::AbstractVector{Float64},
                       arrEventTimes1::AbstractArray,
                       vecTimestamps2::AbstractVector{Float64},
                       vecData2::AbstractVector{Float64},
                       arrEventTimes2::AbstractArray,
                       dblSuperResFactor::Float64,
                       dblUseMaxDur::Float64,
                       intResampNum::Int,
                       boolDirectQuantile::Bool)

    dZETA = _empty_ts_zeta_two()

    vecEventStarts1 = _normalise_event_times(arrEventTimes1)
    vecEventStarts2 = _normalise_event_times(arrEventTimes2)

    # trim data 1
    dblPreUse  = -dblUseMaxDur
    dblPostUse =  dblUseMaxDur * 2.0
    dblStartT1 = minimum(vecEventStarts1) + dblPreUse * 2.0
    dblStopT1  = maximum(vecEventStarts1) + dblPostUse * 2.0
    indKeep1 = (vecTimestamps1 .>= dblStartT1) .& (vecTimestamps1 .<= dblStopT1)
    vecTimestamps1 = vecTimestamps1[indKeep1]
    vecData1       = vecData1[indKeep1]

    if length(vecTimestamps1) < 3
        @warn "calcTsZetaTwo: too few entries in condition 1"
        return dZETA
    end

    # trim data 2
    dblStartT2 = minimum(vecEventStarts2) + dblPreUse * 2.0
    dblStopT2  = maximum(vecEventStarts2) + dblPostUse * 2.0
    indKeep2 = (vecTimestamps2 .>= dblStartT2) .& (vecTimestamps2 .<= dblStopT2)
    vecTimestamps2 = vecTimestamps2[indKeep2]
    vecData2       = vecData2[indKeep2]

    if length(vecTimestamps2) < 3
        @warn "calcTsZetaTwo: too few entries in condition 2"
        return dZETA
    end

    # rescale jointly
    dblMin = min(minimum(vecData1), minimum(vecData2))
    dblMax = max(maximum(vecData1), maximum(vecData2))
    dblRange = dblMax - dblMin
    if dblRange == 0.0
        dblRange = 1.0
        @warn "calcTsZetaTwo: input data has zero variance"
    end
    vecTraceAct1 = (vecData1 .- dblMin) ./ dblRange
    vecTraceAct2 = (vecData2 .- dblMin) ./ dblRange

    # reference times
    vecRefT1 = getTsRefT(vecTimestamps1, vecEventStarts1, dblUseMaxDur)
    vecRefT2 = getTsRefT(vecTimestamps2, vecEventStarts2, dblUseMaxDur)

    dblSampInterval = (median(diff(vecRefT1)) + median(diff(vecRefT2))) / 2.0
    dblTol = dblSampInterval / dblSuperResFactor
    vecRefTime = uniquetol(vcat(vecRefT1, vecRefT2), dblTol)
    intT = length(vecRefTime)

    # matrices
    _, matTracePerTrial1 = getInterpolatedTimeSeries(vecTimestamps1, vecTraceAct1, vecEventStarts1, vecRefTime)
    _, matTracePerTrial2 = getInterpolatedTimeSeries(vecTimestamps2, vecTraceAct2, vecEventStarts2, vecRefTime)

    vecRealDiff, vecRealFrac1, vecRealFrac2 =
        getTimeseriesOffsetTwo(matTracePerTrial1, matTracePerTrial2)

    intZETAIdx = argmax(abs.(vecRealDiff))
    dblMaxD    = abs(vecRealDiff[intZETAIdx])

    # resamplings
    matRandDiff  = fill(NaN, intResampNum, intT)
    vecMaxRandD  = fill(NaN, intResampNum)
    matAggregate = vcat(matTracePerTrial1, matTracePerTrial2)
    intTrials1   = size(matTracePerTrial1, 1)
    intTrials2   = size(matTracePerTrial2, 1)
    intTotTrials = intTrials1 + intTrials2

    for intResampling in 1:intResampNum
        vecUseRand1 = my_randint(intTotTrials; size=intTrials1) .+ 1
        vecUseRand2 = my_randint(intTotTrials; size=intTrials2) .+ 1

        mat1_rand = matAggregate[vecUseRand1, :]
        mat2_rand = matAggregate[vecUseRand2, :]

        vecRandDiff_r, _, _ = getTimeseriesOffsetTwo(mat1_rand, mat2_rand)
        matRandDiff[intResampling, :] = vecRandDiff_r
        dblAddVal = maximum(abs.(vecRandDiff_r))
        if dblAddVal == 0.0; dblAddVal = dblMaxD; end
        vecMaxRandD[intResampling] = dblAddVal
    end

    dblZetaP, dblZETA = getZetaP(dblMaxD, vecMaxRandD, boolDirectQuantile)

    return Dict{String,Any}(
        "vecRefTime"        => vecRefTime,
        "vecRealDiff"       => vecRealDiff,
        "vecRealFrac1"      => vecRealFrac1,
        "vecRealFrac2"      => vecRealFrac2,
        "matRandDiff"       => matRandDiff,
        "dblZetaP"          => dblZetaP,
        "dblZETA"           => dblZETA,
        "intZETAIdx"        => intZETAIdx,
        "matTracePerTrial1" => matTracePerTrial1,
        "matTracePerTrial2" => matTracePerTrial2,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Placeholders
# ──────────────────────────────────────────────────────────────────────────────

function _empty_ts_zeta_one()
    return Dict{String,Any}(
        "vecRealTime"       => nothing,
        "vecRealDeviation"  => nothing,
        "vecRealFrac"       => nothing,
        "vecRealFracLinear" => nothing,
        "cellRandTime"      => nothing,
        "cellRandDeviation" => nothing,
        "dblZetaP"          => 1.0,
        "dblZETA"           => 0.0,
        "intZETAIdx"        => nothing,
    )
end

function _empty_ts_zeta_two()
    return Dict{String,Any}(
        "vecRefTime"        => nothing,
        "vecRealDiff"       => nothing,
        "vecRealFrac1"      => nothing,
        "vecRealFrac2"      => nothing,
        "matRandDiff"       => nothing,
        "dblZetaP"          => 1.0,
        "dblZETA"           => 0.0,
        "intZETAIdx"        => nothing,
        "matTracePerTrial1" => nothing,
        "matTracePerTrial2" => nothing,
    )
end
