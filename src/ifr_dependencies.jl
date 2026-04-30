# ifr_dependencies.jl - Instantaneous firing rate (IFR) computation
# Ported from zetapy/ifr_dependencies.py (Montijn, Heimel)

using Statistics
using Distributions
using Logging

"""
    calcSingleMSD(dblScale, vecT, vecV) -> Vector{Float64}

Compute the single-scale derivative of vecV at each point in vecT,
using a window of width dblScale. O(N²) — this is expected.
"""
function calcSingleMSD(dblScale::Float64,
                       vecT::AbstractVector{Float64},
                       vecV::AbstractVector{Float64})
    intN = length(vecT)
    vecMSD = zeros(Float64, intN)

    for intS in 1:intN
        dblT = vecT[intS]
        dblMinEdge = dblT - dblScale / 2.0
        dblMaxEdge = dblT + dblScale / 2.0

        intIdxMinT = findfirst_zeta(vecT .> dblMinEdge)
        if isnothing(intIdxMinT); intIdxMinT = 1; end

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

"""
    getMultiScaleDeriv(vecT, vecV; kwargs...) -> (vecRate, dMSD::Dict)

Returns multi-scale derivative of the deviation vector; i.e., the
ZETA-derived instantaneous firing rate.
"""
function getMultiScaleDeriv(vecT::AbstractVector{Float64},
                             vecV::AbstractVector{Float64};
                             dblSmoothSd::Float64=2.0,
                             dblMinScale::Union{Float64,Nothing}=nothing,
                             dblBase::Float64=1.5,
                             dblMeanRate::Float64=1.0,
                             dblUseMaxDur::Union{Float64,Nothing}=nothing,
                             boolParallel::Bool=false)

    dblRange = maximum(vecT) - minimum(vecT)
    if isnothing(dblUseMaxDur); dblUseMaxDur = dblRange; end
    if isnothing(dblMinScale)
        dblMinScale = round(log(1.0/1000.0) / log(dblBase))
    end

    # sort and flatten
    perm = sortperm(vecT)
    vecT = Float64.(vec(vecT[perm]))
    vecV = Float64.(vec(vecV[perm]))

    # remove boundary points (t==0 or t==dblUseMaxDur)
    indKeep = .!(vecT .== 0.0) .& .!(vecT .== dblUseMaxDur)
    vecT = vecT[indKeep]
    vecV = vecV[indKeep]

    # build scale vector
    dblMaxScale = log(dblRange / 10.0) / log(dblBase)
    vecExp   = collect(dblMinScale:dblMaxScale-1)
    vecScale = dblBase .^ vecExp
    intScaleNum = length(vecScale)
    intN = length(vecT)

    matMSD = zeros(Float64, intN, intScaleNum)
    for (intScaleIdx, dblScale) in enumerate(vecScale)
        matMSD[:, intScaleIdx] = calcSingleMSD(dblScale, vecT, vecV)
    end

    # Gaussian smoothing column-by-column
    if dblSmoothSd > 0.0
        intSmoothRange = 2 * Int(ceil(dblSmoothSd))
        xs = -intSmoothRange:intSmoothRange
        vecFilt = exp.(-(xs .^ 2) ./ (2.0 * dblSmoothSd^2))
        vecFilt ./= sum(vecFilt)

        intPadSize = Int(floor(length(vecFilt) / 2))
        # edge-pad each column and convolve
        matMSD_smoothed = zeros(Float64, intN, intScaleNum)
        for col in 1:intScaleNum
            coldata = matMSD[:, col]
            # edge-pad
            padded = vcat(fill(coldata[1], intPadSize), coldata, fill(coldata[end], intPadSize))
            # convolve (valid region)
            convolved = _conv1d_valid(padded, vecFilt)
            matMSD_smoothed[:, col] = convolved[1:intN]
        end
        matMSD = matMSD_smoothed
    end

    vecM = mean(matMSD, dims=2)[:, 1]   # column mean

    # weighted average by inter-spike intervals
    dblMeanM = (1.0 / dblUseMaxDur) *
               sum(((vecM[1:end-1] .+ vecM[2:end]) ./ 2.0) .* diff(vecT))

    # rescale to real firing rates
    vecRate = dblMeanRate .* ((vecM .+ 1.0/dblUseMaxDur) ./ (dblMeanM + 1.0/dblUseMaxDur))

    dMSD = Dict{String,Any}(
        "vecRate"     => vecRate,
        "vecT"        => vecT,
        "vecM"        => vecM,
        "vecScale"    => vecScale,
        "matMSD"      => matMSD,
        "vecV"        => vecV,
        "dblSmoothSd" => dblSmoothSd,
        "dblMeanRate" => dblMeanRate,
    )

    return vecRate, dMSD
end

# Manual 1-D 'valid' convolution (replaces scipy convolve2d column-wise)
function _conv1d_valid(signal::Vector{Float64}, kernel::AbstractVector)
    n = length(signal)
    k = length(kernel)
    out_len = n - k + 1
    out = zeros(Float64, out_len)
    for i in 1:out_len
        out[i] = sum(signal[i:i+k-1] .* reverse(kernel))
    end
    return out
end

"""
    getPeak(vecData, vecT; tplRestrictRange=(-Inf, Inf), intSwitchZ=1) -> Dict

Returns highest peak time, width, and location.
"""
function getPeak(vecData::AbstractVector{Float64},
                 vecT::AbstractVector{Float64};
                 tplRestrictRange::Tuple{Float64,Float64}=(-Inf, Inf),
                 intSwitchZ::Int=1)

    # z-score
    if intSwitchZ == 1
        mu = mean(vecData); sg = std(vecData)
        vecDataZ = sg == 0.0 ? zeros(length(vecData)) : (vecData .- mu) ./ sg
    elseif intSwitchZ == 2
        mask02 = vecT .< 0.02
        dblMu = any(mask02) ? mean(vecData[mask02]) : mean(vecData)
        vecDataZ = (vecData .- dblMu) ./ std(vecData)
    else
        vecDataZ = copy(vecData)
    end

    # find positive and negative peaks
    intPeakLoc, dblMaxPosVal, intPosIdx, vecLocsPos, vecPromsPos =
        _find_prominent_peak(vecDataZ, vecT, tplRestrictRange, :positive)
    intPeakLocNeg, dblMaxNegVal, intNegIdx, vecLocsNeg, vecPromsNeg =
        _find_prominent_peak(-vecDataZ, vecT, tplRestrictRange, :negative)

    indPeakMembers = nothing
    usePos = false
    useNeg = false

    if isnothing(dblMaxPosVal) && isnothing(dblMaxNegVal)
        # no peaks
    elseif isnothing(dblMaxNegVal) || (!isnothing(dblMaxPosVal) && abs(dblMaxPosVal) >= abs(dblMaxNegVal))
        usePos = true
    else
        useNeg = true
    end

    if usePos
        dblPeakProm = vecPromsPos[intPosIdx]
        dblCutOff   = vecDataZ[intPeakLoc] - dblPeakProm / 2.0
        indPeakMembers = vecDataZ .> dblCutOff
        finalPeakLoc = intPeakLoc
    elseif useNeg
        dblPeakProm = vecPromsNeg[intNegIdx]
        dblCutOff   = vecDataZ[intPeakLocNeg] + dblPeakProm / 2.0
        indPeakMembers = vecDataZ .< dblCutOff
        finalPeakLoc = intPeakLocNeg
    end

    if isnothing(indPeakMembers)
        return Dict{String,Any}(
            "dblLatencyPeak"     => NaN,
            "dblPeakValue"       => NaN,
            "dblPeakWidth"       => NaN,
            "vecPeakStartStop"   => [NaN, NaN],
            "intPeakLoc"         => nothing,
            "vecPeakStartStopIdx"=> [nothing, nothing],
        )
    end

    # find start/stop of peak region
    diffs = diff([0; Int.(indPeakMembers); 0])
    vecPeakStarts = findall(diffs .== 1)    # 1-based
    vecPeakStops  = findall(diffs .== -1) .- 1

    # closest start to left and stop to right of peak
    starters = vecPeakStarts[vecPeakStarts .<= finalPeakLoc]
    stoppers  = vecPeakStops[vecPeakStops .>= finalPeakLoc]

    if isempty(starters) || isempty(stoppers)
        intPeakStart = finalPeakLoc
        intPeakStop  = finalPeakLoc
    else
        intPeakStart = maximum(starters)
        intPeakStop  = minimum(stoppers)
    end

    intPeakStop = min(intPeakStop, length(vecT))
    dblPeakStartT = vecT[intPeakStart]
    dblPeakStopT  = vecT[intPeakStop]

    return Dict{String,Any}(
        "dblLatencyPeak"      => vecT[finalPeakLoc],
        "dblPeakValue"        => vecData[finalPeakLoc],
        "dblPeakWidth"        => dblPeakStopT - dblPeakStartT,
        "vecPeakStartStop"    => [dblPeakStartT, dblPeakStopT],
        "intPeakLoc"          => finalPeakLoc,
        "vecPeakStartStopIdx" => [intPeakStart, intPeakStop],
    )
end

# Helper: find most prominent peak, return (peakLoc, maxVal, peakIdx, locs, proms)
function _find_prominent_peak(vecDataZ, vecT, tplRestrictRange, sign::Symbol)
    locs = Int[]
    proms = Float64[]

    for i in 2:length(vecDataZ)-1
        if vecDataZ[i] > vecDataZ[i-1] && vecDataZ[i] >= vecDataZ[i+1]
            push!(locs, i)
        end
    end
    # include first/last if they qualify
    if length(vecDataZ) >= 2
        if vecDataZ[1] > vecDataZ[2]; pushfirst!(locs, 1); end
        if vecDataZ[end] > vecDataZ[end-1]; push!(locs, length(vecDataZ)); end
    end

    # compute simple prominence (value at peak minus nearest lower boundary)
    for loc in locs
        push!(proms, vecDataZ[loc])  # simplified: use value as prominence proxy
    end

    if isempty(locs)
        return nothing, nothing, nothing, Int[], Float64[]
    end

    # filter by range
    in_range = [vecT[l] >= tplRestrictRange[1] && vecT[l] <= tplRestrictRange[2] for l in locs]
    locs  = locs[in_range]
    proms = proms[in_range]

    if isempty(locs)
        return nothing, nothing, nothing, Int[], Float64[]
    end

    intIdx  = argmax(vecDataZ[locs])
    peakLoc = locs[intIdx]
    maxVal  = vecDataZ[peakLoc]

    return peakLoc, maxVal, intIdx, locs, proms
end

"""
    getOnset(vecData, vecT; dblLatencyPeak=nothing, tplRestrictRange=nothing) -> Dict

Returns peak onset time (first crossing of peak half-height).
"""
function getOnset(vecData::AbstractVector{Float64},
                  vecT::AbstractVector{Float64};
                  dblLatencyPeak::Union{Float64,Nothing}=nothing,
                  tplRestrictRange::Union{Tuple,Nothing}=nothing)

    if isnothing(tplRestrictRange)
        tplRestrictRange = (minimum(vecT), minimum(vecT) + (maximum(vecT) - minimum(vecT)))
    end

    indKeep = (vecT .>= tplRestrictRange[1]) .& (vecT .<= tplRestrictRange[2])
    vecCropT       = vecT[indKeep]
    vecDataCropped = vecData[indKeep]

    if isnothing(dblLatencyPeak)
        intPeakIdx    = argmax(vecDataCropped)
        dblLatencyPeak = vecCropT[intPeakIdx]
    else
        idx = findfirst_zeta(vecCropT .>= dblLatencyPeak)
        if isnothing(idx)
            @warn "getOnset: supplied peak was invalid; taking max"
            intPeakIdx    = argmax(vecDataCropped)
            dblLatencyPeak = vecCropT[intPeakIdx]
        else
            intPeakIdx = idx
        end
    end

    dblPeakValue = vecDataCropped[intPeakIdx]
    dblBaseValue = vecDataCropped[1]
    dblThresh    = (dblPeakValue - dblBaseValue) / 2.0 + dblBaseValue

    if dblThresh > 0.0
        intOnsetIdx = findfirst_zeta(vecDataCropped .>= dblThresh)
    else
        intOnsetIdx = findfirst_zeta(vecDataCropped .<= dblThresh)
    end

    if isnothing(intOnsetIdx)
        dblLatencyPeakOnset = NaN
        dblValue = NaN
    else
        dblLatencyPeakOnset = vecCropT[intOnsetIdx]
        dblValue = vecDataCropped[intOnsetIdx]
    end

    return Dict{String,Any}(
        "dblLatencyPeakOnset" => dblLatencyPeakOnset,
        "dblValue"            => dblValue,
        "dblBaseValue"        => dblBaseValue,
        "dblLatencyPeak"      => dblLatencyPeak,
        "dblPeakValue"        => dblPeakValue,
    )
end
