# dependencies.jl - Core spike-train ZETA internals
# Ported from zetapy/dependencies.py (Montijn, Meijer, Heimel)

using Statistics
using Distributions
using Logging

# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    findfirst_zeta(condition) -> Union{Int, Nothing}

Returns 1-based index of first `true` in `condition`, or `nothing`.
Equivalent to Python helper `findfirst(indArray)` in dependencies.py.
"""
function findfirst_zeta(condition::AbstractArray{Bool})
    idx = findfirst(condition)
    return idx   # Julia findfirst already returns nothing-or-1-based-Int
end

"""
    my_randint(high; size=nothing) -> Int or Vector{Int}

MATLAB randi-equivalent: returns integers in [0, high-1].
Matches Python my_randint(high, size=...) in dependencies.py.
"""
function my_randint(high::Int; size::Union{Int,Nothing}=nothing)
    if isnothing(size)
        return floor(Int64, rand() * high)
    else
        return floor.(Int64, rand(size) .* high)
    end
end

"""
    my_randperm(n, k) -> Vector{Int}

Returns first k elements of a random permutation of 1:n (1-based).
Equivalent to Python my_randperm(n, k) → 0-based, so add 1 at call sites when used as indices.
"""
function my_randperm(n::Int, k::Int)
    return sortperm(rand(n))[1:k]
end

"""
    flatten_array(l) -> Vector

Recursively flattens any nested iterable into a flat Vector.
"""
function flatten_array(l)
    result = Float64[]
    _flatten_recurse!(result, l)
    return result
end

function _flatten_recurse!(result, l)
    for el in l
        if el isa AbstractArray || el isa AbstractVector
            _flatten_recurse!(result, el)
        else
            push!(result, Float64(el))
        end
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ──────────────────────────────────────────────────────────────────────────────

const EULER_MASCHERONI = 0.5772156649015328606065120900824

"""
    getGumbel(dblE, dblV, arrX) -> (arrP, arrZ)

Calculate p-value and z-score for maximum value of N samples drawn from Gaussian,
using Gumbel extreme-value distribution.
"""
function getGumbel(dblE::Float64, dblV::Float64, arrX::AbstractVector{Float64})
    dblBeta = sqrt(6.0) * sqrt(dblV) / π
    dblMode = dblE - dblBeta * EULER_MASCHERONI

    fGumbelCDF(x) = exp(-exp(-(x - dblMode) / dblBeta))

    arrGumbelCDF = fGumbelCDF.(arrX)
    arrP = 1.0 .- arrGumbelCDF
    arrZ = -quantile.(Normal(), arrP ./ 2.0)

    # approximation for large X (where ppf overflows to Inf)
    for i in eachindex(arrZ)
        if isinf(arrZ[i])
            arrP[i] = exp((dblMode - arrX[i]) / dblBeta)
            arrZ[i] = -quantile(Normal(), arrP[i] / 2.0)
        end
    end

    return arrP, arrZ
end

"""
    getZetaP(arrMaxD, vecMaxRandD, boolDirectQuantile) -> (arrZetaP, arrZETA)

Compute ZETA p-value and z-score from observed max deviation and null distribution.
"""
function getZetaP(arrMaxD, vecMaxRandD::AbstractArray, boolDirectQuantile::Bool)
    vecMaxRandD_flat = sort(unique(filter(!isnan, vec(vecMaxRandD))))

    # wrap scalar in vector
    if !(arrMaxD isa AbstractArray)
        arrMaxD = [Float64(arrMaxD)]
    else
        arrMaxD = Float64.(vec(arrMaxD))
    end

    if boolDirectQuantile
        arrZetaP = fill(NaN, length(arrMaxD))
        for (i, d) in enumerate(arrMaxD)
            n = length(vecMaxRandD_flat)
            if isnan(d) || d < minimum(vecMaxRandD_flat)
                dblValue = 0.0
            elseif d > maximum(vecMaxRandD_flat) || isinf(d)
                dblValue = Float64(n)
            else
                # linear interpolation
                dblValue = _interp1(vecMaxRandD_flat, Float64.(1:n), d)
            end
            arrZetaP[i] = 1.0 - (dblValue / (1.0 + n))
        end
        arrZETA = -quantile.(Normal(), arrZetaP ./ 2.0)
    else
        dblMean = mean(vecMaxRandD_flat)
        dblVar  = var(vecMaxRandD_flat)   # ddof=1 by default in Julia
        arrZetaP, arrZETA = getGumbel(dblMean, dblVar, arrMaxD)
    end

    # unwrap scalars
    if length(arrZetaP) == 1
        return arrZetaP[1], arrZETA[1]
    end
    return arrZetaP, arrZETA
end

# Simple 1-D linear interpolation helper (like np.interp)
function _interp1(x::AbstractVector, y::AbstractVector, xq::Float64)
    if xq <= x[1];   return y[1];   end
    if xq >= x[end]; return y[end]; end
    i = searchsortedlast(x, xq)
    t = (xq - x[i]) / (x[i+1] - x[i])
    return y[i] + t * (y[i+1] - y[i])
end

# ──────────────────────────────────────────────────────────────────────────────
# Spike-time helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    getUniqueSpikes(vecSpikeTimes) -> Vector{Float64}

Add tiny jitter to identical spike times to make them unique.
"""
function getUniqueSpikes(vecSpikeTimes::Vector{Float64})
    vecSpikeTimes = sort(vecSpikeTimes)
    dblUniqueOffset = eps(Float64)
    dblShift = dblUniqueOffset
    indDuplicates = vcat(false, diff(vecSpikeTimes) .< dblUniqueOffset)
    while any(indDuplicates)
        n = sum(indDuplicates)
        vecJitter = vcat(1.0 .+ 9.0 .* rand(n), -1.0 .- 9.0 .* rand(n))
        perm = my_randperm(length(vecJitter), n)
        vecJitter = dblShift .* vecJitter[perm]
        vecSpikeTimes[indDuplicates] .+= vecJitter
        vecSpikeTimes = sort(vecSpikeTimes)
        indDuplicates = vcat(false, diff(vecSpikeTimes) .< dblUniqueOffset)
        dblShift *= 2.0
    end
    return vecSpikeTimes
end

"""
    getSpikeT(vecSpikeTimes, vecEventTimes, dblUseMaxDur) -> Vector{Float64}

Collect all spikes within [dblStartT, dblStartT+dblUseMaxDur) for each event,
expressed as trial-relative times. Returns sorted vector with 0 prepended and
dblUseMaxDur appended.
"""
function getSpikeT(vecSpikeTimes::Vector{Float64},
                   vecEventTimes::AbstractVector{Float64},
                   dblUseMaxDur::Float64)
    # pre-allocate generously
    buf = Vector{Float64}(undef, length(vecSpikeTimes) * 2)
    intIdx = 0

    for dblStartT in vecEventTimes
        dblStopT = dblStartT + dblUseMaxDur
        for s in vecSpikeTimes
            if s > dblStartT && s < dblStopT
                intIdx += 1
                if intIdx > length(buf)
                    resize!(buf, length(buf) * 2)
                end
                buf[intIdx] = s - dblStartT
            end
        end
    end

    vecSpikesInTrial = sort(buf[1:intIdx])
    return vcat([0.0], vecSpikesInTrial, [dblUseMaxDur])
end

"""
    getTempOffsetOne(vecSpikeTimes, vecEventTimes, dblUseMaxDur)
        -> (vecDeviation, vecFrac, vecFracLinear, vecSpikeT)

Computes temporal deviation of spike CDF from uniform (the ZETA deviation vector).
"""
function getTempOffsetOne(vecSpikeTimes::Vector{Float64},
                          vecEventTimes::AbstractVector{Float64},
                          dblUseMaxDur::Float64)
    vecSpikesInTrial = getSpikeT(vecSpikeTimes, vecEventTimes, dblUseMaxDur)
    vecThisSpikeTimes = getUniqueSpikes(vecSpikesInTrial)

    n = length(vecThisSpikeTimes)
    vecThisSpikeFracs  = collect(range(1.0/n, 1.0, length=n))
    vecThisFracLinear  = vecThisSpikeTimes ./ dblUseMaxDur
    vecThisDeviation   = vecThisSpikeFracs .- vecThisFracLinear
    vecThisDeviation .-= mean(vecThisDeviation)

    return vecThisDeviation, vecThisSpikeFracs, vecThisFracLinear, vecThisSpikeTimes
end

"""
    getSpikesInTrial(vecSpikes, vecTrialStarts, dblMaxDur)
        -> (cellTrialPerSpike, cellTimePerSpike)

Returns per-trial spike indices and trial-relative spike times.
"""
function getSpikesInTrial(vecSpikes::Vector{Float64},
                          vecTrialStarts::AbstractVector{Float64},
                          dblMaxDur::Float64)
    cellTrialPerSpike = Vector{Vector{Float64}}()
    cellTimePerSpike  = Vector{Vector{Float64}}()

    for (intTrial, dblStartT) in enumerate(vecTrialStarts)
        mask = (vecSpikes .>= dblStartT) .& (vecSpikes .< (dblStartT + dblMaxDur))
        vecTheseSpikes = vecSpikes[mask] .- dblStartT
        push!(cellTrialPerSpike, fill(Float64(intTrial - 1), length(vecTheseSpikes)))
        push!(cellTimePerSpike,  vecTheseSpikes)
    end

    return cellTrialPerSpike, cellTimePerSpike
end

"""
    getTempOffsetTwo(cellTimePerSpike1, cellTimePerSpike2, dblUseMaxDur;
                     boolFastInterp=false, vecSpikeT=nothing)
        -> (vecSpikeT, vecDiff, vecFrac1, vecSpikeTimes1, vecFrac2, vecSpikeTimes2)
"""
function getTempOffsetTwo(cellTimePerSpike1::Vector{Vector{Float64}},
                          cellTimePerSpike2::Vector{Vector{Float64}},
                          dblUseMaxDur::Float64;
                          boolFastInterp::Bool=false,
                          vecSpikeT::Union{Vector{Float64},Nothing}=nothing)

    vecSpikes1_raw = flatten_array(cellTimePerSpike1)
    vecSpikes2_raw = flatten_array(cellTimePerSpike2)

    vecThisSpikeTimes1 = getUniqueSpikes(sort(Float64.(vecSpikes1_raw)))
    vecThisSpikeTimes2 = getUniqueSpikes(sort(Float64.(vecSpikes2_raw)))

    if isnothing(vecSpikeT)
        vecSpikeT = sort(vcat([0.0], vecThisSpikeTimes1, vecThisSpikeTimes2, [dblUseMaxDur]))
    end

    intSp1 = length(vecThisSpikeTimes1)
    intSp2 = length(vecThisSpikeTimes2)
    intT1  = length(cellTimePerSpike1)
    intT2  = length(cellTimePerSpike2)

    # spike fraction 1 — interpolate cumulative count / nTrials onto vecSpikeT
    vecUniqueSpikeFracs1 = collect(1.0:intSp1) ./ intT1
    xp1 = vcat([0.0], vecThisSpikeTimes1, [dblUseMaxDur])
    yp1 = vcat([0.0], vecUniqueSpikeFracs1, [intSp1 / intT1])
    vecThisFrac1 = _interp1_vec(xp1, yp1, vecSpikeT)

    # spike fraction 2
    vecUniqueSpikeFracs2 = collect(1.0:intSp2) ./ intT2
    xp2 = vcat([0.0], vecThisSpikeTimes2, [dblUseMaxDur])
    yp2 = vcat([0.0], vecUniqueSpikeFracs2, [intSp2 / intT2])
    vecThisFrac2 = _interp1_vec(xp2, yp2, vecSpikeT)

    vecDeviation = vecThisFrac1 .- vecThisFrac2
    vecThisDiff  = vecDeviation .- mean(vecDeviation)

    return vecSpikeT, vecThisDiff, vecThisFrac1, vecThisSpikeTimes1, vecThisFrac2, vecThisSpikeTimes2
end

# Vectorised np.interp equivalent (clamp to boundary)
function _interp1_vec(x::AbstractVector, y::AbstractVector, xq::AbstractVector)
    return [_interp1_clamped(x, y, xi) for xi in xq]
end

function _interp1_clamped(x, y, xi)
    if xi <= x[1];   return y[1];   end
    if xi >= x[end]; return y[end]; end
    i = searchsortedlast(x, xi)
    t = (xi - x[i]) / (x[i+1] - x[i])
    return y[i] + t * (y[i+1] - y[i])
end

# ──────────────────────────────────────────────────────────────────────────────
# Data stitching
# ──────────────────────────────────────────────────────────────────────────────

"""
    getPseudoSpikeVectors(vecSpikeTimes, vecEventTimes, dblWindowDur;
                          boolDiscardEdges=false)
        -> (vecPseudoSpikeTimes, vecPseudoEventT)

Stitch trial windows into a pseudo-continuous spike train.
"""
function getPseudoSpikeVectors(vecSpikeTimes::Vector{Float64},
                               vecEventTimes::AbstractVector{Float64},
                               dblWindowDur::Float64;
                               boolDiscardEdges::Bool=false)

    vecSpikeTimes  = sort(vec(vecSpikeTimes))
    vecEventTimes  = sort(vec(Float64.(vecEventTimes)))

    intSamples = length(vecSpikeTimes)
    intTrials  = length(vecEventTimes)

    cellPseudoSpikeT = Vector{Vector{Float64}}()
    vecPseudoEventT  = fill(NaN, intTrials)
    dblPseudoEventT  = 0.0
    intLastUsedSample = 0    # 1-based; 0 means none used yet
    intFirstSample    = nothing
    dblPseudoT0       = 0.0

    for (intTrial, dblEventT) in enumerate(vecEventTimes)
        intStartSample_raw = findfirst_zeta(vecSpikeTimes .>= dblEventT)
        intEndSample_raw   = findfirst_zeta(vecSpikeTimes .> (dblEventT + dblWindowDur))

        if !isnothing(intStartSample_raw) && !isnothing(intEndSample_raw) &&
                intStartSample_raw > intEndSample_raw
            intEndSample_raw   = nothing
            intStartSample_raw = nothing
        end

        intEndSample = isnothing(intEndSample_raw) ? intSamples : intEndSample_raw

        if isnothing(intStartSample_raw) || isnothing(intEndSample)
            vecUseSamples = Int[]
        else
            intEndSample = intEndSample - 1
            vecElig = collect(intStartSample_raw:intEndSample)
            vecUseSamples = vecElig[1 .<= vecElig .<= intSamples]
        end

        if length(vecUseSamples) > 0
            if intTrial == 1 && !boolDiscardEdges
                vecUseSamples = collect(1:vecUseSamples[end])
            elseif intTrial == intTrials && !boolDiscardEdges
                vecUseSamples = collect(vecUseSamples[1]:intSamples)
            end
        end

        if length(vecUseSamples) > 0
            vecAddT = vecSpikeTimes[vecUseSamples]
            indOverlap = vecUseSamples .<= intLastUsedSample
        else
            vecAddT = Float64[]
            indOverlap = Bool[]
        end

        # compute pseudo event time
        if intTrial == 1
            dblPseudoEventT = 0.0
        else
            if dblWindowDur > (dblEventT - vecEventTimes[intTrial-1])
                # overlapping windows — remove overlap
                if length(vecUseSamples) > 0
                    vecUseSamples = vecUseSamples[.!indOverlap]
                    vecAddT = vecSpikeTimes[vecUseSamples]
                end
                dblPseudoEventT = dblPseudoEventT + dblEventT - vecEventTimes[intTrial-1]
            else
                dblPseudoEventT = dblPseudoEventT + dblWindowDur
            end
        end

        if length(vecUseSamples) == 0
            vecLocalPseudoT = Float64[]
        else
            intLastUsedSample = vecUseSamples[end]
            vecLocalPseudoT = vecAddT .- dblEventT .+ dblPseudoEventT
        end

        if isnothing(intFirstSample) && length(vecUseSamples) > 0
            intFirstSample = vecUseSamples[1]
            dblPseudoT0    = dblPseudoEventT
        end

        push!(cellPseudoSpikeT, vecLocalPseudoT)
        vecPseudoEventT[intTrial] = dblPseudoEventT
    end

    # add beginning
    if !boolDiscardEdges && !isnothing(intFirstSample) && intFirstSample > 1
        dblStepBegin = vecSpikeTimes[intFirstSample] - vecSpikeTimes[intFirstSample-1]
        vecSampAddBeginning = collect(1:intFirstSample-1)
        vsub = vecSpikeTimes[vecSampAddBeginning]
        vecAddBeginningSpikes = vsub .- vsub[1] .+ dblPseudoT0 .- dblStepBegin .-
                                (maximum(vsub) - minimum(vsub))
        push!(cellPseudoSpikeT, vecAddBeginningSpikes)
    end

    # add end
    intTn = intSamples
    dblLastEventEnd = vecEventTimes[end] + dblWindowDur
    intLastUsed_end = findfirst_zeta(vecSpikeTimes .> dblLastEventEnd)
    if !boolDiscardEdges && !isnothing(intLastUsed_end) && intTn > intLastUsed_end
        vecSampAddEnd = collect(intLastUsed_end:intTn)
        dblEventT_last = vecEventTimes[end]
        vecAddEndSpikes = vecSpikeTimes[vecSampAddEnd] .- dblEventT_last .+ dblPseudoEventT .+ dblWindowDur
        push!(cellPseudoSpikeT, vecAddEndSpikes)
    end

    all_spikes = flatten_array(cellPseudoSpikeT)
    vecPseudoSpikeTimes = sort(Float64.(all_spikes))
    return vecPseudoSpikeTimes, vecPseudoEventT
end

# ──────────────────────────────────────────────────────────────────────────────
# Core ZETA calculators
# ──────────────────────────────────────────────────────────────────────────────

"""
    calcZetaOne(vecSpikeTimes, arrEventTimes, dblUseMaxDur, intResampNum,
                boolDirectQuantile, dblJitterSize, boolStitch, boolParallel)
        -> Dict{String,Any}

One-sample spike-train ZETA calculation (inner loop).
"""
function calcZetaOne(vecSpikeTimes::Vector{Float64},
                     arrEventTimes::AbstractArray,
                     dblUseMaxDur::Float64,
                     intResampNum::Int,
                     boolDirectQuantile::Bool,
                     dblJitterSize::Float64,
                     boolStitch::Bool,
                     boolParallel::Bool)

    dZETA = _empty_zeta_one()

    # normalise arrEventTimes → vecEventT
    vecEventT = _normalise_event_times(arrEventTimes)

    # trim spikes
    dblMinPreEventT = minimum(vecEventT) - dblUseMaxDur * 5.0 * dblJitterSize
    dblStartT = max(vecSpikeTimes[1], dblMinPreEventT)
    dblStopT  = maximum(vecEventT) + dblUseMaxDur * 5.0 * dblJitterSize
    vecSpikeTimes = vecSpikeTimes[(vecSpikeTimes .>= dblStartT) .& (vecSpikeTimes .<= dblStopT)]

    if length(vecSpikeTimes) < 3
        @warn "calcZetaOne: too few spikes around events"
        return dZETA
    end

    # stitch
    if boolStitch
        vecPseudoSpikeTimes, vecPseudoEventT = getPseudoSpikeVectors(vecSpikeTimes, vecEventT, dblUseMaxDur)
    else
        vecPseudoSpikeTimes = vecSpikeTimes
        vecPseudoEventT     = vecEventT
    end

    # real deviation
    vecRealDeviation, vecRealFrac, vecRealFracLinear, vecSpikeT =
        getTempOffsetOne(vecPseudoSpikeTimes, vecPseudoEventT, dblUseMaxDur)

    if length(vecRealDeviation) < 3
        @warn "calcZetaOne: too few spikes after stitching"
        return dZETA
    end

    vecRealDeviation .-= mean(vecRealDeviation)
    intZETAIdx = argmax(abs.(vecRealDeviation))
    dblMaxD    = abs(vecRealDeviation[intZETAIdx])

    # jitter resamplings
    intTrials = length(vecPseudoEventT)
    matJitterPerTrial = dblJitterSize .* dblUseMaxDur .*
                        ((rand(intTrials, intResampNum) .- 0.5) .* 2.0)

    cellRandTime      = Vector{Vector{Float64}}(undef, intResampNum)
    cellRandDeviation = Vector{Vector{Float64}}(undef, intResampNum)
    vecMaxRandD       = fill(NaN, intResampNum)

    for intResampling in 1:intResampNum
        vecStimUseOnTime = vec(vecPseudoEventT) .+ matJitterPerTrial[:, intResampling]

        vecRandDiff, _, _, vecThisSpikeTimes =
            getTempOffsetOne(vecPseudoSpikeTimes, vecStimUseOnTime, dblUseMaxDur)

        cellRandTime[intResampling]      = vecThisSpikeTimes
        cellRandDeviation[intResampling] = vecRandDiff .- mean(vecRandDiff)
        vecMaxRandD[intResampling]       = maximum(abs.(cellRandDeviation[intResampling]))
    end

    dblZetaP, dblZETA = getZetaP(dblMaxD, vecMaxRandD, boolDirectQuantile)

    return Dict{String,Any}(
        "vecSpikeT"          => vecSpikeT,
        "vecRealDeviation"   => vecRealDeviation,
        "vecRealFrac"        => vecRealFrac,
        "vecRealFracLinear"  => vecRealFracLinear,
        "cellRandTime"       => cellRandTime,
        "cellRandDeviation"  => cellRandDeviation,
        "dblZetaP"           => dblZetaP,
        "dblZETA"            => dblZETA,
        "intZETAIdx"         => intZETAIdx,
    )
end

"""
    calcZetaTwo(vecSpikeTimes1, arrEventTimes1, vecSpikeTimes2, arrEventTimes2,
                dblUseMaxDur, intResampNum, boolDirectQuantile) -> Dict{String,Any}

Two-sample spike-train ZETA calculation.
"""
function calcZetaTwo(vecSpikeTimes1::Vector{Float64},
                     arrEventTimes1::AbstractArray,
                     vecSpikeTimes2::Vector{Float64},
                     arrEventTimes2::AbstractArray,
                     dblUseMaxDur::Float64,
                     intResampNum::Int,
                     boolDirectQuantile::Bool)

    dZETA = _empty_zeta_two()

    vecEventT1 = _normalise_event_times(arrEventTimes1)
    vecEventT2 = _normalise_event_times(arrEventTimes2)

    _, cellTimePerSpike1 = getSpikesInTrial(vecSpikeTimes1, vecEventT1, dblUseMaxDur)
    _, cellTimePerSpike2 = getSpikesInTrial(vecSpikeTimes2, vecEventT2, dblUseMaxDur)

    vecSpikeT, vecRealDiff, vecRealFrac1, _, vecRealFrac2, _ =
        getTempOffsetTwo(cellTimePerSpike1, cellTimePerSpike2, dblUseMaxDur)

    if length(vecRealDiff) < 2
        return dZETA
    end

    intZETAIdx = argmax(abs.(vecRealDiff))
    dblMaxD    = abs(vecRealDiff[intZETAIdx])

    cellAggregateTrials = vcat(cellTimePerSpike1, cellTimePerSpike2)
    intTrials1  = length(cellTimePerSpike1)
    intTrials2  = length(cellTimePerSpike2)
    intTotTrials = intTrials1 + intTrials2

    cellRandTime = Vector{Any}(undef, intResampNum)
    cellRandDiff = Vector{Any}(undef, intResampNum)
    vecMaxRandD  = fill(NaN, intResampNum)

    for intResampling in 1:intResampNum
        vecUseRand1 = my_randint(intTotTrials; size=intTrials1) .+ 1   # 0-based → 1-based
        vecUseRand2 = my_randint(intTotTrials; size=intTrials2) .+ 1

        ct1 = cellAggregateTrials[vecUseRand1]
        ct2 = cellAggregateTrials[vecUseRand2]

        n1 = sum(length.(ct1))
        n2 = sum(length.(ct2))
        if n1 == 0 && n2 == 0
            dblAddVal = dblMaxD
        else
            _, vecRandDiff_r, _, _, _, _ =
                getTempOffsetTwo(ct1, ct2, dblUseMaxDur; vecSpikeT=vecSpikeT)
            cellRandTime[intResampling] = vecSpikeT
            cellRandDiff[intResampling] = vecRandDiff_r
            dblAddVal = maximum(abs.(vecRandDiff_r))
            if dblAddVal == 0.0; dblAddVal = dblMaxD; end
        end
        vecMaxRandD[intResampling] = dblAddVal
    end

    dblZetaP, dblZETA_val = getZetaP(dblMaxD, vecMaxRandD, boolDirectQuantile)

    return Dict{String,Any}(
        "vecSpikeT"    => vecSpikeT,
        "vecRealDiff"  => vecRealDiff,
        "vecRealFrac1" => vecRealFrac1,
        "vecRealFrac2" => vecRealFrac2,
        "cellRandTime" => cellRandTime,
        "cellRandDiff" => cellRandDiff,
        "dblZetaP"     => dblZetaP,
        "dblZETA"      => dblZETA_val,
        "intZETAIdx"   => intZETAIdx,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

function _normalise_event_times(arrEventTimes::AbstractArray)
    if ndims(arrEventTimes) == 1
        return Float64.(vec(arrEventTimes))
    else
        r, c = size(arrEventTimes)
        if c < 3
            return Float64.(arrEventTimes[:, 1])
        elseif r < 3
            return Float64.(arrEventTimes[1, :])
        else
            error("arrEventTimes must be T×1 or T×2")
        end
    end
end

function _empty_zeta_one()
    return Dict{String,Any}(
        "vecSpikeT"         => nothing,
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

function _empty_zeta_two()
    return Dict{String,Any}(
        "vecSpikeT"    => nothing,
        "vecRealDiff"  => nothing,
        "vecRealFrac1" => nothing,
        "vecRealFrac2" => nothing,
        "cellRandTime" => nothing,
        "cellRandDiff" => nothing,
        "dblZetaP"     => 1.0,
        "dblZETA"      => 0.0,
        "intZETAIdx"   => nothing,
    )
end
