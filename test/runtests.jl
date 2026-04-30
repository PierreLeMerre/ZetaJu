using Test
using ZetaJu
using Random

Random.seed!(42)

# ── helper: generate a Poisson spike train with an event-locked rate increase ──
function make_driven_spikes(; n_trials=50, baseline_rate=5.0, stim_rate=30.0,
                              trial_dur=1.0, stim_dur=0.5, t_total=60.0)
    event_times = collect(range(1.0, step=trial_dur, length=n_trials))
    spikes = Float64[]

    # baseline spikes
    t = 0.0
    while t < t_total
        t += -log(rand()) / baseline_rate
        push!(spikes, t)
    end

    # add stimulus-locked spikes
    for ev in event_times
        t = ev
        while t < ev + stim_dur
            t += -log(rand()) / stim_rate
            push!(spikes, t)
        end
    end

    return sort(spikes), event_times
end

function make_noise_spikes(; n_trials=50, rate=5.0, trial_dur=1.0, t_total=60.0)
    event_times = collect(range(1.0, step=trial_dur, length=n_trials))
    spikes = Float64[]
    t = 0.0
    while t < t_total
        t += -log(rand()) / rate
        push!(spikes, t)
    end
    return sort(spikes), event_times
end

# ── Test 1: zetatest — driven neuron should be significant ──
@testset "zetatest: driven neuron" begin
    spikes, events = make_driven_spikes()
    dblZetaP, dZETA, dRate = zetatest(spikes, Float64.(events))

    @test dblZetaP < 0.05
    @test haskey(dZETA, "dblZETA")
    @test haskey(dZETA, "dblZetaP")
    @test haskey(dZETA, "vecSpikeT")
    @test haskey(dZETA, "vecRealDeviation")
    @test haskey(dZETA, "dblUseMaxDur")
    @test !isnothing(dZETA["intZETAIdx"])
    @test dZETA["dblZetaP"] == dblZetaP
    println("zetatest driven: p = $dblZetaP")
end

# ── Test 2: zetatest — noise neuron should usually not be significant ──
@testset "zetatest: noise neuron" begin
    # run a few times and check that p is not always tiny
    n_sig = 0
    for _ in 1:5
        spikes, events = make_noise_spikes()
        dblZetaP, _, _ = zetatest(spikes, Float64.(events))
        if dblZetaP < 0.05; n_sig += 1; end
    end
    # at most 2/5 should be significant under the null (generous threshold)
    @test n_sig <= 3
    println("zetatest noise: $n_sig/5 significant")
end

# ── Test 3: zetatest with boolReturnRate ──
@testset "zetatest: boolReturnRate" begin
    spikes, events = make_driven_spikes()
    dblZetaP, dZETA, dRate = zetatest(spikes, Float64.(events); boolReturnRate=true)
    @test dblZetaP < 0.05
    # dRate should have vecRate populated
    @test !isnothing(dRate["vecRate"])
    @test length(dRate["vecRate"]) > 0
end

# ── Test 4: zetatest with event stop times (mean-rate test) ──
@testset "zetatest: with stop times" begin
    spikes, events = make_driven_spikes(; stim_dur=0.4, trial_dur=1.0)
    n = length(events)
    arrET = hcat(events, events .+ 0.4)
    dblZetaP, dZETA, _ = zetatest(spikes, arrET)
    @test dblZetaP < 0.05
    @test !isnothing(dZETA["dblMeanP"])
    println("zetatest with stop times: p = $dblZetaP, meanP = $(dZETA["dblMeanP"])")
end

# ── Test 5: zetatest2 — two different conditions ──
@testset "zetatest2" begin
    spikes1, events1 = make_driven_spikes(; stim_rate=30.0)
    spikes2, events2 = make_noise_spikes(; rate=5.0)

    dblZetaP, dZETA = zetatest2(spikes1, Float64.(events1),
                                 spikes2, Float64.(events2))
    @test haskey(dZETA, "vecSpikeT")
    @test haskey(dZETA, "vecRealDiff")
    println("zetatest2: p = $dblZetaP")
end

# ── Test 6: zetatstest — time-series version ──
@testset "zetatstest" begin
    n_trials  = 40
    fs        = 30.0   # Hz
    trial_dur = 2.0
    stim_dur  = 0.5

    events = collect(range(3.0, step=trial_dur, length=n_trials))
    t_total = events[end] + trial_dur + 2.0
    vecTime = collect(0.0:1/fs:t_total)
    vecValue = 0.1 .* randn(length(vecTime))   # baseline noise

    # add stimulus response
    for ev in events
        mask = (vecTime .>= ev) .& (vecTime .< ev + stim_dur)
        vecValue[mask] .+= 1.0
    end

    dblZetaP, dZETA = zetatstest(vecTime, vecValue, Float64.(events))
    @test dblZetaP < 0.05
    @test haskey(dZETA, "vecRealTime")
    @test haskey(dZETA, "vecRealDeviation")
    println("zetatstest: p = $dblZetaP")
end

# ── Test 7: ifr ──
@testset "ifr" begin
    spikes, events = make_driven_spikes()
    vecTime_out, vecRate_out, dIFR = ifr(spikes, Float64.(events))
    @test length(vecTime_out) > 0
    @test length(vecRate_out) == length(vecTime_out)
    @test haskey(dIFR, "vecScale")
    println("ifr: returned $(length(vecRate_out)) rate points")
end

# ── Test 8: edge case — too few spikes ──
@testset "zetatest: too few spikes" begin
    spikes = [0.1, 0.2]
    events = [0.0, 1.0, 2.0, 3.0]
    dblZetaP, dZETA, dRate = zetatest(Float64.(spikes), Float64.(events))
    @test dblZetaP == 1.0
end

println("\nAll tests completed.")
