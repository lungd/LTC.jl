# https://juliareinforcementlearning.org/docs/experiments/experiments/Policy%20Gradient/JuliaRL_A2CGAE_CartPole/#JuliaRL\\_A2CGAE\\_CartPole

using LTC
using ModelingToolkit
using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses
using OrdinaryDiffEq
using DiffEqSensitivity


function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:A2CGAE},
    ::Val{:CartPole},
    ::Nothing;
    seed = 123,
)
    rng = StableRNG(seed)
    N_ENV = 16
    UPDATE_FREQ = 10
    env = MultiThreadEnv([
        CartPoleEnv(; T = Float32, rng = StableRNG(hash(seed + i))) for i in 1:N_ENV
    ])
    ns, na = length(state(env[1])), length(action_space(env[1]))
    RLBase.reset!(env, is_force = true)

    wiring = LTC.FWiring(ns,na)
    net = LTC.Net(wiring, name=:net)
    sys = ModelingToolkit.structural_simplify(net)

    solver = VCABM()
    sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))
    ncp_chain = Flux.Chain(LTC.Mapper(wiring.n_in),
                             LTC.RecurMTK(LTC.MTKCell(wiring.n_in, wiring.n_out, net, sys, solver, sensealg)),
                             LTC.Mapper(wiring.n_out),
                             )

    agent = Agent(
        policy = QBasedPolicy(
            learner = A2CGAELearner(
                approximator = ActorCritic(
                    actor = NeuralNetworkApproximator(
                        model = ncp_chain,
                        optimizer = Flux.Optimiser(ClipValue(1.00f0), ExpDecay(1f0, 0.1f0, 100, 0.00001f0), ADAM()),
                    ),
                    critic = NeuralNetworkApproximator(
                        model = Chain(
                            Dense(ns, 256, relu; init = glorot_uniform(rng)),
                            Dense(256, 1; init = glorot_uniform(rng)),
                        ),
                        optimizer = ADAM(1e-3),
                    ),
                ) |> cpu,
                γ = 0.99f0,
                λ = 0.97f0,
                actor_loss_weight = 1.0f0,
                critic_loss_weight = 0.5f0,
                entropy_loss_weight = 0.001f0,
                update_freq = UPDATE_FREQ,
            ),
            explorer = BatchExplorer(GumbelSoftmaxExplorer(;)),
        ),
        trajectory = CircularArraySARTTrajectory(;
            capacity = UPDATE_FREQ,
            state = Matrix{Float32} => (ns, N_ENV),
            action = Vector{Int} => (N_ENV,),
            reward = Vector{Float32} => (N_ENV,),
            terminal = Vector{Bool} => (N_ENV,),
        ),
    )
    stop_condition = StopAfterStep(50_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalBatchRewardPerEpisode(N_ENV)
    Experiment(agent, env, stop_condition, hook, "# A2CGAE with CartPole")
end


using Plots
using Statistics
ex = E`JuliaRL_A2CGAE_CartPole`
run(ex)
n = minimum(map(length, ex.hook.rewards))
m = mean([@view(x[1:n]) for x in ex.hook.rewards])
s = std([@view(x[1:n]) for x in ex.hook.rewards])
plot(m,ribbon=s)
