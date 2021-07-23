using SciMLBase: issymbollike, sym_to_index, getobserved

Zygote.@adjoint function Base.getindex(VA::ODESolution, sym, j::Int)
    function ODESolution_getindex_pullback(Δ)
        # i = issymbollike(sym) ? sym_to_index(sym, VA) : sym
		i = nothing
        if i === nothing
            getter = getobserved(VA)
            grz = Zygote.pullback(getter, sym, VA.u[j], VA.prob.p, VA.t[j])[2](Δ)
            du = [k == j ? grz[2] : zero(VA.u[1]) for k in 1:length(VA.u)]
            dp = grz[3] # pullback for p
            dprob = remake(VA.prob, p = dp)
            T = eltype(eltype(VA.u))
            N = length(VA.prob.p)
            Δ′ = ODESolution{T, N, typeof(du), Nothing, Nothing, Nothing, Nothing,
                             typeof(dprob), Nothing, Nothing, Nothing}(du, nothing,
                              nothing, nothing, nothing, dprob, nothing, nothing,
                              VA.dense, 0, nothing, VA.retcode)
            (Δ′, nothing, nothing)
        else
            Δ′ = [m == j ? [i == k ? Δ : zero(VA.u[1][1]) for k in 1:length(VA.u[1])] : zero(VA.u[1]) for m in 1:length(VA.u)]
            (Δ′, nothing, nothing)
        end
    end
    VA[sym, j], ODESolution_getindex_pullback
end

@adjoint function Base.getindex(VA::ODESolution, sym)
    function ODESolution_getindex_pullback(Δ)
        i = issymbollike(sym) ? sym_to_index(sym, VA) : sym
        if i === nothing
            throw("Zygote AD of purely-symbolic slicing for observed quantities is not yet supported. Work around this by using `A[sym,i]` to access each element sequentially in the function being differentiated.")
        else
            Δ′ = [ [i == k ? Δ[j] : zero(x[1]) for k in 1:length(x)] for (x, j) in zip(VA.u, 1:length(VA))]
            (Δ′, nothing)
        end
    end
    VA[sym], ODESolution_getindex_pullback
end



# Zygote.@adjoint function Base.getindex(VA::ODESolution, sym, j::Int)
#     function ODESolution_getindex_pullback(Δ)
#         i = SciMLBase.issymbollike(sym) ? SciMLBase.sym_to_index(sym, VA) : sym
#         if i === nothing
#
# 					  zerou = zero(VA.prob.u0)
# 					  _Δ = @. ifelse(Δ == nothing,(zerou,),Δ)
#
# 					  #return (DiffEqBase.build_solution(VA.prob,VA.alg,VA.t,_Δ), nothing, nothing)
# 						return (DiffEqBase.build_solution(VA.prob,VA.alg,VA.t,_Δ), nothing, nothing)
#
# 						getter = SciMLBase.getobserved(VA)
# 						# @show getter
# 						# getter = VA.prob.f.observed
#             grz = Zygote.pullback(getter, sym, VA.u[j], VA.prob.p, VA.t[j])[2](Δ)
# 						# @show grz
#             du = [k == j ? grz[2] : zero(VA.u[1]) for k in 1:length(VA.u)]
#             dp = grz[3] # pullback for p
# 						# @show size(dp)
# 						# dp = dp == nothing ? zeros(eltype(eltype(VA.u)), length(VA.prob.p)) : dp
# 						dprob = remake(VA.prob, p = dp)
#             T = eltype(eltype(VA.u))
#             N = length(VA.prob.p)
#             Δ′ = ODESolution{T, N, typeof(du), Nothing, Nothing, Nothing, Nothing,
#                              typeof(dprob), Nothing, Nothing, Nothing}(du, nothing,
#                               nothing, nothing, nothing, dprob, nothing, nothing,
#                               VA.dense, 0, nothing, VA.retcode)
# 			(Δ′, nothing, nothing)
#         else
#             Δ′ = [m == j ? [i == k ? Δ : zero(VA.u[1][1]) for k in 1:length(VA.u[1])] : zero(VA.u[1]) for m in 1:length(VA.u)]
#             (Δ′, nothing, nothing)
#         end
#     end
#     VA[sym, j], ODESolution_getindex_pullback
# end
#
# Zygote.@adjoint function Base.getindex(VA::DiffEqBase.ODESolution, sym)
#     function ODESolution_getindex_pullback(Δ)
#         i = SciMLBase.issymbollike(sym) ? SciMLBase.sym_to_index(sym, VA) : sym
#         if i === nothing
#             throw("Zygote AD of purely-symbolic slicing for observed quantities is not yet supported. Work around this by using `A[sym,i]` to access each element sequentially in the function being differentiated.")
#         else
#             Δ′ = [ [i == k ? Δ[j] : zero(x[1]) for k in 1:length(x)] for (x, j) in zip(VA.u, 1:length(VA))]
#             (Δ′, nothing)
#         end
#     end
#     VA[sym], ODESolution_getindex_pullback
# end
#

Zygote.@adjoint function Base.getindex(sim::DiffEqBase.EnsembleSolution, i::Int) #where {T,N,S}
	function EnsembleSolution_getindex_pullback(Δ::ODESolution)
		# prob = sim[1].prob
		# du = zeros(eltype(Δ.u[1]),size(Δ.u))
		# dp = zeros(eltype(prob.p),length(sim[1].prob.p))
		# eprob = remake(prob, u=du, p=dp)
		# empty_sol = ODESolution{T, N, typeof(du), Nothing, Nothing, Nothing, Nothing,
		# 										 typeof(eprob), Nothing, Nothing, Nothing}(du, nothing,
		# 											nothing, nothing, nothing, eprob, nothing, nothing,
		# 											Δ.dense, 0, nothing, Δ.retcode)
    # arr = [t == i ? Δ : empty_sol for t in 1:length(sim)]
		# arr = [t == i ? Δ : sim[t] for t in 1:length(sim)]
		arr = [t == i ? Δ : Δ for t in 1:length(sim)]
	  (EnsembleSolution(arr, 0.0, true), nothing)
	end
	sim[i], EnsembleSolution_getindex_pullback
end
