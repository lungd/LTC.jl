@parameters t
D = Differential(t)

function InPin(T::TYPE=Float32; name) where TYPE #<: AbstractFloat
  @parameters x=T(13.0)
  ODESystem(Equation[],t,Num[],[x]; name)
end

function InSPin(T::TYPE=Float32; name) where TYPE #<: AbstractFloat
  @variables x(t)=T(13.0)
  ODESystem(Equation[D(x)~0],t,[x],Num[]; name)
end

# function InDSPin(T=Float32; name)
#   @variables x_in(t)=T(0) f(t)=T(13.0) x_in(t)
#   ODESystem(Equation[D(x)~f, f~x_in, D(x_in) ~ 0],t,[x],Num[]; name)
# end

function InPinTVP(T::TYPE=Float32; name) where TYPE# <: AbstractFloat
  @parameters x(t)=T(1)
  ODESystem(Equation[],t,[],Num[x]; name)
end

function OutPin(T::TYPE=Float32; name) where TYPE# <: AbstractFloat
  @variables x(t) [output=true]
  ODESystem(Equation[],t,[x],Num[]; name)
end

function create_pins(wiring::Wiring{T}) where T #<: AbstractFloat
  inpins = [InSPin(T; name=Symbol("x$(i)_InPin")) for i in 1:wiring.n_in]
  outpins = [OutPin(T; name=Symbol("x$(i)_OutPin")) for i in 1:wiring.n_out]
  inpins, outpins
end


# mutable struct SimType{T} <: DEDataVector{T}
#   x::Array{T,1}
#   observed::Array{T,1}
# end
