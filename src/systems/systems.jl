@parameters t
D = Differential(t)

function InPin(T=Float32; name)
  @parameters x=T(13.0)
  ODESystem(Equation[],t,Num[],[x]; name)
end

function InSPin(T=Float32; name)
  @variables x(t)=T(13.0)
  ODESystem(Equation[D(x)~0],t,[x],Num[]; name)
end

function OutPin(;name)
  @variables x(t) [output=true]
  ODESystem(Equation[],t,[x],Num[]; name)
end

function create_pins(in::Integer, out::Integer)
  inpins = [InSPin(;name=Symbol("x$(i)_InPin")) for i in 1:in]
  outpins = [OutPin(;name=Symbol("x$(i)_OutPin")) for i in 1:out]
  inpins, outpins
end


# mutable struct SimType{T} <: DEDataVector{T}
#   x::Array{T,1}
#   observed::Array{T,1}
# end
