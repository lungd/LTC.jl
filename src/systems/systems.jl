@parameters t
D = Differential(t)

function InPin(;name)
  @parameters x=13f0
  ODESystem(Equation[],t,Num[],[x]; name)
end

function OutPin(;name)
  @variables x(t)
  ODESystem(Equation[],t,[x],Num[]; name)
end

function create_pins(in::Integer, out::Integer)
  inpins = [InPin(;name=Symbol("x$(i)_InPin")) for i in 1:in]
  outpins = [OutPin(;name=Symbol("x$(i)_OutPin")) for i in 1:out]
  inpins, outpins
end


# mutable struct SimType{T} <: DEDataVector{T}
#   x::Array{T,1}
#   observed::Array{T,1}
# end
