@parameters t
D = Differential(t)

function InPin(T::DataType=Float32; name) #where TYPE #<: AbstractFloat
  @parameters x=T(0.0)
  ODESystem(Equation[],t,Num[],[x]; name)
end

function InSPin(T::DataType=Float32; name) #where TYPE #<: AbstractFloat
  @variables x(t)=T(13.0)
  ODESystem(Equation[D(x)~0],t,[x],Num[]; name)
end

# function InDSPin(T=Float32; name)
#   @variables x_in(t)=T(0) f(t)=T(13.0) x_in(t)
#   ODESystem(Equation[D(x)~f, f~x_in, D(x_in) ~ 0],t,[x],Num[]; name)
# end

function InSPinTVP(T::DataType=Float32; name) #where TYPE# <: AbstractFloat
  @variables begin
    (x(t) = T(0)), [input=true]
  end
  @parameters f(..)=T(0)
  # u .~ map(f->f(t), fs)
  ODESystem([D(x) ~ f(t)]; name)
end

function OutPin(T::DataType=Float32; name) #where TYPE# <: AbstractFloat
  @variables x(t) [output=true]
  ODESystem(Equation[],t,[x],Num[]; name)
end

function create_pins(wiring::Wiring{<:AbstractMatrix{T},S2}; p_in=false) where {T,S2} #<: AbstractFloat
  inpins = p_in == false ? [InSPin(T; name=Symbol("x$(i)_InPin")) for i in 1:wiring.n_in] : [InPin(T; name=Symbol("x$(i)_InPin")) for i in 1:wiring.n_in]
  outpins = [OutPin(T; name=Symbol("x$(i)_OutPin")) for i in 1:wiring.n_out]
  inpins, outpins
end


# mutable struct SimType{T} <: DEDataVector{T}
#   x::Array{T,1}
#   observed::Array{T,1}
# end
