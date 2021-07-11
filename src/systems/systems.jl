@parameters t
D = Differential(t)

function InPin(;name)
  vars = @variables x(t)
  ps = @parameters val=13.37f0
  eqs = [x ~ val]
  ODESystem(eqs,t,vars,ps; name)
end

function OutPin(;name)
  #@variables x(t)=133f0 #out(t)=0f0
  vars = @variables x(t)=0f0 #out(t)=0f0 #dout(t)=0f0
  eqs = Equation[]
  ODESystem(eqs,t,vars,Num[]; name)
  # ODESystem(Equation[],t,[x],Num[]; name)
end

function create_pins(in::Integer, out::Integer)
  inpins = [InPin(;name=Symbol("x$(i)_InPin")) for i in 1:in]
  outpins = [OutPin(;name=Symbol("x$(i)_OutPin")) for i in 1:out]
  inpins, outpins
end
