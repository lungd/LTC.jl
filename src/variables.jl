struct VariableLowerBound end
struct VariableUpperBound end
struct VariableOutput end
Symbolics.option_to_metadata_type(::Val{:lower}) = VariableLowerBound
Symbolics.option_to_metadata_type(::Val{:upper}) = VariableUpperBound
Symbolics.option_to_metadata_type(::Val{:output}) = VariableOutput

indexof(sym,syms) = findfirst(isequal(sym),syms)
