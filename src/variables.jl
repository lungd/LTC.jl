struct VariableLowerBound end
struct VariableUpperBound end
Symbolics.option_to_metadata_type(::Val{:lower}) = VariableLowerBound
Symbolics.option_to_metadata_type(::Val{:upper}) = VariableUpperBound

indexof(sym,syms) = findfirst(isequal(sym),syms)
