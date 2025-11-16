import sys
from sympy import symbols, expand
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)

# usage: python script.py path/to/file.txt

a = symbols('a1:10')
b = symbols('b1:10')
symtab = {f"a{i}": a[i-1] for i in range(1, 10)}
symtab.update({f"b{i}": b[i-1] for i in range(1, 10)})

transformations = standard_transformations + (implicit_multiplication_application,)

exprs = {}

with open(sys.argv[1]) as f:
    for raw in f:
        # strip comments and spaces
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        name, rhs = map(str.strip, line.split("=", 1))
        # allow references to previously defined symbols (m*, c*, etc.)
        local_dict = dict(symtab)
        local_dict.update(exprs)
        exprs[name] = parse_expr(
            rhs,
            local_dict=local_dict,
            transformations=transformations,
        )

# print all c1..cn in numeric order
for name in sorted(
    [n for n in exprs if n.startswith("c") and n[1:].isdigit()],
    key=lambda s: int(s[1:])
):
    print(f"{name} = {expand(exprs[name])}")
