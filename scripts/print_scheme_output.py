import argparse
import sys

from sympy import symbols, expand
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)


def parse_input_file(path):
    # Define base symbols
    a = symbols('a1:65')
    b = symbols('b1:65')

    symtab = {f"a{i}": a[i - 1] for i in range(1, 65)}
    symtab.update({f"b{i}": b[i - 1] for i in range(1, 65)})

    transformations = standard_transformations + (implicit_multiplication_application,)

    exprs = {}

    with open(path) as f:
        for raw in f:
            # Strip comments and spaces
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            name, rhs = map(str.strip, line.split("=", 1))

            # Allow references to previously defined symbols (m*, c*, etc.)
            local_dict = dict(symtab)
            local_dict.update(exprs)

            exprs[name] = parse_expr(
                rhs,
                local_dict=local_dict,
                transformations=transformations,
            )

    return exprs


def apply_modulo(expr, modulus):
    # Default behavior: just expand if no modulus is given
    if modulus is None:
        return expand(expr)

    expanded = expand(expr)
    coeffs = expanded.as_coefficients_dict()

    terms = []
    for term, coeff in coeffs.items():
        coeff_mod = coeff % modulus
        if coeff_mod:
            terms.append(coeff_mod * term)

    if not terms:
        return 0

    result = terms[0]
    for t in terms[1:]:
        result += t
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Expand symbolic expressions from a file and optionally reduce coefficients modulo n."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input file with expressions.",
    )
    parser.add_argument(
        "--mod",
        type=int,
        default=None,
        help="Optional modulus for coefficients (natural number).",
    )

    args = parser.parse_args()

    if args.mod is not None and args.mod <= 0:
        parser.error("--mod must be a positive integer")

    exprs = parse_input_file(args.input_file)

    # Print all c1..cn in numeric order
    c_names = [
        n for n in exprs
        if n.startswith("c") and n[1:].isdigit()
    ]
    for name in sorted(c_names, key=lambda s: int(s[1:])):
        simplified = apply_modulo(exprs[name], args.mod)
        print(f"{name} = {simplified}")


if __name__ == "__main__":
    main()
