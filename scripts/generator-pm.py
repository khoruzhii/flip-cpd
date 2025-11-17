# generator-pm.py
# Tensor generator for polynomial multiplication schemes (real and complex).
#
# Operations (by degrees, not by lengths):
#   pm   = real  polynomial multiplication
#   pmr  = Re( A * B ) for complex polynomials
#   pmi  = Im( A * B ) for complex polynomials
#   pmc  = [Re( A * B ); Im( A * B )] stacked for complex polynomials
#
# Input to CLI:
#   d1 = degree of first polynomial  (so its length is n1 = d1 + 1)
#   d2 = degree of second polynomial (length n2 = d2 + 1)
#   d3 = degree of result polynomial (length n3 = d3 + 1)
#
# Convolution constraint:
#   0 <= d3 <= d1 + d2
#   (equivalently 1 <= n3 <= n1 + n2 - 1)
#
# Parameter counts:
#   pm:
#     nU = n1
#     nV = n2
#     nW = n3
#
#   pmr, pmi:
#     first polynomial A: complex, parameters [Re(A), Im(A)] -> length 2 * n1
#     second polynomial B: complex, parameters [Re(B), Im(B)] -> length 2 * n2
#     result is real:
#       nU = 2 * n1
#       nV = 2 * n2
#       nW = n3
#
#   pmc:
#     result is stacked [Re(C); Im(C)], length 2 * n3:
#       nU = 2 * n1
#       nV = 2 * n2
#       nW = 2 * n3
#
# Tensor format:
#   COO int8 tensor of shape (nnz, 4) with columns:
#     alpha = index of parameter of first polynomial
#     beta  = index of parameter of second polynomial
#     gamma = index of result coefficient
#     val   = +1 or -1
#
# Parameter indexing (complex polynomials):
#   For a polynomial of length n:
#     parameters = [Re[0], ..., Re[n-1], Im[0], ..., Im[n-1]]
#   For coefficient index i:
#     idx_real(i, n) = i
#     idx_imag(i, n) = i + n
#
# Naming:
#   Files use degrees (not lengths):
#     name = f"{op}-{d1}{d2}{d3}"
#   Files produced:
#     {op}-{d1}{d2}{d3}.npy
#     {op}-{d1}{d2}{d3}.meta.json
#
# Meta example:
#   {
#     "name": "pmc-224",
#     "nU": ...,
#     "nV": ...,
#     "nW": ...,
#     "op": "pmc",
#     "deg1": d1,
#     "deg2": d2,
#     "deg3": d3,
#     "n1": n1,
#     "n2": n2,
#     "n3": n3
#   }

import json
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False)


# ------------------------------
# COO helpers
# ------------------------------

def ensure_int8_bounds(arr, name):
    """Ensure array fits into unsigned 7-bit range [0, 63] (as required by downstream)."""
    if arr.size == 0:
        return
    mx = int(arr.max())
    mn = int(arr.min())
    if mn < 0 or mx > 63:
        raise ValueError(f"{name} contains indices outside [0,63]: min={mn}, max={mx}")


def coo_stack(alpha, beta, gamma, val):
    """Stack into (nnz,4) int8 array with columns [alpha, beta, gamma, val]."""
    ensure_int8_bounds(alpha, "alpha")
    ensure_int8_bounds(beta, "beta")
    ensure_int8_bounds(gamma, "gamma")
    out = np.empty((alpha.size, 4), dtype=np.int8)
    out[:, 0] = alpha.astype(np.int8, copy=False)
    out[:, 1] = beta.astype(np.int8, copy=False)
    out[:, 2] = gamma.astype(np.int8, copy=False)
    out[:, 3] = val.astype(np.int8, copy=False)
    return out


def empty_coo():
    """Return an empty (0,4) int8 COO tensor."""
    return coo_stack(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int8),
    )


# ------------------------------
# Complex parameter indexing for polynomials
# ------------------------------

def idx_real_poly(i, n):
    """Index of real part of coefficient i for a polynomial of length n."""
    return i


def idx_imag_poly(i, n):
    """Index of imaginary part of coefficient i for a polynomial of length n."""
    return i + n


# ------------------------------
# Builders: real polynomial multiplication
# ------------------------------

def _build_pm_real(n1, n2, n3):
    """
    Real polynomial multiplication (with truncation on high degrees).

    A(x) = sum_{i=0}^{n1-1} a_i x^i
    B(x) = sum_{j=0}^{n2-1} b_j x^j
    C(x) = A(x) * B(x)
    c_k = sum_{i+j = k} a_i b_j

    We store only k = 0..n3-1.
    For every (i, j) with 0 <= i < n1, 0 <= j < n2 and k = i + j < n3
    we add row [i, j, k, +1].
    """
    if n1 <= 0 or n2 <= 0 or n3 <= 0:
        return empty_coo()

    # Grid of indices
    i = np.arange(n1, dtype=np.int64).reshape(-1, 1)  # (n1, 1)
    j = np.arange(n2, dtype=np.int64).reshape(1, -1)  # (1, n2)
    k = i + j                                         # (n1, n2)

    mask = k < n3
    if not mask.any():
        return empty_coo()

    idx_i, idx_j = np.nonzero(mask)
    alpha = idx_i
    beta = idx_j
    gamma = idx_i + idx_j
    val = np.ones_like(alpha, dtype=np.int8)

    return coo_stack(alpha, beta, gamma, val)


# ------------------------------
# Builders: complex polynomial multiplication
# ------------------------------

def _build_pmr(n1, n2, n3):
    """
    pmr: C(x) = Re(A(x) * B(x)) for complex polynomials A, B.

    A_i = a_i^r + i a_i^i
    B_j = b_j^r + i b_j^i
    c_k = sum_{i+j=k} A_i B_j

    Re(c_k) = sum_{i+j=k} (a_i^r b_j^r - a_i^i b_j^i)

    Result is a real polynomial of length n3 (degrees 0..n3-1).
    Parameters:
      first polynomial:  2 * n1 (real then imag)
      second polynomial: 2 * n2 (real then imag)
    """
    if n1 <= 0 or n2 <= 0 or n3 <= 0:
        return empty_coo()

    i = np.arange(n1, dtype=np.int64).reshape(-1, 1)
    j = np.arange(n2, dtype=np.int64).reshape(1, -1)
    k = i + j

    mask = k < n3
    if not mask.any():
        return empty_coo()

    idx_i, idx_j = np.nonzero(mask)
    k_flat = idx_i + idx_j

    # Real-real contributions: +1
    alpha_rr = idx_real_poly(idx_i, n1)
    beta_rr = idx_real_poly(idx_j, n2)
    val_rr = np.ones_like(idx_i, dtype=np.int8)

    # Imag-imag contributions: -1
    alpha_ii = idx_imag_poly(idx_i, n1)
    beta_ii = idx_imag_poly(idx_j, n2)
    val_ii = -np.ones_like(idx_i, dtype=np.int8)

    alpha = np.concatenate([alpha_rr, alpha_ii])
    beta = np.concatenate([beta_rr, beta_ii])
    gamma = np.concatenate([k_flat, k_flat])
    val = np.concatenate([val_rr, val_ii])

    return coo_stack(alpha, beta, gamma, val)


def _build_pmi(n1, n2, n3):
    """
    pmi: C(x) = Im(A(x) * B(x)) for complex polynomials A, B.

    A_i = a_i^r + i a_i^i
    B_j = b_j^r + i b_j^i
    c_k = sum_{i+j=k} A_i B_j

    Im(c_k) = sum_{i+j=k} (a_i^r b_j^i + a_i^i b_j^r)

    Result is an imaginary polynomial block of length n3.
    Parameters:
      first polynomial:  2 * n1 (real then imag)
      second polynomial: 2 * n2 (real then imag)
    """
    if n1 <= 0 or n2 <= 0 or n3 <= 0:
        return empty_coo()

    i = np.arange(n1, dtype=np.int64).reshape(-1, 1)
    j = np.arange(n2, dtype=np.int64).reshape(1, -1)
    k = i + j

    mask = k < n3
    if not mask.any():
        return empty_coo()

    idx_i, idx_j = np.nonzero(mask)
    k_flat = idx_i + idx_j

    # Real-imag contributions: +1
    alpha_ri = idx_real_poly(idx_i, n1)
    beta_ri = idx_imag_poly(idx_j, n2)
    val_ri = np.ones_like(idx_i, dtype=np.int8)

    # Imag-real contributions: +1
    alpha_ir = idx_imag_poly(idx_i, n1)
    beta_ir = idx_real_poly(idx_j, n2)
    val_ir = np.ones_like(idx_i, dtype=np.int8)

    alpha = np.concatenate([alpha_ri, alpha_ir])
    beta = np.concatenate([beta_ri, beta_ir])
    gamma = np.concatenate([k_flat, k_flat])
    val = np.concatenate([val_ri, val_ir])

    return coo_stack(alpha, beta, gamma, val)


def _build_pmc(n1, n2, n3):
    """
    pmc: stacked complex result [Re(C); Im(C)].

    First block: Re(A B) of length n3.
    Second block: Im(A B) of length n3.
    gamma layout: [0..n3-1] for Re, [n3..2*n3-1] for Im.
    """
    if n1 <= 0 or n2 <= 0 or n3 <= 0:
        return empty_coo()

    coo_r = _build_pmr(n1, n2, n3)
    coo_i = _build_pmi(n1, n2, n3)

    if coo_r.size == 0 and coo_i.size == 0:
        return empty_coo()

    off = n3
    alpha = []
    beta = []
    gamma = []
    val = []

    if coo_r.size > 0:
        alpha.append(coo_r[:, 0].astype(np.int64))
        beta.append(coo_r[:, 1].astype(np.int64))
        gamma.append(coo_r[:, 2].astype(np.int64))
        val.append(coo_r[:, 3].astype(np.int8))

    if coo_i.size > 0:
        alpha.append(coo_i[:, 0].astype(np.int64))
        beta.append(coo_i[:, 1].astype(np.int64))
        gamma.append(coo_i[:, 2].astype(np.int64) + off)
        val.append(coo_i[:, 3].astype(np.int8))

    alpha = np.concatenate(alpha)
    beta = np.concatenate(beta)
    gamma = np.concatenate(gamma)
    val = np.concatenate(val)

    return coo_stack(alpha, beta, gamma, val)


# ------------------------------
# Dispatcher
# ------------------------------

OPS = {
    "pm": _build_pm_real,
    "pmr": _build_pmr,
    "pmi": _build_pmi,
    "pmc": _build_pmc,
}

ALLOWED = set(OPS.keys())


def generate(op, d1, d2, d3):
    """
    Map operation code and degrees to builder and build the tensor.

    Input degrees:
      d1, d2, d3 >= 0
    Converted lengths:
      n1 = d1 + 1
      n2 = d2 + 1
      n3 = d3 + 1

    Convolution constraint:
      d3 <= d1 + d2
    """
    if op not in OPS:
        raise ValueError(f"Unknown op '{op}'. Allowed: {', '.join(sorted(ALLOWED))}")

    if d1 < 0 or d2 < 0 or d3 < 0:
        raise ValueError("Degrees d1, d2, d3 must be non-negative.")

    n1 = d1 + 1
    n2 = d2 + 1
    n3 = d3 + 1

    if d3 > d1 + d2:
        raise ValueError(
            f"Invalid degrees: d3={d3} is too large for convolution of degrees d1={d1}, d2={d2}. "
            f"Must satisfy d3 <= d1 + d2."
        )

    builder = OPS[op]
    coo = builder(n1, n2, n3)
    return coo, n1, n2, n3


# ------------------------------
# Dimension helpers for meta.json
# ------------------------------

def compute_dims(op, d1, d2, d3):
    """
    Compute (nU, nV, nW) for a given operation and degrees.

    Degrees:
      d1, d2, d3
    Lengths:
      n1 = d1 + 1
      n2 = d2 + 1
      n3 = d3 + 1
    """
    n1 = d1 + 1
    n2 = d2 + 1
    n3 = d3 + 1

    if op == "pm":
        n_u = n1
        n_v = n2
        n_w = n3
        return n_u, n_v, n_w

    if op in ("pmr", "pmi"):
        n_u = 2 * n1
        n_v = 2 * n2
        n_w = n3
        return n_u, n_v, n_w

    if op == "pmc":
        n_u = 2 * n1
        n_v = 2 * n2
        n_w = 2 * n3
        return n_u, n_v, n_w

    raise ValueError(f"Unknown op '{op}' in compute_dims.")


def save_tensor_and_meta(coo, op, d1, d2, d3, n1, n2, n3, outdir):
    """
    Save tensor as .npy and write <name>.meta.json with
    name, nU, nV, nW, op, degrees and lengths.
    """
    name = f"{op}-{d1}{d2}{d3}"
    tensor_path = outdir / f"{name}.npy"
    np.save(tensor_path, coo)

    n_u, n_v, n_w = compute_dims(op, d1, d2, d3)
    meta = {
        "name": name,
        "nU": int(n_u),
        "nV": int(n_v),
        "nW": int(n_w),
        "op": None,
        "deg1": int(d1),
        "deg2": int(d2),
        "deg3": int(d3),
        "n1": int(n1),
        "n2": int(n2),
        "n3": int(n3),
    }
    meta_path = outdir / f"{name}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    typer.echo(
        f"Saved: {tensor_path} and {meta_path}  "
        f"shape={coo.shape}  dtype={coo.dtype}"
    )


# ------------------------------
# CLI
# ------------------------------

@app.command()
def main(
    op=typer.Argument(..., help="Operation code: pm, pmr, pmi, pmc"),
    d1=typer.Argument(..., help="Degree of first polynomial (>= 0)"),
    d2=typer.Argument(..., help="Degree of second polynomial (>= 0)"),
    d3=typer.Argument(..., help="Degree of result polynomial (0..d1+d2)"),
    outdir=typer.Option("data/tensors", "--outdir", "-o", help="Output directory"),
):
    """
    Generate a real COO tensor T for polynomial multiplication (real or complex)
    and save as int8 .npy (nnz,4), along with a <name>.meta.json file.

    Inputs are degrees, not lengths:
      n1 = d1 + 1
      n2 = d2 + 1
      n3 = d3 + 1

    Names use degrees:
      {op}-{d1}{d2}{d3}.npy / .meta.json
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if op not in ALLOWED:
        raise typer.BadParameter(f"Unknown op '{op}'. Allowed: {', '.join(sorted(ALLOWED))}")

    try:
        d1 = int(d1)
        d2 = int(d2)
        d3 = int(d3)
    except Exception:
        raise typer.BadParameter("d1, d2, d3 must be integers")

    if d1 < 0 or d2 < 0 or d3 < 0:
        raise typer.BadParameter("Degrees d1, d2, d3 must be non-negative")

    if d3 > d1 + d2:
        raise typer.BadParameter(
            f"Invalid degrees: d3={d3} is too large for convolution of degrees d1={d1}, d2={d2}. "
            f"Must satisfy d3 <= d1 + d2."
        )

    coo, n1, n2, n3 = generate(op, d1, d2, d3)
    save_tensor_and_meta(coo, op, d1, d2, d3, n1, n2, n3, outdir)


if __name__ == "__main__":
    app()
