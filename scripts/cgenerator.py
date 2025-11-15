import json
import numpy as np
import typer
from pathlib import Path

app = typer.Typer(add_completion=False)

# ------------------------------
# Packing helpers (same conventions as generator.py)
# ------------------------------

def sym_pack(i, j, n):
    """Packed index for upper-triangular (i <= j) as a flat index in [0, n(n+1)/2)."""
    i2 = np.minimum(i, j)
    j2 = np.maximum(i, j)
    return (i2 * (2 * n - i2 - 1)) // 2 + j2

def skew_pack(i, j, n):
    """Packed index for strictly upper-triangular (i < j) as a flat index in [0, n(n-1)/2)."""
    i2 = np.minimum(i, j)
    j2 = np.maximum(i, j)
    return (i2 * (2 * n - i2 - 1)) // 2 + (j2 - i2 - 1)

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
# Complex parameter indexing
# ------------------------------
# For a complex matrix X in C^{p x q}, parameters are split as:
#   first pq entries -> real part (row-major),
#   next  pq entries -> imaginary part (row-major),
# Total parameter count = 2*p*q.
#
# Index maps below produce global parameter indices for real or imaginary part.

def idx_real(i, j, n_rows, n_cols):
    return i * n_cols + j

def idx_imag(i, j, n_rows, n_cols):
    return i * n_cols + j + n_rows * n_cols

# ------------------------------
# Builders: general x general (AB)
# ------------------------------

def _build_ggr(n1, n2, n3):
    """C = Re(A B). Result shape n1 x n3, gamma in general layout."""
    if n1 == 0 or n2 == 0 or n3 == 0:
        return empty_coo()

    I = np.repeat(np.arange(n1, dtype=np.int64), n2 * n3)
    J = np.tile(np.repeat(np.arange(n2, dtype=np.int64), n3), n1)
    K = np.tile(np.arange(n3, dtype=np.int64), n1 * n2)

    # Pairs contributing to real part: (Ar,Br) with +1 and (Ai,Bi) with -1
    alpha_rr = idx_real(I, J, n1, n2)
    beta_rr  = idx_real(J, K, n2, n3)
    val_rr   = np.ones_like(I, dtype=np.int8)

    alpha_ii = idx_imag(I, J, n1, n2)
    beta_ii  = idx_imag(J, K, n2, n3)
    val_ii   = -np.ones_like(I, dtype=np.int8)

    alpha = np.concatenate([alpha_rr, alpha_ii])
    beta  = np.concatenate([beta_rr,  beta_ii ])
    val   = np.concatenate([val_rr,   val_ii  ])

    gamma = np.concatenate([I * n3 + K, I * n3 + K])

    return coo_stack(alpha, beta, gamma, val)

def _build_ggi(n1, n2, n3):
    """C = Im(A B). Result shape n1 x n3, gamma in general layout."""
    if n1 == 0 or n2 == 0 or n3 == 0:
        return empty_coo()

    I = np.repeat(np.arange(n1, dtype=np.int64), n2 * n3)
    J = np.tile(np.repeat(np.arange(n2, dtype=np.int64), n3), n1)
    K = np.tile(np.arange(n3, dtype=np.int64), n1 * n2)

    # Pairs contributing to imag part: (Ar,Bi) +1 and (Ai,Br) +1
    alpha_rb = idx_real(I, J, n1, n2)
    beta_rb  = idx_imag(J, K, n2, n3)
    val_rb   = np.ones_like(I, dtype=np.int8)

    alpha_ir = idx_imag(I, J, n1, n2)
    beta_ir  = idx_real(J, K, n2, n3)
    val_ir   = np.ones_like(I, dtype=np.int8)

    alpha = np.concatenate([alpha_rb, alpha_ir])
    beta  = np.concatenate([beta_rb,  beta_ir ])
    val   = np.concatenate([val_rb,   val_ir  ])

    gamma = np.concatenate([I * n3 + K, I * n3 + K])

    return coo_stack(alpha, beta, gamma, val)

def _build_ggc(n1, n2, n3):
    """C = [Re(AB); Im(AB)] concatenated. Result gamma has size 2*n1*n3."""
    if n1 == 0 or n2 == 0 or n3 == 0:
        return empty_coo()

    coo_r = _build_ggr(n1, n2, n3)
    coo_i = _build_ggi(n1, n2, n3)

    if coo_r.size == 0 and coo_i.size == 0:
        return empty_coo()

    off = n1 * n3
    alpha = np.concatenate([coo_r[:, 0].astype(np.int64), coo_i[:, 0].astype(np.int64)])
    beta  = np.concatenate([coo_r[:, 1].astype(np.int64), coo_i[:, 1].astype(np.int64)])
    gamma = np.concatenate([coo_r[:, 2].astype(np.int64), coo_i[:, 2].astype(np.int64) + off])
    val   = np.concatenate([coo_r[:, 3], coo_i[:, 3]]).astype(np.int8)

    return coo_stack(alpha, beta, gamma, val)

# ------------------------------
# Builders: X X^T (transpose)
# ------------------------------

def _build_gtr(n1, n2, n3):
    """C = Re(X X^T) with X in C^{n1 x n2}. Result stored as packed upper Sym_n1. Requires n3==n1."""
    if n1 == 0 or n2 == 0:
        return empty_coo()
    if n3 != n1:
        raise ValueError("gtr requires n3 == n1 (packed upper result Sym_n1).")

    iu, ku = np.triu_indices(n1)
    m = iu.size
    if m == 0:
        return empty_coo()

    I = np.repeat(iu, n2)
    K = np.repeat(ku, n2)
    J = np.tile(np.arange(n2, dtype=np.int64), m)

    # Real part: Xr*Xr^T (+1) and Xi*Xi^T (-1)
    alpha_rr = idx_real(I, J, n1, n2)
    beta_rr  = idx_real(K, J, n1, n2)
    val_rr   = np.ones_like(I, dtype=np.int8)

    alpha_ii = idx_imag(I, J, n1, n2)
    beta_ii  = idx_imag(K, J, n1, n2)
    val_ii   = -np.ones_like(I, dtype=np.int8)

    alpha = np.concatenate([alpha_rr, alpha_ii])
    beta  = np.concatenate([beta_rr,  beta_ii ])
    val   = np.concatenate([val_rr,   val_ii  ])

    gamma = np.concatenate([
        sym_pack(I, K, n1),
        sym_pack(I, K, n1)
    ])

    return coo_stack(alpha, beta, gamma, val)

def _build_gti(n1, n2, n3):
    """C = Im(X X^T) with X in C^{n1 x n2}. Packed upper Sym_n1. Requires n3==n1."""
    if n1 == 0 or n2 == 0:
        return empty_coo()
    if n3 != n1:
        raise ValueError("gti requires n3 == n1 (packed upper result Sym_n1).")

    iu, ku = np.triu_indices(n1)
    m = iu.size
    if m == 0:
        return empty_coo()

    I = np.repeat(iu, n2)
    K = np.repeat(ku, n2)
    J = np.tile(np.arange(n2, dtype=np.int64), m)

    # Imag part: Xr*Xi^T (+1) and Xi*Xr^T (+1)
    alpha_ri = idx_real(I, J, n1, n2)
    beta_ri  = idx_imag(K, J, n1, n2)
    val_ri   = np.ones_like(I, dtype=np.int8)

    alpha_ir = idx_imag(I, J, n1, n2)
    beta_ir  = idx_real(K, J, n1, n2)
    val_ir   = np.ones_like(I, dtype=np.int8)

    alpha = np.concatenate([alpha_ri, alpha_ir])
    beta  = np.concatenate([beta_ri,  beta_ir ])
    val   = np.concatenate([val_ri,   val_ir  ])

    gamma = np.concatenate([
        sym_pack(I, K, n1),
        sym_pack(I, K, n1)
    ])

    return coo_stack(alpha, beta, gamma, val)

def _build_gtc(n1, n2, n3):
    """C = [Re(XX^T); Im(XX^T)] with packed upper for both blocks. Requires n3==n1."""
    if n1 == 0 or n2 == 0:
        return empty_coo()
    if n3 != n1:
        raise ValueError("gtc requires n3 == n1 (packed upper result Sym_n1).")

    coo_r = _build_gtr(n1, n2, n3)
    coo_i = _build_gti(n1, n2, n3)

    if coo_r.size == 0 and coo_i.size == 0:
        return empty_coo()

    off = n1 * (n1 + 1) // 2
    alpha = np.concatenate([coo_r[:, 0].astype(np.int64), coo_i[:, 0].astype(np.int64)])
    beta  = np.concatenate([coo_r[:, 1].astype(np.int64), coo_i[:, 1].astype(np.int64)])
    gamma = np.concatenate([coo_r[:, 2].astype(np.int64), coo_i[:, 2].astype(np.int64) + off])
    val   = np.concatenate([coo_r[:, 3], coo_i[:, 3]]).astype(np.int8)

    return coo_stack(alpha, beta, gamma, val)

# ------------------------------
# Builders: X X^H (Hermitian)
# ------------------------------

def _build_ghr(n1, n2, n3):
    """C = Re(X X^H) with X in C^{n1 x n2}. Packed upper Sym_n1. Requires n3==n1."""
    if n1 == 0 or n2 == 0:
        return empty_coo()
    if n3 != n1:
        raise ValueError("ghr requires n3 == n1 (packed upper result Sym_n1).")

    iu, ku = np.triu_indices(n1)
    m = iu.size
    if m == 0:
        return empty_coo()

    I = np.repeat(iu, n2)
    K = np.repeat(ku, n2)
    J = np.tile(np.arange(n2, dtype=np.int64), m)

    # Real part: Xr*Xr^H (+1) and Xi*Xi^H (+1)
    alpha_rr = idx_real(I, J, n1, n2)
    beta_rr  = idx_real(K, J, n1, n2)
    val_rr   = np.ones_like(I, dtype=np.int8)

    alpha_ii = idx_imag(I, J, n1, n2)
    beta_ii  = idx_imag(K, J, n1, n2)
    val_ii   = np.ones_like(I, dtype=np.int8)

    alpha = np.concatenate([alpha_rr, alpha_ii])
    beta  = np.concatenate([beta_rr,  beta_ii ])
    val   = np.concatenate([val_rr,   val_ii  ])

    gamma = np.concatenate([
        sym_pack(I, K, n1),
        sym_pack(I, K, n1)
    ])

    return coo_stack(alpha, beta, gamma, val)

def _build_ghi(n1, n2, n3):
    """C = Im(X X^H) with X in C^{n1 x n2}. Packed strictly upper Skew_n1. Requires n3==n1."""
    if n1 == 0 or n2 == 0:
        return empty_coo()
    if n3 != n1:
        raise ValueError("ghi requires n3 == n1 (packed strictly upper Skew_n1).")

    iu, ku = np.triu_indices(n1, k=1)  # strictly upper: i < k
    m = iu.size
    if m == 0:
        return empty_coo()

    I = np.repeat(iu, n2)
    K = np.repeat(ku, n2)
    J = np.tile(np.arange(n2, dtype=np.int64), m)

    # Imag part: Xi*Xr^H (+1) and Xr*Xi^H (-1)
    alpha_ir = idx_imag(I, J, n1, n2)
    beta_ir  = idx_real(K, J, n1, n2)
    val_ir   = np.ones_like(I, dtype=np.int8)

    alpha_ri = idx_real(I, J, n1, n2)
    beta_ri  = idx_imag(K, J, n1, n2)
    val_ri   = -np.ones_like(I, dtype=np.int8)

    alpha = np.concatenate([alpha_ir, alpha_ri])
    beta  = np.concatenate([beta_ir,  beta_ri ])
    val   = np.concatenate([val_ir,   val_ri  ])

    gamma = np.concatenate([
        skew_pack(I, K, n1),
        skew_pack(I, K, n1)
    ])

    return coo_stack(alpha, beta, gamma, val)

def _build_ghc(n1, n2, n3):
    """C = [Re(XX^H) in Sym_n1; Im(XX^H) in Skew_n1] concatenated. Requires n3==n1."""
    if n1 == 0 or n2 == 0:
        return empty_coo()
    if n3 != n1:
        raise ValueError("ghc requires n3 == n1 (Sym_n1 + Skew_n1).")

    coo_r = _build_ghr(n1, n2, n3)
    coo_i = _build_ghi(n1, n2, n3)

    if coo_r.size == 0 and coo_i.size == 0:
        return empty_coo()

    off = n1 * (n1 + 1) // 2  # size of Sym_n1 block
    alpha = np.concatenate([coo_r[:, 0].astype(np.int64), coo_i[:, 0].astype(np.int64)])
    beta  = np.concatenate([coo_r[:, 1].astype(np.int64), coo_i[:, 1].astype(np.int64)])
    gamma = np.concatenate([coo_r[:, 2].astype(np.int64), coo_i[:, 2].astype(np.int64) + off])
    val   = np.concatenate([coo_r[:, 3], coo_i[:, 3]]).astype(np.int8)

    return coo_stack(alpha, beta, gamma, val)

# ------------------------------
# Dispatcher
# ------------------------------

MUL_OPS = {
    "ggr": _build_ggr,
    "ggi": _build_ggi,
    "ggc": _build_ggc,
}

T_OPS = {
    "gtr": _build_gtr,
    "gti": _build_gti,
    "gtc": _build_gtc,
}

H_OPS = {
    "ghr": _build_ghr,
    "ghi": _build_ghi,
    "ghc": _build_ghc,
}

ALLOWED = set(list(MUL_OPS.keys()) + list(T_OPS.keys()) + list(H_OPS.keys()))

def generate(op, n1, n2, n3):
    """Map operation code to builder and build the tensor."""
    if op in MUL_OPS:
        return MUL_OPS[op](n1, n2, n3)
    if op in T_OPS:
        return T_OPS[op](n1, n2, n3)
    if op in H_OPS:
        return H_OPS[op](n1, n2, n3)
    raise ValueError(f"Unknown op: {op}")

# ------------------------------
# Dimension helpers for meta.json
# ------------------------------

def compute_dims(op, n1, n2, n3):
    """
    Compute (nU, nV, nW) for a given complex-aware operation and sizes.
    This matches how alpha, beta, gamma are constructed in the builders.
    """
    if op in ("ggr", "ggi", "ggc"):
        # A in C^{n1 x n2}, B in C^{n2 x n3}
        n_u = 2 * n1 * n2
        n_v = 2 * n2 * n3
        if op in ("ggr", "ggi"):
            n_w = n1 * n3
        else:  # ggc: stacked [Re; Im]
            n_w = 2 * n1 * n3
        return n_u, n_v, n_w

    if op in ("gtr", "gti", "gtc", "ghr", "ghi", "ghc"):
        if n3 != n1:
            raise ValueError(f"{op} requires n3 == n1.")
        # X in C^{n1 x n2}, parameters shared on both sides
        n_u = 2 * n1 * n2
        n_v = n_u
        sym = n1 * (n1 + 1) // 2
        skew = n1 * (n1 - 1) // 2

        if op in ("gtr", "gti"):
            n_w = sym
        elif op == "gtc":
            n_w = 2 * sym
        elif op == "ghr":
            n_w = sym
        elif op == "ghi":
            n_w = skew
        elif op == "ghc":
            n_w = sym + skew  # equals n1^2
        else:
            raise ValueError(f"Unhandled op '{op}' in compute_dims.")
        return n_u, n_v, n_w

    raise ValueError(f"Unknown op '{op}' in compute_dims.")

def save_tensor_and_meta(coo, op, n1, n2, n3, outdir):
    """
    Save tensor as .npy and write <name>.meta.json with
    name, nU, nV, nW, op, n1, n2, n3.
    """
    name = f"{op}-{n1}{n2}{n3}"
    tensor_path = outdir / f"{name}.npy"
    np.save(tensor_path, coo)

    n_u, n_v, n_w = compute_dims(op, n1, n2, n3)
    meta = {
        "name": name,
        "nU": int(n_u),
        "nV": int(n_v),
        "nW": int(n_w),
        "op": op,
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
    op = typer.Argument(..., help="Operation code: ggr, ggi, ggc, gtr, gti, gtc, ghr, ghi, ghc"),
    n1 = typer.Argument(..., help="Rows of A / X and C"),
    n2 = typer.Argument(..., help="Inner dimension / cols of A / X"),
    n3 = typer.Argument(..., help="Cols of B and C (must equal n1 for transpose/hermitian results)"),
    outdir = typer.Option("data/tensors", "--outdir", "-o", help="Output directory"),
):
    """
    Generate a real COO tensor T for complex-aware ops and save as int8 .npy (nnz,4),
    along with a <name>.meta.json file describing dimensions.

    Conventions:
      - Parameters are split into real, then imaginary parts.
      - Indices alpha/beta address 2*p*q sized parameter pools.
      - Result gamma packs as: general, Sym_n, Skew_n, or their concatenations depending on the op.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if op not in ALLOWED:
        raise typer.BadParameter(f"Unknown op '{op}'. Allowed: {', '.join(sorted(ALLOWED))}")

    try:
        n1 = int(n1)
        n2 = int(n2)
        n3 = int(n3)
    except Exception:
        raise typer.BadParameter("n1, n2, n3 must be integers")

    coo = generate(op, n1, n2, n3)
    save_tensor_and_meta(coo, op, n1, n2, n3, outdir)

if __name__ == "__main__":
    app()
