# generator.py
# Unified tensor generator for structured matrix multiplication schemes.
# Structures:
#   g = general
#   u = upper-triangular
#   l = lower-triangular
#   s = symmetric
#   k = skew-symmetric
#   w = "skew+diag": off-diagonal like skew-symmetric, diagonal is independent like symmetric
# Suffix 't' means "X @ X.T" with symmetric result stored in packed upper (Sym_n).
#
# Supported ops:
#   gg, ug, sg, kg, wg,
#   uu, us, uk, uw, ul,
#   ss, sk, sw,
#   kk, ww,
#   gt, ut, st, kt, wt
#
# Recursive suffixes (square matrices only):
#   "" (empty) = standard tensor
#   "a" = exclude C[0,0]
#   "b" = exclude C[n-1,n-1]
#   "c" = exclude both C[0,0] and C[n-1,n-1]
#
# Notes:
# - All generators return an int8 COO tensor of shape (nnz, 4): [alpha, beta, gamma, val].
# - alpha indexes parameters of the left operand; beta indexes parameters of the right operand.
# - gamma indexes positions of the output:
#     * general result: gamma = i * n3 + k  (i in [0..n1-1], k in [0..n3-1])
#     * packed upper:   gamma = sym_pack(i, k, n1) with i <= k
# - For 'k' (skew), diagonal terms are excluded; signs are +1 if first index < second, -1 otherwise.
# - For 'w' (skew+diag), diagonal is kept with sign +1; off-diagonal signs as for skew.

import json
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False)

# ------------------------------
# Packing helpers
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

def _filter_suffix(coo, suffix, gamma_mode, n):
    """Filter COO tensor for recursive suffixes a, b, c."""
    if suffix == "" or coo.size == 0:
        return coo

    gamma_col = coo[:, 2]

    gamma_first = 0
    if gamma_mode == "general":
        gamma_last = n * n - 1
    elif gamma_mode == "packed_upper":
        gamma_last = n * (n + 1) // 2 - 1
    else:
        raise ValueError(f"Unknown gamma_mode: {gamma_mode}")

    mask = np.ones(coo.shape[0], dtype=bool)
    if suffix in ("a", "c"):
        mask &= gamma_col != gamma_first
    if suffix in ("b", "c"):
        mask &= gamma_col != gamma_last

    return coo[mask]

# ------------------------------
# Structure policies (vectorized)
# ------------------------------

def _left_policy(struct, I, J, n1, n2):
    """
    Left operand A of shape (n1 x n2) evaluated at (i,j).
    Returns (alpha, sign, mask).
    """
    if I.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int8), np.array([], dtype=bool)

    if struct == "g":
        alpha = I * n2 + J
        sign = np.ones_like(I, dtype=np.int8)
        mask = np.ones_like(I, dtype=bool)
        return alpha, sign, mask

    if n2 != n1:
        raise ValueError(f"Left structure '{struct}' requires square left operand: n2 == n1.")

    if struct == "u":
        alpha = sym_pack(I, J, n1)
        sign = np.ones_like(I, dtype=np.int8)
        mask = J >= I
        return alpha, sign, mask

    if struct == "l":
        alpha = sym_pack(I, J, n1)
        sign = np.ones_like(I, dtype=np.int8)
        mask = J <= I
        return alpha, sign, mask

    if struct == "s":
        alpha = sym_pack(I, J, n1)
        sign = np.ones_like(I, dtype=np.int8)
        mask = np.ones_like(I, dtype=bool)
        return alpha, sign, mask

    if struct == "k":
        p = np.minimum(I, J)
        q = np.maximum(I, J)
        alpha = skew_pack(p, q, n1)
        sign = np.where(I < J, 1, -1).astype(np.int8)
        mask = I != J
        return alpha, sign, mask

    if struct == "w":
        p = np.minimum(I, J)
        q = np.maximum(I, J)
        alpha = sym_pack(p, q, n1)
        sign = np.where(I <= J, 1, -1).astype(np.int8)
        mask = np.ones_like(I, dtype=bool)
        return alpha, sign, mask

    raise ValueError(f"Unknown left structure '{struct}'.")

def _right_policy(struct, J, K, n2, n3):
    """
    Right operand B of shape (n2 x n3) evaluated at (j,k).
    Returns (beta, sign, mask).
    """
    if J.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int8), np.array([], dtype=bool)

    if struct == "g":
        beta = J * n3 + K
        sign = np.ones_like(J, dtype=np.int8)
        mask = np.ones_like(J, dtype=bool)
        return beta, sign, mask

    if n2 != n3:
        raise ValueError(f"Right structure '{struct}' requires square right operand: n2 == n3.")

    n = n2

    if struct == "u":
        beta = sym_pack(J, K, n)
        sign = np.ones_like(J, dtype=np.int8)
        mask = J <= K
        return beta, sign, mask

    if struct == "l":
        beta = sym_pack(J, K, n)
        sign = np.ones_like(J, dtype=np.int8)
        mask = J >= K
        return beta, sign, mask

    if struct == "s":
        beta = sym_pack(J, K, n)
        sign = np.ones_like(J, dtype=np.int8)
        mask = np.ones_like(J, dtype=bool)
        return beta, sign, mask

    if struct == "k":
        p = np.minimum(J, K)
        q = np.maximum(J, K)
        beta = skew_pack(p, q, n)
        sign = np.where(J < K, 1, -1).astype(np.int8)
        mask = J != K
        return beta, sign, mask

    if struct == "w":
        p = np.minimum(J, K)
        q = np.maximum(J, K)
        beta = sym_pack(p, q, n)
        sign = np.where(J <= K, 1, -1).astype(np.int8)
        mask = np.ones_like(J, dtype=bool)
        return beta, sign, mask

    raise ValueError(f"Unknown right structure '{struct}'.")

# ------------------------------
# Templates
# ------------------------------

def _build_general(a_struct, b_struct, n1, n2, n3, gamma_mode="general", suffix=""):
    """
    Generic builder for C = A B with specified left/right structures.
    gamma_mode: "general" or "packed_upper".
    suffix: "", "a", "b", or "c" for recursive filtering.
    """
    if n1 == 0 or n2 == 0 or n3 == 0:
        return empty_coo()

    I = np.repeat(np.arange(n1, dtype=np.int64), n2 * n3)
    J = np.tile(np.repeat(np.arange(n2, dtype=np.int64), n3), n1)
    K = np.tile(np.arange(n3, dtype=np.int64), n1 * n2)

    alpha, s_left, m_left = _left_policy(a_struct, I, J, n1, n2)
    beta, s_right, m_right = _right_policy(b_struct, J, K, n2, n3)

    mask = m_left & m_right

    if gamma_mode == "packed_upper":
        if n3 != n1:
            raise ValueError("packed_upper gamma requires n3 == n1.")
        mask = mask & (I <= K)
        gamma = sym_pack(I, K, n1)
    elif gamma_mode == "general":
        gamma = I * n3 + K
    else:
        raise ValueError("Unknown gamma_mode")

    if mask.size == 0 or not mask.any():
        return empty_coo()

    I = I[mask]
    J = J[mask]
    K = K[mask]
    alpha = alpha[mask]
    beta = beta[mask]
    gamma = gamma[mask]
    val = (s_left[mask] * s_right[mask]).astype(np.int8)

    coo = coo_stack(alpha, beta, gamma, val)

    if suffix:
        coo = _filter_suffix(coo, suffix, gamma_mode, n1)

    return coo

def _build_t(struct, n1, n2, n3, suffix=""):
    """
    Builder for Up(X X^T) where X has the given structure.
    Result gamma is packed upper (Sym_n1). Requires n3 == n1.
    suffix: "", "a", "b", or "c" for recursive filtering.
    """
    if n1 == 0 or n2 == 0:
        return empty_coo()
    if n3 != n1:
        raise ValueError("t-operations require n3 == n1 (packed upper result Sym_n1).")

    iu, ku = np.triu_indices(n1)
    m = iu.size
    if m == 0:
        return empty_coo()

    I = np.repeat(iu, n2)
    K = np.repeat(ku, n2)
    J = np.tile(np.arange(n2, dtype=np.int64), m)

    alpha, s_left, m_left = _left_policy(struct, I, J, n1, n2)
    beta, s_right, m_right = _left_policy(struct, K, J, n1, n2)

    mask = m_left & m_right
    if mask.size == 0 or not mask.any():
        return empty_coo()

    alpha = alpha[mask]
    beta = beta[mask]
    gamma = sym_pack(I[mask], K[mask], n1)
    val = (s_left[mask] * s_right[mask]).astype(np.int8)

    coo = coo_stack(alpha, beta, gamma, val)

    if suffix:
        coo = _filter_suffix(coo, suffix, "packed_upper", n1)

    return coo

# ------------------------------
# Ops tables (exposed for batch mode)
# ------------------------------

MUL_OPS = {
    "gg": ("g", "g", "general"),
    "ug": ("u", "g", "general"),
    "sg": ("s", "g", "general"),
    "kg": ("k", "g", "general"),
    "wg": ("w", "g", "general"),

    "uu": ("u", "u", "packed_upper"),
    "us": ("u", "s", "general"),
    "uk": ("u", "k", "general"),
    "uw": ("u", "w", "general"),
    "ul": ("u", "l", "general"),

    "ss": ("s", "s", "general"),
    "sk": ("s", "k", "general"),
    "sw": ("s", "w", "general"),

    "kk": ("k", "k", "general"),
    "ww": ("w", "w", "general"),
}

T_OPS = {
    "gt": "g",
    "ut": "u",
    "st": "s",
    "kt": "k",
    "wt": "w",
}

ALLOWED = set(list(MUL_OPS.keys()) + list(T_OPS.keys()))

# ------------------------------
# Dimension helpers for meta.json
# ------------------------------

def _struct_dim(struct, n_rows, n_cols):
    """Number of free parameters for a structured matrix of shape (n_rows x n_cols)."""
    if struct == "g":
        return n_rows * n_cols

    if n_rows != n_cols:
        raise ValueError(f"Structure '{struct}' requires square matrix: n_rows == n_cols.")

    n = n_rows

    if struct in ("u", "l", "s", "w"):
        return n * (n + 1) // 2

    if struct == "k":
        return n * (n - 1) // 2

    raise ValueError(f"Unknown structure '{struct}' in _struct_dim.")

def compute_dims(op, n1, n2, n3):
    """
    Compute (nU, nV, nW) for a given operation and sizes.
    This matches how alpha, beta, gamma are constructed in the generators.
    """
    if op in MUL_OPS:
        a_struct, b_struct, mode = MUL_OPS[op]
        n_u = _struct_dim(a_struct, n1, n2)
        n_v = _struct_dim(b_struct, n2, n3)

        if mode == "general":
            n_w = n1 * n3
        elif mode == "packed_upper":
            if n1 != n3:
                raise ValueError("packed_upper result requires n1 == n3.")
            n_w = n1 * (n1 + 1) // 2
        else:
            raise ValueError(f"Unknown mode '{mode}' in compute_dims.")

        return n_u, n_v, n_w

    if op in T_OPS:
        struct = T_OPS[op]
        if n3 != n1:
            raise ValueError("t-operations require n3 == n1.")
        n_u = _struct_dim(struct, n1, n2)
        n_v = n_u
        n_w = n1 * (n1 + 1) // 2
        return n_u, n_v, n_w

    raise ValueError(f"Unknown op '{op}' in compute_dims.")

def save_tensor_and_meta(coo, op, n1, n2, n3, suffix, outdir):
    """
    Save tensor as .npy and write <name>.meta.json with
    name, nU, nV, nW, op, n1, n2, n3.
    Suffix is encoded only in the name, not as a separate meta field.
    """
    if suffix is None:
        suffix = ""

    name = f"{op}-{n1}{n2}{n3}{suffix}"
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
# Dispatcher
# ------------------------------

def generate(op, n1, n2, n3, suffix=""):
    """
    Map operation code to template parameters and build the tensor.
    suffix: "", "a", "b", or "c" for recursive filtering (requires n1==n2==n3).
    """
    if suffix and suffix not in ("a", "b", "c"):
        raise ValueError(f"Invalid suffix '{suffix}'. Must be '', 'a', 'b', or 'c'.")

    if suffix and (n1 != n2 or n2 != n3):
        raise ValueError(f"Suffix '{suffix}' requires square matrices: n1==n2==n3.")

    if op in MUL_OPS:
        a_struct, b_struct, mode = MUL_OPS[op]
        return _build_general(a_struct, b_struct, n1, n2, n3, mode, suffix)

    if op in T_OPS:
        struct = T_OPS[op]
        return _build_t(struct, n1, n2, n3, suffix)

    raise ValueError(f"Unknown op: {op}")

# ------------------------------
# CLI
# ------------------------------

@app.command()
def main(
    op=typer.Argument(None, help="Operation code or leave empty with --all-square"),
    n1=typer.Argument(None, help="Rows of A and C (square ops use n1==n2==n3 unless noted)"),
    n2=typer.Argument(None, help="Inner dimension / cols of A / rows of B"),
    n3=typer.Argument(None, help="Cols of B and C (and must equal n1 for packed-upper results)"),
    outdir=typer.Option("data/tensors", "--outdir", "-o", help="Output directory"),
    suffix=typer.Option("", "--suffix", "-s", help="Recursive suffix: a, b, or c"),
    all_square: bool = typer.Option(
        False,
        "--all-square",
        "-a",
        is_flag=True,
        help="Generate all square ops for n=2..8 (ignores other args)",
    ),
    all_square_recursive: bool = typer.Option(
        False,
        "--all-square-recursive",
        "-r",
        is_flag=True,
        help="Generate recursive variants for n=2..8",
    ),
):
    """
    Generate a real COO tensor T and save as int8 .npy (nnz,4),
    along with a <name>.meta.json file describing dimensions.

    With --all-square:
      - Generates all square ops: gg, ug, sg, kg, wg, uu, us, uk, uw, ul,
        ss, sk, sw, kk, ww, gt, ut, st, kt, wt
      - For sizes n = 2..8 (n1=n2=n3=n)
      - Saves as <op>-<n><n><n>.npy and <op>-<n><n><n>.meta.json under --outdir

    With --all-square-recursive:
      - Generates recursive variants for n=2..8:
        * t-ops (gt, ut, st, kt, wt): suffixes a, b, c
        * uu: suffixes a, b, c
        * u* ops (ug, us, uk, uw, ul): suffix b only
      - Saves as <op>-<n><n><n><suffix>.npy and meta.json under --outdir
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if all_square_recursive:
        t_ops = list(T_OPS.keys())
        uu_ops = ["uu"]
        u_star_ops = ["ug", "us", "uk", "uw", "ul"]

        for n in range(2, 9):
            for op_code in t_ops:
                for sfx in ["a", "b", "c"]:
                    coo = generate(op_code, n, n, n, sfx)
                    save_tensor_and_meta(coo, op_code, n, n, n, sfx, outdir)

            for sfx in ["a", "b", "c"]:
                coo = generate("uu", n, n, n, sfx)
                save_tensor_and_meta(coo, "uu", n, n, n, sfx, outdir)

            for op_code in u_star_ops:
                coo = generate(op_code, n, n, n, "b")
                save_tensor_and_meta(coo, op_code, n, n, n, "b", outdir)

        return

    if all_square:
        ops = list(MUL_OPS.keys()) + list(T_OPS.keys())
        for n in range(2, 9):
            for op_code in ops:
                coo = generate(op_code, n, n, n)
                save_tensor_and_meta(coo, op_code, n, n, n, "", outdir)
        return

    if op is None or n1 is None or n2 is None or n3 is None:
        raise typer.BadParameter(
            "Provide <op> <n1> <n2> <n3> or use --all-square / --all-square-recursive"
        )

    try:
        n1 = int(n1)
        n2 = int(n2)
        n3 = int(n3)
    except Exception:
        raise typer.BadParameter("n1, n2, n3 must be integers")

    if op not in ALLOWED:
        raise typer.BadParameter(f"Unknown op '{op}'. Allowed: {', '.join(sorted(ALLOWED))}")

    coo = generate(op, n1, n2, n3, suffix)
    save_tensor_and_meta(coo, op, n1, n2, n3, suffix, outdir)

if __name__ == "__main__":
    app()
