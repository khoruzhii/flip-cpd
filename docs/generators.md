# Tensor generators

This document describes the three tensor generators in `scripts/`:

- `generator.py`   – real structured matrix multiplication operators,
- `generator-c.py` – complex matrix operators encoded as real tensors,
- `generator-pm.py` – polynomial multiplication operators (real and complex).

All generators produce **real 4-tensors in COO format** and save them as a pair of files:

- `*.npy` – tensor data, shape `(nnz, 4)`, `dtype=int8`,
- `*.meta.json` – metadata describing dimensions and sizes.

These tensors are consumed by downstream C++ code as fixed, precomputed linear operators.

## 1. Common output format

### 1.1. `.npy`: COO tensor

Each generator writes a single `.npy` file containing a 2D array of shape `(nnz, 4)` and type `int8`:

| column | name   | meaning                                                                |
|--------|--------|------------------------------------------------------------------------|
| 0      | alpha  | index of a parameter of the left operand (U-space)                    |
| 1      | beta   | index of a parameter of the right operand (V-space)                   |
| 2      | gamma  | index of an output coordinate (W-space)                               |
| 3      | val    | coefficient in `{−1, +1}` (zero entries are not stored explicitly)    |

Constraints:

- `alpha`, `beta`, `gamma` are integers in the closed range `[0, 63]`.
- `val` is always `+1` или `−1`.
- The COO representation is purely structural; all semantics (which operator, which layout, which shapes) come from `.meta.json` and from the operation code (`op`).

The *layout* of `gamma` depends on the operation:

- **general** (dense matrix)  
  `gamma` encodes `(row, col)` in row-major order:  
  `gamma = i * n3 + k` for `i ∈ [0, n1), k ∈ [0, n3)`.

- **packed upper triangular** (symmetric result `Sym_n`)  
  Only entries with `i ≤ j` are stored; `(i, j)` is mapped to a flat index in `[0, n(n+1)/2)` via
  the helper `sym_pack(i, j, n)`.

- **packed strictly upper triangular** (skew-symmetric result `Skew_n`)  
  Only entries with `i < j` are stored; `(i, j)` is mapped to `[0, n(n−1)/2)` via `skew_pack(i, j, n)`.

The particular variant (general vs packed) is determined by the operation code (`op`).

### 1.2. `.meta.json`: metadata

For each tensor `<name>.npy` there is a matching `<name>.meta.json` with at least the following keys:

- `name` – tensor name (matches the filename stem),
- `op` – operation code (`"gg"`, `"gtr"`, `"pmc"`, …),
- `nU` – dimension of the left parameter space (max `alpha + 1`),
- `nV` – dimension of the right parameter space (max `beta + 1`),
- `nW` – dimension of the output space (max `gamma + 1`).

Additional fields depend on the generator:

- **Matrix generators (`generator.py`, `generator-c.py`)**

  ```jsonc
  {
    "op": "gg",
    "n1": 3,         // rows of A and C
    "n2": 3,         // inner dimension
    "n3": 3          // cols of B and C (also size for gamma layout)
  }

  ```

* **Polynomial generator (`generator-pm.py`)**

  ```jsonc
  {
    "op": null,
    "deg1": 2,       // degree of first polynomial
    "deg2": 2,       // degree of second polynomial
    "deg3": 4,       // degree of result
    "n1": 3,         // lengths n1 = deg1 + 1
    "n2": 3,
    "n3": 5
  }
  ```

Downstream code is expected to treat `.npy` and `.meta.json` as an inseparable pair.

### 1.3. Naming conventions

Each tensor has a name of the form:

* **Real matrices (`generator.py`)**

  ```text
  <name> = <op>-<n1><n2><n3><suffix>
  ```

  where:

  * `<op>` ∈ `{gg, ug, sg, kg, wg, uu, us, uk, uw, ul, ss, sk, sw, kk, ww,
               gt, ut, st, kt, wt}`,
  * `<n1>`, `<n2>`, `<n3>` – integer sizes,
  * `<suffix>` – optional recursive suffix: `""`, `"a"`, `"b"`, `"c"`.

* **Complex matrices (`generator-c.py`)**

  ```text
  <name> = <op>-<n1><n2><n3>
  ```

* **Polynomials (`generator-pm.py`)**

  ```text
  <name> = <op>-<deg1><deg2><deg3>
  ```

Thus for any given `name` the corresponding files are:

```text
data/tensors/<name>.npy
data/tensors/<name>.meta.json
```

## 2. `generator.py`: real structured matrices

`generator.py` builds tensors for real matrix multiplication operators

```text
C = A B
or
C = X X^T
```

with various structures for `A`, `B`, `X`. All tensors are real, and all structure is encoded via index layouts and signs.

### 2.1. Supported structures

Structures are encoded as single letters:

| code | name             | description                                   | square-only | #params for `n×n` |
| ---- | ---------------- | --------------------------------------------- | ----------: | ----------------- |
| `g`  | general          | full matrix                                   |          no | `n_rows * n_cols` |
| `u`  | upper-triangular | includes diagonal                             |         yes | `n(n+1)/2`        |
| `l`  | lower-triangular | includes diagonal                             |         yes | `n(n+1)/2`        |
| `s`  | symmetric        | `A[i,j] = A[j,i]`                             |         yes | `n(n+1)/2`        |
| `k`  | skew-symmetric   | `A[i,j] = −A[j,i]`, diagonal is zero          |         yes | `n(n−1)/2`        |
| `w`  | skew+diag        | off-diagonal as skew; diagonal is independent |         yes | `n(n+1)/2`        |

Notes:

* Non-general structures (`u`, `l`, `s`, `k`, `w`) require square matrices.
* For `k` (skew), diagonal entries are omitted from the parameterization.
* For `w`, diagonal entries are stored as in the symmetric case; off-diagonal entries follow skew-symmetric sign rules.

### 2.2. Supported operations

Operation codes combine left and right structures:

* `MUL_OPS` – general products `C = A B`,
* `T_OPS` – “t-ops”, products of the form `C = X X^T`.

#### 2.2.1. Multiplication `C = A B`

| op   | A structure | B structure | gamma layout | modes / notes                    |
| ---- | ----------- | ----------- | ------------ | -------------------------------- |
| `gg` | `g`         | `g`         | general      | fully dense matrices             |
| `ug` | `u`         | `g`         | general      | left upper-triangular            |
| `sg` | `s`         | `g`         | general      | left symmetric                   |
| `kg` | `k`         | `g`         | general      | left skew-symmetric              |
| `wg` | `w`         | `g`         | general      | left skew+diag                   |
| `uu` | `u`         | `u`         | packed upper | result in `Sym_n` (upper-packed) |
| `us` | `u`         | `s`         | general      |                                  |
| `uk` | `u`         | `k`         | general      |                                  |
| `uw` | `u`         | `w`         | general      |                                  |
| `ul` | `u`         | `l`         | general      |                                  |
| `ss` | `s`         | `s`         | general      |                                  |
| `sk` | `s`         | `k`         | general      |                                  |
| `sw` | `s`         | `w`         | general      |                                  |
| `kk` | `k`         | `k`         | general      |                                  |
| `ww` | `w`         | `w`         | general      |                                  |

Constraints:

* For non-general structures the corresponding operand must be square:

  * left operand: `n1 == n2` for structures `u`, `l`, `s`, `k`, `w`,
  * right operand: `n2 == n3` for structures `u`, `l`, `s`, `k`, `w`.
* For `uu` the result is symmetric; `gamma` uses packed-upper layout (`Sym_n`).

#### 2.2.2. t-operations `C = X X^T`

Codes:

* `gt`, `ut`, `st`, `kt`, `wt`.

Interpretation:

* `X` has the structure from the first letter (`g`, `u`, `s`, `k`, `w`),
* `C = X X^T` is symmetric,
* result is stored in `Sym_n` (packed upper) with

  ```text
  n3 == n1
  gamma ∈ [0, n1*(n1+1)/2)
  ```

There is a single parameter pool for `X` shared by both sides; nevertheless `nU` and `nV` in metadata are both set equal to the number of parameters of `X` (so `alpha` and `beta` live in identical copies of the same index set).

### 2.3. Gamma layout and packing

For **general** outputs:

* `C` has shape `(n1, n3)`,
* `gamma = i * n3 + k`, `i ∈ [0, n1), k ∈ [0, n3)`.

For **packed symmetric** outputs (`mode = "packed_upper"`):

* `C` is logically symmetric `n1 × n1`,
* only `i ≤ k` are stored,
* `gamma = sym_pack(i, k, n1)` where `sym_pack` flattens the upper triangle into `[0, n1(n1+1)/2)`.

The internal helpers guarantee that `I, J, K` are filtered consistently with the structural constraints, and only non-zero contributions survive in the COO tensor.

### 2.4. Recursive suffixes

Some operations support **recursive variants** controlled by the suffix:

* suffix values:

  * `""` – default, no filtering,
  * `"a"` – exclude the “top-left” output index,
  * `"b"` – exclude the “bottom-right” output index,
  * `"c"` – exclude both.

The mapping “top-left” and “bottom-right” depends on `gamma` layout:

* **general** (`n1 × n3`):

  * first output index: `gamma = 0` corresponds to `(i, k) = (0, 0)`,
  * last output index: `gamma = n1*n3 − 1`.

* **packed upper (`Sym_n`)**:

  * first output index: `gamma = 0` corresponds to `(0, 0)`,
  * last output index: `gamma = n(n+1)/2 − 1` corresponds to `(n−1, n−1)`.

The suffix is applied via a simple filter on the `gamma` column. It is only allowed for **square** problems:

```text
suffix in {a, b, c}  ⇒  n1 == n2 == n3
```

Supported in the CLI:

* all `t`-ops (`gt`, `ut`, `st`, `kt`, `wt`): suffix `a`, `b`, `c`,
* `uu`: suffix `a`, `b`, `c`,
* `u*` ops (`ug`, `us`, `uk`, `uw`, `ul`): suffix `b` only (in batch recursive mode).

### 2.5. CLI usage (`generator.py`)

Single tensor:

```bash
python scripts/generator.py gg 3 3 3
python scripts/generator.py uu 4 4 4 --suffix a
python scripts/generator.py ut 5 5 5 -o custom/dir
```

Batch generation for square matrices:

* All square ops, sizes `n = 2..8`:

  ```bash
  python scripts/generator.py --all-square
  ```

  This generates all operations from `MUL_OPS` and `T_OPS` with `n1 = n2 = n3 = n`.

* Recursive variants for square matrices:

  ```bash
  python scripts/generator.py --all-square-recursive
  ```

  This generates:

  * for each `n = 2..8`:

    * `gt`, `ut`, `st`, `kt`, `wt` with suffixes `a`, `b`, `c`,
    * `uu` with suffixes `a`, `b`, `c`,
    * `ug`, `us`, `uk`, `uw`, `ul` with suffix `b`.

The output directory is controlled by `--outdir` / `-o` (default: `data/tensors`).

## 3. `generator-c.py`: complex matrices

`generator-c.py` builds real COO tensors for linear operators on **complex matrices**. The parameters are always real; complex quantities are represented by splitting real and imaginary parts.

### 3.1. Parameterization of complex matrices

For a complex matrix `X ∈ C^{p×q}`:

* The underlying parameter vector is of length `2 * p * q`,
* The first `p*q` entries are `Re(X)` in row-major order,
* The next `p*q` entries are `Im(X)` in row-major order.

Thus:

```text
nU = 2 * p * q
nV = 2 * r * s
```

depending on the shapes of the left and right operands.

Indexing helpers (informal):

* real part: `idx_real(i, j, n_rows, n_cols) = i * n_cols + j`,
* imag part: `idx_imag(i, j, n_rows, n_cols) = i * n_cols + j + n_rows * n_cols`.

`alpha` and `beta` always refer to such unified parameter pools.

### 3.2. Supported operations

Operations are grouped by the type of product:

* `MUL_OPS`: general multiplication `A B`,
* `T_OPS`: products `X X^T`,
* `H_OPS`: products `X X^H` (Hermitian).

#### 3.2.1. General multiplication `C = A B`

Operand sizes:

* `A ∈ C^{n1 × n2}`,
* `B ∈ C^{n2 × n3}`,
* `C ∈ C^{n1 × n3}`.

Codes:

| op    | meaning                      | `nU`      | `nV`      | `nW`      | gamma layout       |
| ----- | ---------------------------- | --------- | --------- | --------- | ------------------ |
| `ggr` | `Re(A B)`                    | `2*n1*n2` | `2*n2*n3` | `n1*n3`   | general (`n1*n3`)  |
| `ggi` | `Im(A B)`                    | `2*n1*n2` | `2*n2*n3` | `n1*n3`   | general            |
| `ggc` | `[Re(A B); Im(A B)]` stacked | `2*n1*n2` | `2*n2*n3` | `2*n1*n3` | two general blocks |

For `ggc` the result layout is:

* `gamma ∈ [0, n1*n3)` – `Re(A B)`,
* `gamma ∈ [n1*n3, 2*n1*n3)` – `Im(A B)`.

#### 3.2.2. Transpose products `C = X X^T`

Operand:

* `X ∈ C^{n1 × n2}`,

Constraints:

* `n3 == n1`,
* result is symmetric; gamma uses packed upper layout (`Sym_n1`).

Codes:

| op    | meaning                          | `nU`, `nV`          | `nW`               | gamma layout            |
| ----- | -------------------------------- | ------------------- | ------------------ | ----------------------- |
| `gtr` | `Re(X X^T)`                      | `nU = nV = 2*n1*n2` | `sym = n1(n1+1)/2` | `Sym_n1` upper-packed   |
| `gti` | `Im(X X^T)`                      | same                | `sym`              | upper-packed            |
| `gtc` | `[Re(X X^T); Im(X X^T)]` stacked | same                | `2*sym`            | two packed upper blocks |

For `gtc`:

* `gamma ∈ [0, sym)` – `Re(X X^T)`,
* `gamma ∈ [sym, 2*sym)` – `Im(X X^T)`.

#### 3.2.3. Hermitian products `C = X X^H`

Operand:

* `X ∈ C^{n1 × n2}`,

Constraints:

* `n3 == n1`.

The Hermitian product `X X^H` is always Hermitian; its real part is symmetric, imaginary part is skew-symmetric.

Codes:

| op    | meaning                                       | `nU`, `nV`          | `nW`                | gamma layout                     |
| ----- | --------------------------------------------- | ------------------- | ------------------- | -------------------------------- |
| `ghr` | `Re(X X^H)`                                   | `nU = nV = 2*n1*n2` | `sym = n1(n1+1)/2`  | `Sym_n1` packed upper            |
| `ghi` | `Im(X X^H)`                                   | same                | `skew = n1(n1−1)/2` | `Skew_n1` packed strictly upper  |
| `ghc` | `[Re(X X^H) in Sym_n1; Im(X X^H) in Skew_n1]` | same                | `sym + skew = n1^2` | first Sym block, then Skew block |

For `ghc`:

* `gamma ∈ [0, sym)` – `Re(X X^H)` in `Sym_n1`,
* `gamma ∈ [sym, sym + skew)` – `Im(X X^H)` in `Skew_n1` (strictly upper, via `skew_pack`).

### 3.3. Sign conventions

Internally each builder decomposes complex products into real contributions. The tensors follow the standard algebra:

* For `C = A B`:

  ```text
  A = Ar + i Ai
  B = Br + i Bi

  Re(A B) = Ar·Br − Ai·Bi
  Im(A B) = Ar·Bi + Ai·Br
  ```

  This is reflected in the COO rows:

  * `Re`: contributions from `(Ar, Br)` with `val = +1`, and from `(Ai, Bi)` with `val = −1`,
  * `Im`: contributions from `(Ar, Bi)` and `(Ai, Br)` with `val = +1`.

* For `C = X X^T` the same pattern holds, but the right index is transposed.

* For `C = X X^H` the Hermitian conjugate introduces different signs for `Im`:

  ```text
  X^H = (X̄)^T

  Im(X X^H) comes from Xi·Xr^H ( +1 ) and Xr·Xi^H ( −1 )
  ```

Those patterns are encoded via `val` in the generated tensor and explained here so downstream consumers can validate them against their own implementations.

### 3.4. CLI usage (`generator-c.py`)

Single tensor:

```bash
python scripts/generator-c.py ggr 3 4 5
python scripts/generator-c.py gtc 4 3 4          # n3 must equal n1
python scripts/generator-c.py ghc 5 2 5 -o data/tensors
```

Arguments:

* `op`   – one of `ggr, ggi, ggc, gtr, gti, gtc, ghr, ghi, ghc`,
* `n1`   – rows of `A` or `X` and rows of the result,
* `n2`   – inner dimension / columns of `A` or `X`,
* `n3`   – columns of `B` and result; must satisfy `n3 == n1` for transpose/Hermitian ops,
* `--outdir` – output directory (default: `data/tensors`).

The `.meta.json` file contains `nU`, `nV`, `nW`, `op`, and `n1`, `n2`, `n3`.

## 4. `generator-pm.py`: polynomial multiplication

`generator-pm.py` builds real COO tensors for convolution-style polynomial multiplication, in both real and complex settings. The CLI works in terms of **degrees**, not lengths.

### 4.1. Degrees and lengths

Let:

* `d1` – degree of the first polynomial,
* `d2` – degree of the second polynomial,
* `d3` – degree of the result.

Then lengths are:

```text
n1 = d1 + 1
n2 = d2 + 1
n3 = d3 + 1
```

Convolution constraint:

```text
0 ≤ d3 ≤ d1 + d2
```

Equivalently:

```text
1 ≤ n3 ≤ n1 + n2 − 1
```

These constraints are enforced both in the generator and at the CLI level.

### 4.2. Parameterization

We view polynomials as vectors of coefficients:

* For real polynomials:

  ```text
  A(x) = Σ_{i=0}^{n1−1} a_i x^i
  B(x) = Σ_{j=0}^{n2−1} b_j x^j
  ```

* For complex polynomials:

  ```text
  A_i = a_i^r + i a_i^i
  B_j = b_j^r + i b_j^i
  ```

Complex coefficients are parameterized similarly to matrices:

* The parameter vector for a polynomial of length `n` is of length `2*n`,
* Reals first, then imaginaries:

  ```text
  params = [Re[0], ..., Re[n−1], Im[0], ..., Im[n−1]]
  ```

Indexing (conceptually):

* `idx_real_poly(i, n) = i`,
* `idx_imag_poly(i, n) = i + n`.

### 4.3. Supported operations

The generator supports four operations:

| op    | description                         | left params (`nU`) | right params (`nV`) | result length (`nW`) |
| ----- | ----------------------------------- | ------------------ | ------------------- | -------------------- |
| `pm`  | real polynomial multiplication      | `n1`               | `n2`                | `n3`                 |
| `pmr` | `Re(A * B)` for complex polynomials | `2*n1`             | `2*n2`              | `n3`                 |
| `pmi` | `Im(A * B)` for complex polynomials | `2*n1`             | `2*n2`              | `n3`                 |
| `pmc` | `[Re(A * B); Im(A * B)]` stacked    | `2*n1`             | `2*n2`              | `2*n3`               |

Here `n1 = d1 + 1`, `n2 = d2 + 1`, `n3 = d3 + 1`.

#### 4.3.1. Real multiplication `pm`

For real polynomials:

```text
C(x) = A(x) * B(x)
c_k = Σ_{i+j = k} a_i b_j
```

The generator:

* enumerates all pairs `(i, j)` with `0 ≤ i < n1`, `0 ≤ j < n2`,
* keeps only those with `k = i + j < n3`,
* emits rows `[alpha = i, beta = j, gamma = k, val = +1]`.

#### 4.3.2. Complex multiplication `pmr`, `pmi`, `pmc`

For complex polynomials:

```text
A_i = a_i^r + i a_i^i
B_j = b_j^r + i b_j^i
C_k = Σ_{i+j = k} A_i B_j
```

We have:

```text
Re(C_k) = Σ_{i+j = k} (a_i^r b_j^r − a_i^i b_j^i)
Im(C_k) = Σ_{i+j = k} (a_i^r b_j^i + a_i^i b_j^r)
```

The generators follow this decomposition:

* `pmr` (real part):

  * contributions from `(Re(A_i), Re(B_j))` with `val = +1`,
  * contributions from `(Im(A_i), Im(B_j))` with `val = −1`.

* `pmi` (imag part):

  * contributions from `(Re(A_i), Im(B_j))` with `val = +1`,
  * contributions from `(Im(A_i), Re(B_j))` with `val = +1`.

* `pmc` (stacked):

  * builds `pmr` and `pmi` internally,
  * concatenates their COO rows,
  * shifts `gamma` of the imaginary block by `n3`:

    ```text
    gamma ∈ [0, n3)          – Re(C)
    gamma ∈ [n3, 2*n3)       – Im(C)
    ```

### 4.4. CLI usage (`generator-pm.py`)

Single tensor:

```bash
# real multiplication: degrees 2, 3, 5 (lengths 3, 4, 6)
python scripts/generator-pm.py pm 2 3 5

# complex result, real part only
python scripts/generator-pm.py pmr 4 4 6

# stacked [Re; Im]
python scripts/generator-pm.py pmc 2 2 4 -o data/tensors
```

Arguments:

* `op`  – `pm`, `pmr`, `pmi`, or `pmc`,
* `d1`  – degree of the first polynomial (`≥ 0`),
* `d2`  – degree of the second polynomial (`≥ 0`),
* `d3`  – degree of the result (`0 ≤ d3 ≤ d1 + d2`),
* `--outdir` – output directory (default: `data/tensors`).

Metadata includes both degrees (`deg1`, `deg2`, `deg3`) and lengths (`n1`, `n2`, `n3`) so downstream code can work in either convention.

