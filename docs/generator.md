# Structured tensor generator (`generator.py`)

`generator.py` builds real 4-tensor representations of structured matrix multiplication operators and writes them to disk.

Each generated tensor is stored in two files under `data/tensors/` (or a custom `--outdir`):

- `<name>.npy` – NumPy array of shape `(nnz, 4)` and type `int8`, in COO format:
  - column 0: `alpha` – index of a parameter of the left operand (U-space),
  - column 1: `beta`  – index of a parameter of the right operand (V-space),
  - column 2: `gamma` – index of an output coordinate (W-space),
  - column 3: `val`   – coefficient in `{−1, 0, +1}`.
- `<name>.meta.json` – metadata describing the tensor:
  - `name` – tensor name (matches the file stem),
  - `nU`, `nV`, `nW` – dimensions of the parameter and output spaces,
  - `op` – operation code (e.g. `"gg"`, `"ut"`, `"sw"`),
  - `n1`, `n2`, `n3` – matrix sizes used to construct the tensor.

The tensor name is derived from the operation and sizes:
`<name> = <op>-<n1><n2><n3><suffix>`, where `suffix` is optional (`""`, `"a"`, `"b"`, or `"c"` for recursive variants).

Typical usages to generate Single tensor:

```bash
python scripts/generator.py gg 3 3 3
```

Downstream C++ code can recover `nU`, `nV`, `nW` (and optionally `op`, `n1`, `n2`, `n3`) by loading the corresponding `.meta.json` for a given tensor name.
