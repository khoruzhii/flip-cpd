# Command-line tools

This document describes the command-line tools provided by the C++ part of the project.

Most tools come in two variants:

- suffix `2` (e.g. `search2`, `lift2`, `select2`, `select_modular2`) — operates over the field $F_2$,
- suffix `3` (e.g. `search3`, `lift3`, `select3`, `select_modular3`) — operates over the field $F_3$.

Apart from the field, the interfaces of the `*2` and `*3` variants are identical. All tools expect tensor metadata and data in `data/tensors/<name>.meta.json` and `data/tensors/<name>.npy`. By default, intermediate and final schemes are read from and written to the `data/` subdirectories:

- modular schemes: `data/schemes_modp/`
- lifted schemes: `data/schemes_lifted/`
- rational schemes: `data/schemes_rational/`
- selected schemes: `data/schemes_selected/` (and the immutable reference schemes from the paper in `data/schemes_paper/`)

An end-to-end example using all tools is provided in the `examples/` folder.

## `search`

Pool-based flip-graph search for low-rank tensor decompositions over a finite field.

Typical usage is:

```bash
bin/search2 gt-444 --plus --save
````

which runs a pool-based search for schemes of the tensor `gt-444` over $\mathbb{F}_2$ and saves the discovered modular schemes and logs under `data/schemes_modp/gt-444/` and `data/logs/gt-444/`. The `search3` binary behaves identically, but works over $\mathbb{F}_3$.

The full help output is:

```text
Pool-based Tensor Flip Graph Search

bin\search2 [OPTIONS] name

POSITIONALS:
  name TEXT REQUIRED          Tensor name (e.g., gt-444)

OPTIONS:
  -h,     --help              Print this help message and exit
          --id TEXT           Output identifier (affects scheme names only)
  -f,     --path-limit INT [1000000]
                              Path length limit
  -s,     --pool-size INT [200]
                              Pool size limit
  -r,     --target-rank INT [0]
                              Target rank
  -p,     --plus-lim INT [50000]
                              Flips before plus transition
  -t,     --threads INT:POSITIVE [4]
                              Number of worker threads
  -m,     --max-attempts INT:POSITIVE [1000]
                              Max attempts per rank level
          --stop INT:POSITIVE [20000]
                              Stop if nothing found after this many attempts
          --plus              Enable plus transitions
          --save              Save verified pools to files and JSON logs
  -v,     --verbose INT [0]   Verbose level (0 = default, 1 = show pool progress)
```

## `lift`

Hensel lifting and rational reconstruction of modular schemes.

Typical usage is:

```bash
bin/lift2 gt-444 34
```

which reads modular rank-34 schemes for `gt-444` (produced by `search2`), lifts them from $F_2$ to an extension field $F_{2^k}$, and then reconstructs coefficients over $\mathbb{Z}$ or $\mathbb{Q}$. The output is written to `data/schemes_lifted/gt-444/` and `data/schemes_rational/gt-444/`. The `lift3` binary behaves identically over $F_3$.

The full help output is:

```text
Hensel lifting + rational reconstruction

bin\lift2 [OPTIONS] name rank

POSITIONALS:
  name TEXT REQUIRED          Tensor name (e.g., gt-444)
  rank INT REQUIRED           Rank of schemes

OPTIONS:
  -h,     --help              Print this help message and exit
  -k,     --steps INT [10]    Number of lifting steps
          --id TEXT           Output identifier (affects output paths only)
  -t,     --threads UINT      Number of threads
  -b,     --bound INT         Reconstruction bound (default: sqrt(M/2))
  -v,     --verify            Verify reconstructed schemes against tensor
```

## `select`

Selection of Pareto-optimal schemes with recursion analysis.

A typical call is:

```bash
bin/select2 gt-444 34
```

which reads rational rank-34 schemes for `gt-444` from `data/schemes_rational/gt-444/`, analyses possible recursions, and selects Pareto-optimal schemes according to rank, recursion-related costs and number of additions. The selected schemes are written in both `.npy` and `.txt` formats to `data/schemes_selected/`. The `select3` binary behaves identically over $F_3$.

The full help output is:

```text
Pareto-optimal scheme selection with recursion analysis

bin\select2 [OPTIONS] name rank

POSITIONALS:
  name TEXT REQUIRED          Tensor name (e.g., gt-444, ggc-333)
  rank INT REQUIRED           Rank of schemes

OPTIONS:
  -h,     --help              Print this help message and exit
          --path TEXT         Path to rational schemes .npy. If omitted, all matching files are
                              loaded from data/schemes_rational/<name>/
          --output-dir TEXT [data/schemes_selected]
                              Output directory
          --semicolon         Add semicolons in .txt output
```
