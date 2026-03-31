# MCTSyn

`MCTSyn` now hosts three BLIF logic synthesis search implementations in one repository:

- `alphasyn`: a minimal `AlphaSyn w/o nn` flow with MCTS
- `SASyn`: a minimal simulated annealing flow
- `MABSyn`: multi-armed bandit baselines with UCB1 and LinUCB

All implementations target `ABC`. `alphasyn` and `SASyn` reuse the checked-in `tc_public/` and `benchmarks/` directories, while `MABSyn` reuses the checked-in `tc_public/` directory. The algorithms remain separate: the MCTS path stays under `alphasyn/`, the simulated annealing path stays under `SASyn/`, and the bandit baselines stay under `MABSyn/`.

## Shared Scope

- `--dataset-root` points to the directory that contains target `.blif` files
- Dataset scan is recursive, so layouts such as `benchmarks/VTR`, `benchmarks/EPFL`, or nested `tc_public/*` are supported
- If duplicate filenames exist under one dataset root, the CLI automatically uses relative paths such as `tc_public_1/input.blif` as design names
- Backend targets `ABC` only
- Action space is fixed to `balance`, `rewrite`, `rewrite-z`, `refactor`, `refactor-z`, `resub`, `resub-z`

Set up `ABC` in `PATH` or pass `--abc-bin` explicitly:

```bash
export PATH="$HOME/abc:$PATH"
```

## `alphasyn` (MCTS)

`alphasyn` keeps the original MCTSyn behavior: step-wise MCTS with `Q + R + U` selection, `Q + R` root action choice, tree reuse, no neural network, no fixed subsequence injection, no resource allocator, and no parallel search.

Prepare a dataset manifest:

```bash
python -m alphasyn prepare-data
python -m alphasyn prepare-data --dataset-root benchmarks/VTR
```

Run the core search:

```bash
python -m alphasyn run-search --abc-bin /path/to/abc --search-iterations 64 --sequence-length 24
python -m alphasyn run-search --dataset-root tc_public --design tc_public_1/input.blif --sequence-length 10 --search-iterations 10
python -m alphasyn run-search --dataset-root tc_public --sequence-length 10 --search-iterations 10 --cpuct 1.0
python -m alphasyn run-search --dataset-root benchmarks/VTR --design bfly.abc.blif --sequence-length 10 --search-iterations 20
python -m alphasyn run-search --dataset-root benchmarks/VTR --sequence-length 10 --search-iterations 30 --cpuct 1.0
python -m alphasyn run-search --dataset-root benchmarks/VTR --design bfly.abc.blif --debug-search
```

Aggregate JSON results into CSV:

```bash
python -m alphasyn summarize
```

Common parameters:

- `--dataset-root benchmarks/VTR`: choose the concrete folder to scan
- `--design bfly.abc.blif`: choose one exact `.blif` filename under that folder; omit it to run all files in the folder tree
- `--debug-search`: save per-step root action `Q/R/U/visits` under `steps[*].action_debug`, save per-iteration node traces under `steps[*].iteration_traces`, and emit debug CSV files beside the JSON
- `--abc-bin /home/hxy/abc/abc`: explicit `ABC` path; omit it if `abc` is already in `PATH`
- `--sequence-length 24`: search sequence length
- `--search-iterations 64`: MCTS iterations per step
- `--cpuct 1.0`: exploration strength for the `U` term
- `--mu-discount 0.9`: long-term discount in backpropagation
- `--seed 0`: random seed
- `--workdir .alphasyn_work`: output directory; it must stay inside the current project directory

Outputs:

- Default output directory is `.alphasyn_work/`
- `results/*.json` stores detailed search results
- `results/*.debug.csv` stores one row per `(step, action)` when `--debug-search` is enabled
- `results/*.trace.csv` stores one row per `(step, iteration, node, action)` when `--debug-search` is enabled
- `cache/` is namespaced per `run-search` invocation and is automatically deleted when that run finishes
- `summary.csv` stores one row per design, comparing `resyn2` heuristic results against MCTS results

## `SASyn` (Simulated Annealing)

`SASyn` keeps the original simulated annealing behavior: full-sequence search with the normalized final cost
`0.7 * (and / initial_and) + 0.3 * (lev / initial_lev)`.

Prepare a dataset manifest:

```bash
python -m SASyn prepare-data
python -m SASyn prepare-data --dataset-root benchmarks/VTR
```

Run simulated annealing search:

```bash
python -m SASyn run-search --dataset-root tc_public --design tc_public_1/input.blif --sequence-length 10 --search-iterations 1000
python -m SASyn run-search --dataset-root tc_public --sequence-length 10 --search-iterations 1000
python -m SASyn run-search --dataset-root benchmarks/VTR --design bfly.abc.blif --search-iterations 2000 --debug-search
```

Aggregate JSON results into CSV:

```bash
python -m SASyn summarize
```

Common parameters:

- `--dataset-root`: root directory containing `.blif` files
- `--design`: run one exact design under the dataset root; omit it to run all discovered designs
- `--sequence-length`: fixed sequence length used by simulated annealing
- `--search-iterations`: total annealing iterations
- `--and-weight`: weight of normalized AND count in the final cost
- `--lev-weight`: weight of normalized level count in the final cost
- `--initial-temperature`: override the auto-calibrated starting temperature
- `--min-temperature`: lower bound of the geometric cooling schedule
- `--seed`: random seed
- `--debug-search`: emit per-iteration CSV traces
- `--workdir .sasyn_work`: output directory; it must stay inside the current project directory

Outputs:

- Default output directory is `.sasyn_work/`
- `manifest.json`: dataset manifest
- `base_aig/`: per-design temporary AIG starting points generated from input BLIF files
- `results/*.json`: per-design search results
- `results/*.debug.csv`: per-iteration trace when `--debug-search` is enabled
- `summary.csv`: aggregated report

Each SA result JSON includes the final sequence as an `ABC` command string, final `and` and `lev`, baseline information derived from the initial design and `resyn2`, and the full annealing trace under `iterations`.

## `MABSyn` (Bandit Baselines)

`MABSyn` keeps two original bandit-style baselines:

- `baseline_mab`: UCB1 sequence search
- `linucb`: contextual bandit search with LinUCB

Both are now invoked with subcommands and store generated files under `.mabsyn_work/` by default.

Run search:

```bash
python -m MABSyn.baseline_mab run-search --dataset-root tc_public --design tc_public_1/input.blif --episodes 20 --steps 10 --ucb-c 1.5 --debug-search
python -m MABSyn.baseline_mab run-search --dataset-root benchmarks/VTR --design bfly.abc.blif --episodes 20 --steps 10
python -m MABSyn.baseline_mab run-search --dataset-root tc_public --episodes 20 --steps 10
python -m MABSyn.linucb run-search --dataset-root tc_public --design tc_public_1/input.blif --episodes 20 --steps 10
python -m MABSyn.linucb run-search --dataset-root benchmarks/VTR --design bfly.abc.blif --episodes 20 --steps 10
python -m MABSyn.linucb run-search --dataset-root tc_public --episodes 20 --steps 10
```

Summaries:

```bash
python -m MABSyn.baseline_mab summarize
python -m MABSyn.linucb summarize
```

Common parameters:

- `--workdir .mabsyn_work`: root directory for result JSON, summaries, and caches
- `--abc-bin`: explicit `ABC` path
- `--dataset-root`: root directory containing `.blif` files
- `--design`: run one exact design under the dataset root; omit it to run all discovered designs
- `--steps`: fixed sequence length or steps per episode
- `--episodes`: total episodes / iterations used by the method
- `--seed`: random seed
- `--actions`: comma-separated action override
- `--result-json`: optional explicit aggregate output path
- `--debug-search`: enable detailed per-episode debug artifacts for `baseline_mab`

Method-specific parameters remain unchanged, including `--ucb-c` for `baseline_mab` and `--alpha`, `--linucb-alpha`, `--lambda`, `--linucb-lambda`, `--long-term-rollouts`, and `--long-term-horizon` for `linucb`.

Outputs:

- Default root is `.mabsyn_work/`
- `baseline_mab` and `linucb` no longer write default aggregate `*_results.json` files; pass `--result-json` if you want one
- Per-benchmark results are stored under `.mabsyn_work/results/baseline_mab/` and `.mabsyn_work/results/linucb/`
- `baseline_mab` writes `.debug.csv` and `.debug.json` beside the result JSON when `--debug-search` is enabled
- Summary CSV is written under `.mabsyn_work/results/summary.csv` or per-method summary paths
- LinUCB cache files are written under `.mabsyn_work/cache/linucb/`

## Testing

The repository uses the standard library `unittest` suite:

```bash
python -m unittest discover -s tests -v
```

python compare_algorithm_summaries.py \
  --output your_aggregate.csv \
  --details-output your_details.csv
