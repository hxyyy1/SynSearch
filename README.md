# MCTSyn

`MCTSyn` contains a minimal `AlphaSyn w/o nn` implementation for BLIF designs, using a Python MCTS front-end and an `ABC` backend.

## Scope

- `--dataset-root` should point to the folder that contains the target `.blif` files
- The scan is recursive, so folders such as `benchmarks/VTR`, `benchmarks/EPFL`, or any future subfolder layout are supported
- Backend only targets `ABC`
- Action space is fixed to `balance`, `rewrite`, `rewrite-z`, `refactor`, `refactor-z`, `resub`, `resub-z`
- No neural network, no fixed subsequence injection, no resource allocator, no parallel search

## Commands

export PATH="$HOME/abc:$PATH"

Prepare a dataset manifest:

```bash
python -m alphasyn prepare-data
python -m alphasyn prepare-data --dataset-root benchmarks/VTR
```

Run the core search:

```bash
python -m alphasyn run-search --abc-bin /path/to/abc --search-iterations 64 --sequence-length 24
python -m alphasyn run-search --dataset-root tc_public/tc_public_1 --design input.blif --sequence-length 10 --search-iterations 10
python -m alphasyn run-search --dataset-root benchmarks/VTR --design bfly.abc.blif --sequence-length 10 --search-iterations 20
python -m alphasyn run-search --dataset-root benchmarks/VTR --sequence-length 10 --search-iterations 10
```


Common parameters:

- `--dataset-root benchmarks/VTR`: choose the concrete folder to scan
- `--design bfly.abc.blif`: choose one exact `.blif` filename under that folder; omit it to run all files in the folder tree
- If one `--dataset-root` contains duplicate filenames, narrow the folder path further so each `.blif` filename is unique
- `--abc-bin /home/hxy/abc/abc`: explicit `ABC` path; omit it if `abc` is already in `PATH`
- `--sequence-length 24`: search sequence length
- `--search-iterations 64`: MCTS iterations per step
- `--cpuct 1.0`: exploration strength for the `U` term
- `--mu-discount 0.9`: long-term discount in backpropagation
- `--seed 0`: random seed
- `--workdir .alphasyn_work`: output directory; it must stay inside the current project directory

Aggregate JSON results into CSV:

```bash
python -m alphasyn summarize
```

The summary table is one row per design with these columns:

- `design_name`
- `initial_and`
- `initial_lev`
- `heuristic_and`
- `heuristic_lev`
- `final_and`
- `final_lev`

## Outputs

- All generated files stay under the current project directory
- Default output directory is `.alphasyn_work/`
- `results/*.json` stores detailed search results
- `cache/` is namespaced per `run-search` invocation, so rerunning the same command will evaluate prefixes again instead of reusing a previous run's on-disk cache
- In each result JSON, `sequence` is stored as a plain `ABC` command string that can be copied directly into `abc`
- `summary.csv` stores one row per design, comparing `resyn2` heuristic results against MCTS results

## Testing

The repository uses the standard library `unittest` suite:

```bash
python -m unittest discover -s tests -v
```



项目构成：
alphasyn/cli.py：命令行入口，提供 prepare-data、run-search、summarize 三个子命令。
alphasyn/types.py：核心数据结构，定义了 SearchConfig、SearchResult、BaselineInfo、BackendResult，也放了固定动作空间和 baseline 用的启发式脚本展开序列。
alphasyn/backend.py：ABC 后端封装。负责：读取 tc_public/*/input.blif，执行动作前缀，调 print_stats 解析 and = ...，把每个前缀结果缓存到 .alphasyn_work/cache/
alphasyn/mcts.py：AlphaSyn w/o nn 的核心搜索逻辑。当前实现是：单线程、无神经网络、无资源分配、无固定子序列注入、Selection 用 Q + R + U、最终决策用根节点 argmax(Q + R)、树复用开启
alphasyn/dataset.py：扫描 tc_public、生成 manifest、做文件 hash。
tests/test_core.py：不依赖真实 ABC 的单元测试，覆盖奖励、解析和 MCTS 决策。
README.md：基础说明。

数据和输出：
输入数据固定在 tc_public 下，目录形如 tc_public_1/input.blif。
运行后默认输出到 .alphasyn_work/：manifest.json、cache/*.aig 和 cache/*.json、results/*.json、summary.csv
