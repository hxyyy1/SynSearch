# MCTSyn

`MCTSyn` 现在在同一个仓库中包含四种 BLIF 逻辑综合搜索实现：

- `alphasyn`：MCTS 流程
- `hybridsyn`：UCB1 预热启动加 MCTS 续搜
- `SASyn`：模拟退火流程
- `MABSyn`：基于 UCB1 和 LinUCB 的多臂老虎机基线方法

所有实现都以 `ABC` 为目标。`alphasyn`、`hybridsyn` 和 `SASyn` 复用仓库中已有的 `tc_public/` 与 `benchmarks/` 目录，而 `MABSyn` 复用仓库中已有的 `tc_public/` 目录。各算法彼此独立：纯 MCTS 路径位于 `alphasyn/`，混合 UCB1+MCTS 路径位于 `hybridsyn/`，模拟退火路径位于 `SASyn/`，而 bandit 方法位于 `MABSyn/`。

## 共享范围

- `--dataset-root` 指向包含目标 `.blif` 文件的目录
- 数据集扫描是递归的，因此支持 `benchmarks/VTR`、`benchmarks/EPFL` 或嵌套的 `tc_public/*` 等目录布局
- 如果同一个 dataset root 下存在重名文件，CLI 会自动使用诸如 `tc_public_1/input.blif` 这样的相对路径作为 design 名称
- 后端仅支持 `ABC`
- 动作空间固定为 `balance`、`rewrite`、`rewrite-z`、`refactor`、`refactor-z`、`resub`、`resub-z`

将 `ABC` 加入 `PATH`，或者显式传入 `--abc-bin`：

```bash
export PATH="$HOME/abc:$PATH"
```

## `alphasyn`（MCTS）

`alphasyn` 保留了原始 MCTSyn 的行为：逐步执行的 MCTS，使用 `Q + R + U` 进行选择，在根节点使用 `Q + R` 选择动作，支持树复用，不使用神经网络，不注入固定子序列，不使用资源分配器，也不进行并行搜索。

准备数据集清单：

```bash
python -m alphasyn prepare-data
python -m alphasyn prepare-data --dataset-root benchmarks/VTR
```

运行核心搜索：

```bash
python -m alphasyn run-search --abc-bin /path/to/abc --search-iterations 64 --sequence-length 24
python -m alphasyn run-search --dataset-root tc_public --design tc_public_1/input.blif --sequence-length 10 --search-iterations 10
python -m alphasyn run-search --dataset-root tc_public --sequence-length 10 --search-iterations 10 --external-monitor
python -m alphasyn run-search --dataset-root tc_public --sequence-length 10 --search-iterations 10 --cpuct 1.0
python -m alphasyn run-search --dataset-root benchmarks/VTR --design bfly.abc.blif --sequence-length 10 --search-iterations 20
python -m alphasyn run-search --dataset-root benchmarks/VTR --sequence-length 10 --search-iterations 30 --cpuct 1.0
python -m alphasyn run-search --dataset-root benchmarks/VTR --design bfly.abc.blif --debug-search
```

将 JSON 结果汇总为 CSV：

```bash
python -m alphasyn summarize
```

常用参数：

- `--dataset-root benchmarks/VTR`：选择要扫描的具体文件夹
- `--design bfly.abc.blif`：选择该文件夹下某一个精确的 `.blif` 文件名；省略时会运行该文件夹树中的全部文件
- `--debug-search`：将每一步根节点动作的 `Q/R/U/visits` 保存到 `steps[*].action_debug`，将每次迭代的节点轨迹保存到 `steps[*].iteration_traces`，并在 JSON 同目录输出调试 CSV 文件
- `--abc-bin /home/hxy/abc/abc`：显式指定 `ABC` 路径；如果 `abc` 已在 `PATH` 中可省略
- `--sequence-length 24`：搜索序列长度
- `--search-iterations 64`：每一步的 MCTS 迭代次数
- `--cpuct 1.0`：`U` 项的探索强度
- `--mu-discount 0.9`：回传时使用的长期折扣
- `--seed 0`：随机种子
- `--actions balance,rewrite,rewrite-z,refactor,refactor-z,resub,resub-z`：覆盖默认搜索动作空间；额外支持的标签包括 `fraig`、`fx`、`mfs`、`dsd`、`dch`、`extract` 和 `collapse`
- `--external-monitor`：将每个 design 作为受外部监控的子进程运行，并把 `runtime` / `peak_memory` 回填到结果 JSON 中
- `--workdir .alphasyn_work`：输出目录；必须位于当前项目目录内

输出：

- 默认输出目录为 `.alphasyn_work/`
- `results/*.json` 存储详细搜索结果
- 启用 `--debug-search` 时，`results/*.debug.csv` 按 `(step, action)` 存储每一行
- 启用 `--debug-search` 时，`results/*.trace.csv` 按 `(step, iteration, node, action)` 存储每一行
- `cache/` 会按每次 `run-search` 调用分别命名空间隔离，并在该次运行结束后自动删除
- `summary.csv` 为每个 design 存储一行，用于对比 `resyn2` 启发式结果与 MCTS 结果

## `hybridsyn`（UCB1 预热启动 + MCTS）

`hybridsyn` 会先执行逐步式的 `UCB1` 预热启动，以确定前期前缀，然后把该前缀交给 `alphasyn` 风格的 MCTS 核心来完成剩余搜索步骤。

准备数据集清单：

```bash
python -m hybridsyn prepare-data
python -m hybridsyn prepare-data --dataset-root benchmarks/VTR
```

运行混合搜索：

```bash
python -m hybridsyn run-search --abc-bin /path/to/abc --sequence-length 24 --warmup-steps 4 --search-iterations 64
python -m hybridsyn run-search --dataset-root tc_public --design tc_public_1/input.blif --sequence-length 10 --warmup-steps 3 --warmup-episodes 20 --search-iterations 10
python -m hybridsyn run-search --dataset-root tc_public --sequence-length 10 --warmup-steps 3 --warmup-episodes 20 --search-iterations 10 --external-monitor
python -m hybridsyn run-search --dataset-root tc_public --sequence-length 10 --warmup-steps 4 --search-iterations 30 --ucb-c 0.4 --cpuct 1.0
python -m hybridsyn run-search --dataset-root benchmarks/VTR --sequence-length 10 --warmup-steps 4 --warmup-episodes 20 --search-iterations 30 --warmup-top-k 3
python -m hybridsyn run-search --dataset-root benchmarks/VTR --design bfly.abc.blif --sequence-length 10 --warmup-steps 4 --search-iterations 20 --debug-search
```

将 JSON 结果汇总为 CSV：

```bash
python -m hybridsyn summarize
```

常用参数：

- `--dataset-root benchmarks/VTR`：选择要扫描的具体文件夹
- `--design bfly.abc.blif`：选择该文件夹下某一个精确的 `.blif` 文件名；省略时会运行该文件夹树中的全部文件
- `--abc-bin /home/hxy/abc/abc`：显式指定 `ABC` 路径；如果 `abc` 已在 `PATH` 中可省略
- `--sequence-length 24`：总搜索序列长度
- `--warmup-steps 4`：由 UCB1 负责的前期步骤数；超过序列长度的值会被截断
- `--warmup-episodes 64`：UCB1 预热阶段的 episode 数；若省略则默认等于 `--search-iterations`
- `--ucb-c 0.4`：UCB1 预热阶段的探索强度
- `--search-iterations 64`：预热之后每一步的 MCTS 迭代次数
- `--cpuct 1.0`：MCTS 中 `U` 项的探索强度
- `--mu-discount 0.9`：MCTS 回传时使用的长期折扣
- `--seed 0`：随机种子
- `--actions balance,rewrite,rewrite-z,refactor,refactor-z,resub,resub-z`：覆盖默认搜索动作空间；额外支持的标签包括 `fraig`、`fx`、`mfs`、`dsd`、`dch`、`extract` 和 `collapse`
- `--external-monitor`：将每个 design 作为受外部监控的子进程运行，并把 `runtime` / `peak_memory` 回填到结果 JSON 中
- `--debug-search`：输出预热阶段 CSV 轨迹，以及 MCTS 调试 CSV 和迭代轨迹 CSV
- `--workdir .hybridsyn_work`：输出目录；必须位于当前项目目录内

输出：

- 默认输出目录为 `.hybridsyn_work/`
- `results/*.json` 存储详细的混合搜索结果
- 启用 `--debug-search` 时，`results/*.warmup.csv` 为 UCB1 预热阶段按 `(episode, step, action)` 存储每一行
- 启用 `--debug-search` 时，`results/*.debug.csv` 为 MCTS 阶段按 `(step, action)` 存储每一行
- 启用 `--debug-search` 时，`results/*.trace.csv` 为 MCTS 阶段按 `(step, iteration, node, action)` 存储每一行
- `summary.csv` 为每个 design 存储一行，包含混合变体标签以及最终 `and` / `lev` 指标

## `SASyn`（模拟退火）

`SASyn` 保留原始模拟退火行为：对完整序列进行搜索，最终归一化代价为
`0.7 * (and / initial_and) + 0.3 * (lev / initial_lev)`.

准备数据集清单：

```bash
python -m SASyn prepare-data
python -m SASyn prepare-data --dataset-root benchmarks/VTR
```

运行模拟退火搜索：

```bash
python -m SASyn run-search --dataset-root tc_public --design tc_public_1/input.blif --sequence-length 10 --search-iterations 1000
python -m SASyn run-search --dataset-root tc_public --sequence-length 10 --search-iterations 1000 --external-monitor
python -m SASyn run-search --dataset-root tc_public --sequence-length 10 --search-iterations 1000
python -m SASyn run-search --dataset-root benchmarks/VTR --design bfly.abc.blif --search-iterations 2000 --debug-search
```

将 JSON 结果汇总为 CSV：

```bash
python -m SASyn summarize
```

常用参数：

- `--dataset-root`：包含 `.blif` 文件的根目录
- `--design`：运行 dataset root 下某一个精确 design；省略时会运行所有发现到的 design
- `--sequence-length`：模拟退火使用的固定序列长度
- `--search-iterations`：总退火迭代次数
- `--and-weight`：最终代价中归一化 AND 数的权重
- `--lev-weight`：最终代价中归一化层级数的权重
- `--initial-temperature`：覆盖自动校准得到的初始温度
- `--min-temperature`：几何降温调度的下界
- `--seed`：随机种子
- `--actions balance,rewrite,rewrite-z,refactor,refactor-z,resub,resub-z`：覆盖默认搜索动作空间；额外支持的标签包括 `fraig`、`fx`、`mfs`、`dsd`、`dch`、`extract` 和 `collapse`
- `--external-monitor`：将每个 design 作为受外部监控的子进程运行，并把 `runtime` / `peak_memory` 回填到结果 JSON 中
- `--debug-search`：输出逐迭代 CSV 轨迹
- `--workdir .sasyn_work`：输出目录；必须位于当前项目目录内

输出：

- 默认输出目录为 `.sasyn_work/`
- `manifest.json`：数据集清单
- `base_aig/`：由输入 BLIF 文件生成的、按 design 区分的临时 AIG 起始点
- `results/*.json`：按 design 存储的搜索结果
- 启用 `--debug-search` 时，`results/*.debug.csv`：逐迭代轨迹
- `summary.csv`：汇总报告

每个 SA 结果 JSON 都包含：以 `ABC` 命令字符串表示的最终序列、最终 `and` 与 `lev`、从初始 design 和 `resyn2` 推导出的基线信息，以及位于 `iterations` 下的完整退火轨迹。

## `MABSyn`（Bandit 基线方法）

`MABSyn` 保留了三种 bandit 风格的基线方法：

- `baseline_mab`：UCB1 序列搜索（一个 bandit）
- `baseline_mab_prefix`：带前缀评估 AIG 缓存的 UCB1（每一步一个 bandit）
- `linucb`：基于 LinUCB 的上下文 bandit 搜索

它们现在都通过子命令调用，并默认将生成文件存放在 `.mabsyn_work/` 下。

运行搜索：

```bash
python -m MABSyn.baseline_mab run-search --dataset-root tc_public --design tc_public_1/input.blif --episodes 20 --steps 10 --ucb-c 1.5 --debug-search
python -m MABSyn.baseline_mab run-search --dataset-root tc_public --episodes 20 --steps 10 --ucb-c 1.5 --external-monitor
python -m MABSyn.baseline_mab run-search --dataset-root benchmarks/VTR --design bfly.abc.blif --episodes 20 --steps 10
python -m MABSyn.baseline_mab run-search --dataset-root tc_public --episodes 10 --steps 10
python -m MABSyn.linucb run-search --dataset-root tc_public --design tc_public_1/input.blif --episodes 20 --steps 10
python -m MABSyn.linucb run-search --dataset-root tc_public --episodes 20 --steps 10 --external-monitor
python -m MABSyn.linucb run-search --dataset-root benchmarks/VTR --design bfly.abc.blif --episodes 20 --steps 10
python -m MABSyn.linucb run-search --dataset-root tc_public --episodes 20 --steps 10
python -m MABSyn.baseline_mab_prefix run-search --dataset-root tc_public --design tc_public_1/input.blif --episodes 20 --steps 10
python -m MABSyn.baseline_mab_prefix run-search --dataset-root tc_public --episodes 20 --steps 10 --external-monitor
python -m MABSyn.baseline_mab_prefix run-search --dataset-root benchmarks/VTR --design bfly.abc.blif --episodes 20 --steps 10
```

汇总：

```bash
python -m MABSyn.baseline_mab summarize
python -m MABSyn.linucb summarize
python -m MABSyn.baseline_mab_prefix summarize
```

常用参数：

- `--workdir .mabsyn_work`：结果 JSON、汇总文件和缓存的根目录
- `--abc-bin`：显式指定 `ABC` 路径
- `--dataset-root`：包含 `.blif` 文件的根目录
- `--design`：运行 dataset root 下某一个精确 design；省略时会运行所有发现到的 design
- `--steps`：固定序列长度，或每个 episode 的步数
- `--episodes`：该方法使用的总 episode / 迭代次数
- `--seed`：随机种子
- `--actions`：以逗号分隔的动作覆盖列表
- `--result-json`：可选的显式汇总输出路径
- `--external-monitor`：将每个 design 作为受外部监控的子进程运行，并把 `runtime` / `peak_memory` 回填到结果 JSON 中
- `--debug-search`：为 `baseline_mab` 启用详细的逐 episode 调试产物

各方法特有参数保持不变，包括 `baseline_mab` 的 `--ucb-c`，以及 `linucb` 的 `--alpha`、`--linucb-alpha`、`--lambda`、`--linucb-lambda`、`--long-term-rollouts` 和 `--long-term-horizon`。

输出：

- 默认根目录为 `.mabsyn_work/`
- `baseline_mab` 和 `linucb` 不再写出默认汇总 `*_results.json` 文件；如果需要请传入 `--result-json`
- 按 benchmark 划分的结果存储在 `.mabsyn_work/results/baseline_mab/` 和 `.mabsyn_work/results/linucb/` 下
- 启用 `--debug-search` 时，`baseline_mab` 会在结果 JSON 同目录写出 `.debug.csv` 和 `.debug.json`
- 汇总 CSV 会写入 `.mabsyn_work/results/summary.csv` 或各方法对应的汇总路径
- LinUCB 缓存文件会写入 `.mabsyn_work/cache/linucb/`

## 测试

该仓库使用标准库 `unittest` 测试套件：

```bash
python -m unittest discover -s tests -v
```

## 外部监控

当前所有运行时间和峰值内存统计都应来自外部监控器，而不是内部的 ABC 插桩。

- 使用 `run-search --external-monitor` 启用受监控执行
- 在批处理模式下，CLI 会自动为每个 design 启动一个子进程，从而让每个 design 都获得独立的外部测量 `runtime_sec` 和 `peak_memory_kb`
- 在 `run-search` 之后，运行已有的 `summarize` 命令来生成 `summary.csv`

## 超参数调优

对于 `alphasyn`、`baseline_mab` 和 `baseline_mab_prefix`，仓库提供了一个网格搜索调优驱动脚本：

```bash
python scripts/grid_search_tuning.py \
  --algorithms alphasyn baseline_mab baseline_mab_prefix \
  --dataset-root tc_public \
  --design tc_public_1/input.blif \
  --design tc_public_2/input.blif \
  --max-trials-per-algorithm 3
```

行为：

- 每个参数组合都会在各自隔离的 workdir 中运行
- `run-search` 会带上 `--external-monitor` 调用
- 脚本会自动运行各算法对应的 `summarize`
- 每次 trial 只记录它自己的 `summary.csv`
- 当所有 trial 结束后，会使用 `compare_algorithm_summaries.py` 对所有已选算法的全部 trial summary 统一重新评分
- 结果会写入 `.grid_search/`

默认输出：

- `.grid_search/alphasyn/results.csv`
- `.grid_search/baseline_mab/results.csv`
- `.grid_search/baseline_mab_prefix/results.csv`
- `.grid_search/all_results.csv`
- `.grid_search/global_compare/aggregate.csv`
- `.grid_search/global_compare/details.csv`

你可以使用 JSON 文件覆盖默认网格：

```bash
python scripts/grid_search_tuning.py --grid-config your_grid.json
```

示例网格文件：

```json
{
  "alphasyn": {
    "sequence-length": [8, 10],
    "search-iterations": [10, 20],
    "cpuct": [0.5, 1.0],
    "mu-discount": [0.8, 0.9],
    "seed": [0]
  },
  "baseline_mab": {
    "steps": [8, 10],
    "episodes": [10, 20],
    "ucb-c": [0.4, 1.0],
    "seed": [0]
  },
  "baseline_mab_prefix": {
    "steps": [8, 10],
    "episodes": [10, 20],
    "ucb-c": [0.2, 0.4],
    "seed": [0]
  }
}
```

## 动作评估

仓库提供了三个脚本，用于评估候选 `ABC` 命令是否值得加入动作空间。

内置候选标签包括：

- `fraig`
- `fx` 映射到 `renode; sop; fx; strash`
- `mfs` 映射到 `renode; mfs; strash`
- `dsd` 映射到 `dsd; strash`
- `dch` 映射到 `dch; strash`
- `extract`
- `extract -a`
- `collapse` 映射到 `collapse; strash`

### 第 1 步：静态单命令评估

在每个 design 上执行 `strash` 之后，对每个候选命令各运行一次；然后使用相同的四指标排序规则，对所有候选命令以及 `baseline` 空操作进行比较：

```bash
python scripts/evaluate_actions_static.py \
  --dataset-root tc_public \
  --design tc_public_1/input.blif \
  --design tc_public_2/input.blif
```

输出：

- `.action_eval/static/raw_details.csv`
- `.action_eval/static/summaries/*.csv`
- `.action_eval/static/compare/aggregate.csv`
- `.action_eval/static/compare/details.csv`

### 第 2 步：中间状态单步评估

从已有搜索结果 JSON 文件中采样中间前缀，重建这些状态，并从每个采样状态出发，对一个额外命令进行基准测试：

```bash
python scripts/evaluate_actions_prefixes.py \
  --result-path .alphasyn_work/results \
  --max-prefixes-per-design 10
```

输出：

- `.action_eval/prefixes/raw_details.csv`
- `.action_eval/prefixes/summaries/*.csv`
- `.action_eval/prefixes/compare/aggregate.csv`
- `.action_eval/prefixes/compare/details.csv`

### 第 3 步：完整搜索消融实验

针对多个算法和动作预设运行完整的 `run-search + summarize` 实验，然后把所有实验统一做全局比较：

```bash
python scripts/evaluate_actions_search_ablation.py \
  --algorithms alphasyn baseline_mab baseline_mab_prefix \
  --dataset-root tc_public \
  --config-json scripts/action_ablation_config.example.json \
  --action-label fraig
```

默认会创建以下预设：

- `baseline`
- `plus_fraig`
- `plus_fx`
- `plus_mfs`
- `plus_dsd`
- `plus_dch`
- `plus_extract`
- `plus_extract-a`
- `plus_collapse`

你还可以提供：

- `--action-label ...`：只测试候选标签的一个子集
- `--preset-json your_presets.json`：完全控制预设定义
- `--config-json your_algo_args.json`：为各算法设置 `run-search` 参数
- `--include-combined-preset`：添加一个包含所有已选候选标签的预设

示例 `--config-json`：

```json
{
  "alphasyn": {
    "sequence-length": 10,
    "search-iterations": 20,
    "cpuct": 1.0,
    "mu-discount": 0.9,
    "seed": 0
  },
  "baseline_mab": {
    "steps": 10,
    "episodes": 20,
    "ucb-c": 1.0,
    "seed": 0
  },
  "baseline_mab_prefix": {
    "steps": 10,
    "episodes": 20,
    "ucb-c": 0.4,
    "seed": 0
  }
}
```

输出：

- `.action_eval/search_ablation/<algorithm>/trials.csv`
- `.action_eval/search_ablation/<algorithm>/failures.csv`
- `.action_eval/search_ablation/<algorithm>/results.csv`
- `.action_eval/search_ablation/all_results.csv`
- `.action_eval/search_ablation/failures.csv`
- `.action_eval/search_ablation/global_compare/aggregate.csv`
- `.action_eval/search_ablation/global_compare/details.csv`

复用行为：

- 如果某个预设的 workdir 中已经包含预期的 `summary.csv`，第 3 步会直接复用它，并跳过该预设的 `run-search` 和 `summarize`
- 如果想强制重跑某个预设，请删除 `.action_eval/search_ablation/<algorithm>/` 下对应的预设目录

失败处理行为：

- 第 3 步会为每个预设一次只运行一个 design
- 如果某个预设下有一个 design 崩溃，失败会被记录到 `failures.csv` 中，后续 design 会继续运行
- 只要至少有一个 design 成功并且能够生成 `summary.csv`，该预设仍会参与最终比较

## 跨算法比较

```bash
python compare_algorithm_summaries.py \
  --output your_aggregate.csv \
  --details-output your_details.csv
```
