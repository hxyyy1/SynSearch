# SynSearch

**Search-based Logic Synthesis with ABC**

SynSearch 是一个面向 BLIF 电路的逻辑综合序列搜索工具集。它通过 ABC 执行综合命令，用 MCTS、UCB1、LinUCB 和模拟退火探索命令序列，记录 AIG 节点数（`and`）、逻辑层级（`lev`）及搜索过程，支持在同一组 benchmark 上比较不同搜索方法。

## 搜索方法

| 方法 | Python 模块 | 搜索方式 |
| --- | --- | --- |
| AlphaSyn | `alphasyn` | 逐步 MCTS，支持树复用，无神经网络 |
| HybridSyn | `hybridsyn` | UCB1 确定前期前缀，MCTS 完成后续搜索 |
| SASyn | `SASyn` | 对完整命令序列进行模拟退火 |
| UCB1 | `MABSyn.baseline_mab` | 全局 Bandit 序列搜索 |
| UCB1 Prefix | `MABSyn.baseline_mab_prefix` | 按步骤使用 Bandit，缓存前缀 AIG |
| LinUCB | `MABSyn.linucb` | 使用电路状态特征的上下文 Bandit |

## 环境准备

- Python 3.10 或更高版本。
- 可执行的 ABC 逻辑综合工具，需支持 `read_blif`、`strash` 及所选综合命令。
- NumPy：LinUCB 和完整测试套件需要，其他搜索方法仅使用 Python 标准库。
- 外部资源监控使用 GNU `time`；建议在 Linux 上运行。

在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

# 替换为本机 ABC 可执行文件的绝对路径；若 abc 已在 PATH 中则可省略。
export ABC_BIN=/path/to/abc
"$ABC_BIN" -c 'version'
```

也可以在搜索命令中使用 `--abc-bin /path/to/abc`。以下命令均从仓库根目录运行。

## 电路数据集

所有搜索入口、网格搜索和静态动作评估默认递归扫描 `benchmarks/`，仅读取 `.blif` 文件。

| 目录 | BLIF 电路数量 | 说明 |
| --- | ---: | --- |
| `benchmarks/EPFL/` | 20 | EPFL 电路；同目录还保留 AIG、Verilog 和 VHDL 表示 |
| `benchmarks/VTR/` | 6 | VTR 电路，文件名形如 `bfly.abc.blif` |

使用 `--dataset-root benchmarks/EPFL` 或 `--dataset-root benchmarks/VTR` 可以选择单个子集。所有算法使用一致的电路名称：文件名唯一时使用文件名（如 `adder.blif`），重名时使用相对于数据集根目录的路径。省略 `--design` 会运行选定目录中的全部 BLIF 电路。

生成包含文件路径和 SHA-256 的数据清单，不需要 ABC：

```bash
python3 -m alphasyn prepare-data
# 其他支持该命令的入口：hybridsyn、SASyn
```

清单默认位于 `.alphasyn_work/manifest.json`。搜索命令也可以直接运行，无需先生成清单。

## 快速开始

在 EPFL 的 `adder.blif` 上运行一个小预算 MCTS 搜索，并汇总结果：

```bash
python3 -m alphasyn run-search \
  --design adder.blif \
  --sequence-length 4 \
  --search-iterations 4 \
  --seed 0 \
  --external-monitor

python3 -m alphasyn summarize
```

结果 JSON 位于 `.alphasyn_work/results/`，CSV 位于 `.alphasyn_work/summary.csv`。JSON 包含搜索序列、最终电路指标和搜索配置。上述预算用于验证流程，实际优化时可增加序列长度与迭代次数。

运行完整 VTR 子集：

```bash
python3 -m alphasyn run-search \
  --dataset-root benchmarks/VTR \
  --sequence-length 10 --search-iterations 20 --external-monitor
```

## 运行其他方法

下列示例使用相同的 `adder.blif`，方便生成可比较的结果。

```bash
# UCB1 预热 + MCTS
python3 -m hybridsyn run-search --design adder.blif \
  --sequence-length 10 --warmup-steps 3 --warmup-episodes 20 \
  --search-iterations 20 --seed 0 --external-monitor
python3 -m hybridsyn summarize

# 模拟退火
python3 -m SASyn run-search --design adder.blif \
  --sequence-length 10 --search-iterations 100 --seed 0 --external-monitor
python3 -m SASyn summarize

# 全局 UCB1
python3 -m MABSyn.baseline_mab run-search --design adder.blif \
  --steps 10 --episodes 20 --ucb-c 1.0 --seed 0 --external-monitor
python3 -m MABSyn.baseline_mab summarize

# 带前缀缓存的 UCB1
python3 -m MABSyn.baseline_mab_prefix run-search --design adder.blif \
  --steps 10 --episodes 20 --ucb-c 0.4 --seed 0 --external-monitor
python3 -m MABSyn.baseline_mab_prefix summarize

# 上下文 LinUCB
python3 -m MABSyn.linucb run-search --design adder.blif \
  --steps 10 --episodes 20 --seed 0 --external-monitor
python3 -m MABSyn.linucb summarize
```

### 参数与目标

- `--dataset-root`、`--design`、`--abc-bin`、`--seed`、`--workdir` 和 `--external-monitor` 适用于所有搜索入口。
- AlphaSyn / HybridSyn 的 `--search-iterations` 是每一步的 MCTS 迭代数；SASyn 中它是总退火迭代数。Bandit 方法使用 `--steps` 和 `--episodes`。不同方法的相同数值不代表相同计算预算。
- AlphaSyn 使用 `--cpuct` 控制探索强度、`--mu-discount` 控制长期回报折扣；HybridSyn 还提供 `--warmup-steps`、`--warmup-episodes`、`--warmup-top-k` 和 `--ucb-c`。
- SASyn 默认代价为 `0.7 × and/initial_and + 0.3 × lev/initial_lev`，可通过 `--and-weight`、`--lev-weight` 和温度参数调整。
- LinUCB 提供 `--alpha`、`--lambda`、`--long-term-rollouts` 和 `--long-term-horizon` 等参数。
- AlphaSyn、HybridSyn、SASyn 和全局 UCB1 支持 `--debug-search`，用于导出更详细的搜索轨迹。

默认动作包括 `balance`、`rewrite`、`rewrite -z`、`refactor`、`refactor -z`、`resub` 和 `resub -z`。自定义动作示例：

```bash
# AlphaSyn / HybridSyn / SASyn 使用动作标签
python3 -m alphasyn run-search --design adder.blif \
  --actions balance,rewrite,rewrite-z --sequence-length 4 --search-iterations 4

# MABSyn 的 --actions 接受 ABC 命令字符串
python3 -m MABSyn.baseline_mab run-search --design adder.blif \
  --actions 'balance,rewrite,rewrite -z' --steps 4 --episodes 4
```

完整参数以各模块的 `run-search --help` 为准。

## 结果与比较

| 方法 | 默认汇总 CSV |
| --- | --- |
| AlphaSyn | `.alphasyn_work/summary.csv` |
| HybridSyn | `.hybridsyn_work/summary.csv` |
| SASyn | `.sasyn_work/summary.csv` |
| UCB1 | `.mabsyn_work/results/baseline_mab/summary.csv` |
| UCB1 Prefix | `.mabsyn_work/results/baseline_mab_prefix/summary.csv` |
| LinUCB | `.mabsyn_work/results/linucb/summary.csv` |

`--external-monitor` 为每个电路启动独立子进程，并回填运行时间和峰值内存。对比时间与内存时，各方法应使用相同的监控方式。`and` 和 `lev` 是 AIG 结构指标，不是工艺映射后的面积或物理时延。

生成所需 CSV 后，可显式选择参与比较的方法：

```bash
python3 compare_algorithm_summaries.py \
  --summary AlphaSyn=.alphasyn_work/summary.csv \
  --summary UCB1=.mabsyn_work/results/baseline_mab/summary.csv \
  --summary LinUCB=.mabsyn_work/results/linucb/summary.csv \
  --print-report
```

不传 `--summary` 时，脚本读取 AlphaSyn、HybridSyn、SASyn、UCB1 和 UCB1 Prefix 的五个默认汇总文件。默认按电路比较至少两个具有完整有效指标的结果；`--missing-policy error` 可要求电路集合完全一致。输出写入 `.compare_results/aggregate.csv` 和 `.compare_results/details.csv`。

评分根据每个电路上各指标的排名加权计算，默认权重为 AND 数 0.5、层级 0.2、时间 0.2、内存 0.1；总分越高越好。比较时应保持电路集合、ABC 版本和运行环境一致，并记录预算与随机种子。不同种子的汇总选择规则随方法而异，建议使用独立 `--workdir` 保存每次实验。

所有默认结果目录、缓存、测试产物和 `abc.history` 均由 `.gitignore` 排除。自定义 `--workdir` 必须位于当前工作目录内；如需保留多次实验，可使用默认工作目录下的子目录，并在 `summarize` 时传入同一 `--workdir`。

## 调优与动作评估

网格搜索支持 AlphaSyn、UCB1 和 UCB1 Prefix。每个参数组合使用独立工作目录，自动执行搜索、外部监控、汇总和比较：

```bash
python3 scripts/grid_search_tuning.py \
  --dataset-root benchmarks/EPFL --design adder.blif \
  --max-trials-per-algorithm 3
```

结果位于 `.grid_search/`；`--grid-config` 可指定自定义 JSON 参数网格。

动作评估分为静态单步评估、已有搜索前缀评估和完整搜索消融：

```bash
python3 scripts/evaluate_actions_static.py \
  --dataset-root benchmarks/EPFL --design adder.blif --action-label fraig

python3 scripts/evaluate_actions_prefixes.py \
  --result-path .alphasyn_work/results --max-prefixes-per-design 5 --action-label fraig

python3 scripts/evaluate_actions_search_ablation.py \
  --dataset-root benchmarks/EPFL --design adder.blif \
  --config-json scripts/action_ablation_config.example.json --action-label fraig
```

结果位于 `.action_eval/`。完整搜索消融会复用输出目录中已存在的预设汇总；更换电路或参数后，应使用新的 `--output-root`。各脚本的 `--help` 列出完整选项。

## 开发与测试

```bash
python3 -m unittest discover -s tests -v
```

```text
alphasyn/                       MCTS 搜索与 ABC 后端
hybridsyn/                      UCB1 + MCTS 混合搜索
SASyn/                          模拟退火搜索
MABSyn/                         UCB1、前缀缓存 UCB1、LinUCB
benchmarks/EPFL/                EPFL 电路
benchmarks/VTR/                 VTR 电路
scripts/                       调优、动作评估和外部监控
compare_algorithm_summaries.py  跨算法比较
external_monitoring.py         CLI 外部监控封装
tests/                         自动化测试
```
