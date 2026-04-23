# 结果文件说明

本文档用于说明当前仓库中会一并上传的结果文件各自表示什么，方便快速理解结果，而不需要先阅读全部代码。

## 建议阅读顺序

如果只想先看结论，建议按下面顺序查看：

1. `.compare_results/aggregate.csv`
2. `.compare_results/details.csv`
3. 各算法自己的 `summary.csv`
4. `.grid_search/` 下的调参汇总
5. `.action_eval/` 下的动作评估汇总

## 通用指标说明

大多数结果文件都会围绕以下几类指标展开：

- `and` / `final_and` / `initial_and`
  表示电路中的 AND 节点数，一般越小越好。
- `lev` / `final_lev` / `initial_lev`
  表示逻辑层级（depth / level），一般越小越好。
- `runtime_sec` / `total_runtime_sec`
  运行时间，一般越小越好。
- `peak_memory_kb`
  峰值内存，一般越小越好。
- `variant_label`
  当前结果对应的算法配置字符串，例如序列长度、搜索迭代次数、`cpuct`、`ucb_c`、随机种子等。
- `final_score`
  由比较脚本计算出的综合分。它不是单一物理量，而是基于多个指标的相对排名得分汇总，通常越高越好。

补充说明：

- 如果 `peak_memory_kb` 为 `-` 或 `null`，通常表示该次结果没有拿到外部监控内存值。
- `design_count` 表示该汇总实际纳入比较的设计数量。
- 当 `design_count` 小于预期值时，通常表示有部分设计失败或被跳过。

## 一、各算法主结果

### `.alphasyn_work/summary.csv`

这是 `alphasyn`（纯 MCTS）的主汇总表，每一行对应一个设计。

主要列含义：

- `design_name`: 设计名
- `variant_label`: 当前 MCTS 配置
- `initial_and`, `initial_lev`: 初始设计指标
- `heuristic_and`, `heuristic_lev`: `resyn2` 启发式基线结果
- `final_and`, `final_lev`: MCTS 最终结果
- `total_runtime_sec`, `peak_memory_kb`: 本设计的运行时间和峰值内存

适合回答的问题：

- MCTS 相比初始设计改进了多少
- MCTS 相比 `resyn2` 基线是否更好
- 单个设计的时间和内存开销如何

### `.hybridsyn_work/summary.csv`

这是 `hybridsyn`（UCB1 预热 + MCTS）的主汇总表，列结构与 `alphasyn` 基本一致。

`variant_label` 中额外会出现：

- `warmup`
- `warmup_episodes`
- `topk`
- `ucb_c`

适合与 `alphasyn` 直接横向对比，观察“先 UCB1 再 MCTS”是否能更快或更优。

### `.sasyn_work/summary.csv`

这是 `SASyn`（模拟退火）的主汇总表，列结构也与 `alphasyn` 基本一致。

`variant_label` 中常见参数包括：

- `seq`
- `iters`
- `min_temp`
- `seed`

适合比较模拟退火与 MCTS / bandit 方法在质量、速度上的差异。

### `.mabsyn_work/results/baseline_mab/summary.csv`

这是 `MABSyn` 中 `baseline_mab` 的汇总表。它的列更精简：

- `file`: 设计名
- `variant_label`: 配置
- `and`, `lev`: 最终结果
- `runtime_sec`, `peak_memory_kb`: 运行时间和峰值内存

这个文件不直接带 `initial_*` 或 `heuristic_*`，所以更适合做“最终结果对比”，而不是看单个设计从初始状态改进多少。

### `.mabsyn_work/results/baseline_mab_prefix/summary.csv`

这是 `baseline_mab_prefix` 的汇总表，结构和 `baseline_mab/summary.csv` 一致。

它可以与 `baseline_mab` 直接对比，用来看“前缀缓存”是否改善了结果质量或运行效率。

### `.mabsyn_work/results/baseline_mab/_summary.json`

### `.mabsyn_work/results/baseline_mab_prefix/_summary.json`

这两个 JSON 是 MAB 类方法更详细的结果摘要，适合在需要看单个设计内部信息时使用。

其中常见字段：

- `meta`: 运行环境、数据集路径、设计数量
- `results[]`: 每个设计的详细结果
- `initial.nodes`, `initial.level`: 初始指标
- `best.nodes`, `best.level`: 最优结果
- `best.recipe_str`: 搜索得到的命令序列
- `improvement`: 改进幅度
- `config`: 本次运行参数
- `summary`: 该 JSON 内所有设计的整体统计

如果想知道“最终最优序列具体是什么”，优先看这里面的 `recipe_str`。

## 二、跨算法总比较

### `.compare_results/aggregate.csv`

这是当前上传结果里最重要的总览文件。它把不同算法的汇总结果放在一起，做统一评分和排序。

主要列含义：

- `algorithm`: 算法名称
- `variant_label`: 参与比较的配置
- `design_count`: 参与评分的设计数
- `final_score`: 综合分，越高越好
- `summary_path`: 该算法原始汇总文件路径
- `avg_and`, `avg_lev`, `avg_runtime`, `avg_memory`: 各指标平均值
- `avg_*_rank`: 各指标平均排名
- `avg_*_score`: 各指标平均得分

适合回答的问题：

- 当前上传结果里，哪个算法整体最好
- 某个算法是“结果更好”还是“只是速度更快”
- 综合排序由哪些维度拉高或拉低

### `.compare_results/details.csv`

这是逐设计的详细比较表。每一行表示“某个设计下，某个算法”的得分情况。

主要列含义：

- `design`: 设计名
- `rank`: 该设计下该算法的最终名次
- `algorithm`, `variant_label`: 算法及参数
- `final_score`: 该设计下的综合分
- `and`, `lev`, `runtime`, `memory`: 该设计上的四个实际指标
- `*_rank`: 在该设计上单指标的名次
- `*_score`: 在该设计上单指标的得分
- `active_weight_sum`: 当前行实际参与计分的总权重

适合回答的问题：

- 某个设计上到底是谁赢了
- 某个算法是因为面积更好、层级更好，还是只是更快
- 为什么某个算法总体排名高，但在个别设计上表现一般

## 三、网格搜索调参结果

### `.grid_search/alphasyn/results.csv`

### `.grid_search/baseline_mab/results.csv`

### `.grid_search/baseline_mab_prefix/results.csv`

这三个文件分别是每个算法内部的调参排序结果。

主要列含义：

- `rank`: 在该算法内部的排名
- `trial_index`, `trial_name`: 第几个配置组合
- `summary_path`: 该 trial 对应的结果汇总文件
- `final_score`: 该 trial 的综合分
- `design_count`: 该 trial 实际参与比较的设计数量
- 其后各列如 `cpuct`、`ucb-c`、`search-iterations`、`steps`、`episodes`、`seed` 等：
  表示这个 trial 的具体超参数

适合回答的问题：

- 单个算法内部，哪组参数最好
- 某些超参数变化是否显著影响结果

### `.grid_search/all_results.csv`

这是所有已跑 trial 的总表，跨算法汇总，但仍保留 trial 级别信息。

适合回答的问题：

- 所有被尝试过的配置里，哪些 trial 值得重点看
- 某个最优 trial 对应的原始 `summary.csv` 在哪里

### `.grid_search/all_trials_ranked.csv`

这个文件与 `all_results.csv` 类似，但多了：

- `global_rank`: 所有算法、所有 trial 混在一起后的全局排名

因此它更适合直接回答“所有调参结果里，全局最强配置是哪一个”。

### `.grid_search/global_rescored/aggregate.csv`

### `.grid_search/global_rescored/details.csv`

这两个文件表示把所有 trial 放到同一个评价体系下重新打分后的结果。

可以理解为：

- `results.csv`: 更像“先跑出来，再记录各自结果”
- `global_rescored/*.csv`: 更像“把所有 trial 一起放到同一张考卷里重新排名”

如果要做正式汇报，优先看 `global_rescored/aggregate.csv`。

## 四、动作评估结果

动作评估对应的是“是否值得把某个 ABC 命令加入动作空间”的分析。

### `.action_eval/prefixes/raw_details.csv`

这是中间状态单步评估的最细粒度明细表。

每一行表示：

- 从某个已有搜索结果中取一个中间前缀状态
- 在这个状态上额外执行一个候选动作
- 记录执行后的指标变化

主要列含义：

- `sample_name`: 采样样本名
- `design_name`: 设计名
- `source_result`: 来源搜索结果文件
- `prefix`: 当前前缀动作序列
- `prefix_length`: 前缀长度
- `action_label`: 候选动作标签
- `abc_command`: 实际执行的 ABC 命令
- `state_and`, `state_lev`: 加动作前的状态指标
- `final_and`, `final_lev`: 加动作后的指标
- `delta_and`, `delta_lev`: 指标变化，负数通常表示改进
- `status`, `returncode`, `error`: 执行状态

适合回答的问题：

- 某个动作在中间状态上是否通常有帮助
- 某个动作是不是容易把结果变差

### `.action_eval/prefixes/summaries/*.csv`

每个文件对应一个候选动作或 `baseline`，是对 `raw_details.csv` 的聚合结果。

### `.action_eval/prefixes/aggregate.csv`

### `.action_eval/prefixes/compare/aggregate.csv`

### `.action_eval/prefixes/compare/details.csv`

这些文件是对单步动作评估做排序比较后的结果。

其中：

- `compare/aggregate.csv`：看各候选动作总体表现
- `compare/details.csv`：看每个样本上各动作的具体名次

如果只想快速判断“哪个候选动作值得加入动作空间”，优先看 `compare/aggregate.csv`。

### `.action_eval/search_ablation/alphasyn/trials.csv`

### `.action_eval/search_ablation/baseline_mab/trials.csv`

### `.action_eval/search_ablation/baseline_mab_prefix/trials.csv`

这些文件记录“完整搜索消融实验”里，每个算法测试了哪些动作预设。

主要列含义：

- `preset_label`: 预设名，如 `baseline`、`plus_fraig`、`plus_extract`
- `action_labels`: 预设中包含的动作标签
- `actions_arg`: 实际传给 CLI 的动作列表
- `summary_path`: 该预设对应的结果汇总

### `.action_eval/search_ablation/alphasyn/results.csv`

### `.action_eval/search_ablation/baseline_mab/results.csv`

### `.action_eval/search_ablation/baseline_mab_prefix/results.csv`

这些文件是各算法内部的动作预设排名结果。

主要列含义：

- `global_rank`: 在所有算法、所有预设中的全局名次
- `algorithm_rank`: 在当前算法内部的名次
- `preset_label`: 当前动作预设
- `final_score`: 综合分
- `design_count`: 成功参与比较的设计数

适合回答的问题：

- 对某个算法来说，加哪个动作最好
- 某个动作加入后是否明显提升结果

### `.action_eval/search_ablation/all_results.csv`

这是完整搜索消融实验的总汇总表，跨算法、跨动作预设汇总。

### `.action_eval/search_ablation/global_compare/aggregate.csv`

### `.action_eval/search_ablation/global_compare/details.csv`

这两个文件是完整搜索消融实验最重要的结果文件。

它们回答的是：

- 在完整搜索条件下，哪种“算法 + 动作预设”组合整体最好
- 动作扩展到底是否真的提升了最终搜索结果

### `.action_eval/search_ablation/failures.csv`

这个文件记录消融实验中的失败样本。

主要列含义：

- `algorithm`, `preset_label`, `design_name`: 哪个算法、哪个动作预设、哪个设计失败
- `returncode`: 退出码
- `command`: 执行命令
- `output_excerpt`: 错误摘要

当某个结果文件的 `design_count` 小于预期时，通常就需要来这里查原因。

## 五、如何快速解读当前上传结果

如果协作者只想快速把握当前结论，可以这样看：

1. 先看 `.compare_results/aggregate.csv`
   了解当前主实验里各算法的总体排名。
2. 再看 `.compare_results/details.csv`
   了解某个算法到底在哪些设计上更强。
3. 如果要看某个算法自己的详细结果，再回到对应 `summary.csv`。
4. 如果要看“参数怎么调出来的”，看 `.grid_search/`。
5. 如果要看“新增动作值不值得加”，看 `.action_eval/prefixes/compare/aggregate.csv` 和 `.action_eval/search_ablation/global_compare/aggregate.csv`。
6. 如果发现有设计缺失或分数异常，再看 `.action_eval/search_ablation/failures.csv`。

## 六、注意事项

- 上传的是“便于协作者阅读和比较”的关键结果，不是全部原始中间文件。
- 很多逐设计原始 JSON、缓存目录、trial 工作目录没有上传，是为了避免仓库体积过大。
- 如果后续需要复现实验过程，可根据各汇总文件中的 `summary_path`、`variant_label`、超参数列和 `actions_arg` 反推运行配置。
