# FSFP LLM Peer Workflow

本文档记录 `FSFP_llm_peer` 目录下的年度同行社区分组与周度 LLM 修正 FSFP 同行因子计算流程。

## 目录结构

```text
FSFP_llm_peer/
├── similarity_matrices/
│   └── similarity_matrix_YYYY.pkl
├── data/
│   └── business_communities_all_years.csv
├── fsfp_weekly_corrected.csv
├── 分组.py
├── peer_fsfp_llm_weekly.py
├── peer_fsfp_llm_weekly_0509.csv
├── run_peer_fsfp_llm_weekly_0509.log
└── run_peer_fsfp_llm_weekly_0509.err
```

目录中仍保留 `peer_fsfp_weekly_0509.csv` 及旧运行日志，属于此前词频版/原脚本产物，不是当前 LLM peer 因子的主输出。

## 1. 年度同行社区分组

脚本：

```powershell
python .\分组.py
```

输入：

- `similarity_matrices/similarity_matrix_YYYY.pkl`
- 每个 pkl 文件包含：
  - `matrix`：公司间相似度矩阵
  - `companies`：矩阵行列对应的公司代码
  - `company_to_idx`：公司代码到矩阵下标的映射

处理逻辑：

1. 逐年读取 `similarity_matrix_YYYY.pkl`。
2. 使用相似度阈值 `0.5` 建立无向网络。
3. 只使用矩阵非对角线的上三角，避免 pkl 中对角线为 `1` 形成自环。
4. 使用 `networkx.greedy_modularity_communities` 做社区划分。
5. 输出每个公司在每年的社区 ID 和社区规模。

输出：

```text
data/business_communities_all_years.csv
```

输出字段：

- `Year`
- `Community_ID`
- `Symbol`
- `Community_Size`

当前分组结果概况：

- 覆盖年份：`2001-2024`
- 记录数：`47,872`
- 年度社区总数：`11,916`
- 覆盖公司数：`5,520`
- 相似度矩阵文件数：`24`

## 2. 周度 LLM 同行 FSFP 计算

脚本：

```powershell
python .\peer_fsfp_llm_weekly.py
```

输入：

- `fsfp_weekly_corrected.csv`
- `data/business_communities_all_years.csv`
- `similarity_matrices/similarity_matrix_YYYY.pkl`

`fsfp_weekly_corrected.csv` 的主要字段：

- `stock_code`
- `week_end`
- `fsfp`
- `n_total_news`
- `n_fraud_news`

脚本读取后会做字段标准化：

- `stock_code` 标准化为 `Stkcd`
- `week_end` 转换为 `Week`
- `fsfp` 标准化为内部计算使用的 `FSFP`

`Week` 的生成方式：

```python
week_end = pd.to_datetime(fsfp_df["week_end"], errors="raise")
fsfp_df["Week"] = week_end.dt.strftime("%Y%W")
```

这里使用日历年加周一制周号，保持与 LLM FSFP 生成脚本中 `week_end.year` 的年份口径一致。

处理逻辑：

1. 读取周度 LLM 修正 FSFP 数据。
2. 建立 `(year, week_num, Stkcd) -> FSFP` 的快速查找表。
3. 读取年度社区分组结果。
4. 读取年度相似度矩阵。
5. 对每一年、每一周、每个社区分别计算加权同行 FSFP：

```text
Peer_FSFP_i = sum(similarity_ij * FSFP_j) / sum(similarity_ij)
```

6. 如果社区内只有一家公司，则 `Peer_FSFP` 等于自身 `FSFP`。
7. 如果某一年没有相似度矩阵，则该年份全部回退为自身 `FSFP`。
8. 如果某条记录没有被社区计算覆盖，也回退为自身 `FSFP`。

输出：

```text
peer_fsfp_llm_weekly_0509.csv
```

输出字段：

- `Stkcd`
- `Week`
- `Peer_FSFP`

## 3. 本次执行记录

执行目录：

```powershell
cd C:\Users\chenyu\Desktop\财务造假\代码汇总\data\因子peer_weekly\FSFP_llm_peer
```

执行命令：

```powershell
python -u .\peer_fsfp_llm_weekly.py `
  1> .\run_peer_fsfp_llm_weekly_0509.log `
  2> .\run_peer_fsfp_llm_weekly_0509.err
```

执行时间：

- 完成时间：`2026-06-30 11:59`
- 错误日志：`run_peer_fsfp_llm_weekly_0509.err`，大小 `0` 字节
- 运行日志：`run_peer_fsfp_llm_weekly_0509.log`

日志尾部统计：

```text
结果已保存到 peer_fsfp_llm_weekly_0509.csv
原始FSFP统计: 均值=1.6077, 标准差=8.3578
同行FSFP统计: 均值=1.4783, 标准差=6.3636
```

当前结果校验：

- 输入文件：`fsfp_weekly_corrected.csv`
- 输出文件：`peer_fsfp_llm_weekly_0509.csv`
- 输入行数：`4,960,958`
- 输出行数：`4,960,958`
- 输入公司数：`5,134`
- 输出公司数：`5,134`
- 输入周数：`3,405`
- 输出周数：`3,405`
- 输入日期范围：`1960-10-31` 至 `2026-01-26`
- 输出 `Week` 范围：`196044` 至 `202604`
- 输入 `fsfp` 均值：`1.6077`
- 输入 `fsfp` 标准差：`8.3578`
- 输出 `Peer_FSFP` 均值：`1.4783`
- 输出 `Peer_FSFP` 标准差：`6.3636`

注意事项：

- 相似度矩阵覆盖 `2001-2024`。
- 输入数据覆盖 `1960-2026`。
- `1960-2000`、`2025-2026` 没有相似度矩阵，脚本已按逻辑回退为自身 `fsfp`。

## 4. 推荐执行顺序

如果需要从头重跑：

```powershell
cd C:\Users\chenyu\Desktop\财务造假\代码汇总\data\因子peer_weekly\FSFP_llm_peer

python .\分组.py
python -u .\peer_fsfp_llm_weekly.py `
  1> .\run_peer_fsfp_llm_weekly_0509.log `
  2> .\run_peer_fsfp_llm_weekly_0509.err
```

重跑完成后重点检查：

1. `data/business_communities_all_years.csv` 是否存在且覆盖 `2001-2024`。
2. `peer_fsfp_llm_weekly_0509.csv` 行数是否等于 `fsfp_weekly_corrected.csv`。
3. `run_peer_fsfp_llm_weekly_0509.err` 是否为空。
4. 日志中是否显示 `使用 fsfp 列作为FSFP计算输入`。
5. 年份超出 `2001-2024` 的记录是否接受回退为自身 `fsfp`。

## 5. 依赖

主要 Python 依赖：

```powershell
python -m pip install pandas numpy networkx
```
