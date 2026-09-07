# FSFP Peer Workflow

本文档记录 `FSFP_peer` 目录下的同行分组与同行 FSFP 计算流程。

## 目录结构

```text
FSFP_peer/
├── similarity_matrices/
│   └── similarity_matrix_YYYY.pkl
├── data/
│   └── business_communities_all_years.csv
├── fsfp_weekly_filled_0509.csv
├── 分组.py
├── peer_fsfp_weekly.py
└── peer_fsfp_weekly_0509.csv
```

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
3. 只使用矩阵非对角线上三角，避免 pkl 中对角线为 `1` 形成自环。
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

当前结果概况：

- 覆盖年份：`2001-2024`
- 总记录数：`47,872`
- 年度社区总数：`11,916`

## 2. 周度同行 FSFP 计算

脚本：

```powershell
python .\peer_fsfp_weekly.py
```

输入：

- `fsfp_weekly_filled_0509.csv`
- `data/business_communities_all_years.csv`
- `similarity_matrices/similarity_matrix_YYYY.pkl`

重要说明：

`fsfp_weekly_filled_0509.csv` 中有两个 FSFP 相关字段：

- `fsfp_original`
- `FSFP`

当前流程应使用 `fsfp_original` 作为同行 FSFP 的计算输入。脚本会在读取数据后把 `fsfp_original` 标准化为内部使用的 `FSFP` 列，后续计算函数继续复用原有字段名。

处理逻辑：

1. 读取周度 FSFP 数据。
2. 将 `code` 标准化为 `Stkcd`，将 `week` 标准化为 `Week`。
3. 使用 `fsfp_original` 覆盖内部计算列 `FSFP`。
4. 读取年度社区分组结果。
5. 读取年度相似度矩阵。
6. 对每一年、每一周、每个社区分别计算加权同行 FSFP：

```text
Peer_FSFP_i = sum(similarity_ij * FSFP_j) / sum(similarity_ij)
```

7. 如果社区内只有一家公司，则同行 FSFP 等于自身 `fsfp_original`。
8. 如果某年份没有相似度矩阵，则该年份全部回退为自身 `fsfp_original`。
9. 对没有被社区计算覆盖的记录，也回退为自身 `fsfp_original`。

输出：

```text
peer_fsfp_weekly_0509.csv
```

输出字段：

- `Stkcd`
- `Week`
- `Peer_FSFP`

当前结果概况：

- 输入行数：`2,203,058`
- 输出行数：`2,203,058`
- 公司数：`5,312`
- 周数：`551`
- 输入覆盖年份：`2015-2025`
- 相似度矩阵覆盖年份：`2001-2024`
- `2025` 年无相似度矩阵，已回退为自身 `fsfp_original`
- `Peer_FSFP` 均值：`1.8738`
- `Peer_FSFP` 标准差：`7.1285`

## 依赖

主要 Python 依赖：

```powershell
python -m pip install pandas numpy tqdm networkx
```

本机执行时曾遇到 `pip` 代理问题，已通过手动下载并本地安装 `networkx 3.1` 解决。

## 推荐执行顺序

```powershell
cd C:\Users\chenyu\Desktop\财务造假\代码汇总\data\FSFP_peer

python .\分组.py
python .\peer_fsfp_weekly.py
```

执行完成后重点检查：

1. `data/business_communities_all_years.csv` 是否生成。
2. `peer_fsfp_weekly_0509.csv` 行数是否等于 `fsfp_weekly_filled_0509.csv`。
3. 日志中是否显示 `使用 fsfp_original 列作为FSFP计算输入`。
4. 若输入包含超过 `2024` 的年份，确认这些年份是否应继续回退为自身 `fsfp_original`。
