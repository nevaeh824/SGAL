"""
季度频率 PeerMeta / SG-AL 当期识别。

数据流：
1. 读取周频最终表，并为每只股票、每个季度保留最后一个可用周的整行；
2. 用该行的 143 个特征识别同一行、同一季度的 FSFP；
3. 每个窗口用连续 20 个季度训练，随后 1 个季度测试，并逐季滚动；
4. 第一层 17 个模型生成 5 折 OOF 训练特征和测试特征；
5. 第二层 17 个模型在完整 OOF 特征上各拟合一次；
6. 按第二层模型分歧度从低到高生成伪标签并反馈，直到测试池为空。

脚本刻意把季度抽样、特征契约、折内预处理、模型拟合和输出审计拆开，
便于逐段验证时间对齐，并避免测试季度统计量进入训练预处理。
"""

from __future__ import annotations

import gc
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from multiprocessing import get_context
from pathlib import Path

# 所有基础模型的内部数值线程固定为 1。模型级并行只由外层有界进程池控制，
# 避免 6 个模型进程再次各自创建一组 BLAS/OpenMP 线程。
for thread_env_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[thread_env_name] = "1"

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from sgal_model_registry import (
    REQUIRED_MODEL_NAMES,
    required_model_factories,
    validate_required_model_names,
)


# ============================================================
# 1. 固定配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "output" / "final_merged_0509_cts_cleaned_FSFP.csv"

STOCK_COL = "code"
WEEK_COL = "week"
SOURCE_TARGET_COL = "FSFP"
TARGET_COL = "FSFP_current"

# 五年固定为 20 个季度；每个 OOF 验证块包含连续 4 个季度。
TRAIN_WINDOW_QUARTERS = 20
OOF_FOLDS = 5
QUARTERS_PER_FOLD = TRAIN_WINDOW_QUARTERS // OOF_FOLDS
LABELS = [0, 1, 2]

# full 是正式运行；single_quarter_test 仅用于依赖和流程冒烟测试。
SG_AL_RUN_MODE = os.getenv("SG_AL_RUN_MODE", "full").strip().lower()
SG_AL_SINGLE_TEST_QUARTER = os.getenv(
    "SG_AL_SINGLE_TEST_QUARTER", ""
).strip().upper()
SG_AL_SINGLE_QUARTER_TRAIN_MAX_ROWS = int(
    os.getenv("SG_AL_SINGLE_QUARTER_TRAIN_MAX_ROWS", "5000")
)
SG_AL_RANDOM_STATE = 42
# 每轮固定按“初始测试池”的 10% 反馈，因此通常约 10 轮清空测试池。
SG_AL_CONSENSUS_FRACTION = 0.10
SG_AL_D_PER_ROUND = None
SG_AL_MAX_ITERATIONS = None
SG_AL_VERBOSE = True

# 正式入口默认逐模型串行；首窗口计时入口在导入后单独覆盖为 6。
# 此处不读取外部环境变量，避免终端中残留的配置改变正式运行方式。
MODEL_PARALLEL_JOBS = 1
MODEL_INNER_MAX_THREADS = 1
NCA_EXCLUSIVE_PROCESS = False
AVAILABLE_CPU_COUNT = os.cpu_count() or 1

if SG_AL_RUN_MODE not in {"single_quarter_test", "full"}:
    raise ValueError(
        "SG_AL_RUN_MODE 只能是 single_quarter_test 或 full。"
    )
if TRAIN_WINDOW_QUARTERS % OOF_FOLDS != 0:
    raise ValueError(
        f"{TRAIN_WINDOW_QUARTERS} 个训练季度必须能被 "
        f"{OOF_FOLDS} 个 OOF 折整除。"
    )
if MODEL_PARALLEL_JOBS < 1:
    raise ValueError("MODEL_PARALLEL_JOBS 必须是大于等于 1 的整数。")
if MODEL_PARALLEL_JOBS > AVAILABLE_CPU_COUNT:
    raise ValueError(
        "MODEL_PARALLEL_JOBS 不能超过当前可用逻辑处理器数 "
        f"{AVAILABLE_CPU_COUNT}。"
    )

# 新版 Windows 常不再提供 wmic；显式给 loky 可用 CPU 上限可避免其
# 物理核心探测警告。真正同时运行的进程数仍由 MODEL_PARALLEL_JOBS 控制。
os.environ.setdefault(
    "LOKY_MAX_CPU_COUNT",
    str(AVAILABLE_CPU_COUNT),
)

RUN_OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / (
        "sg_al_current_quarter_single_quarter_model17_nca_full"
        if SG_AL_RUN_MODE == "single_quarter_test"
        else "sg_al_current_quarter_quarterly_full_model17_nca_full"
    )
)

FEATURE_GROUP_SPECS = [
    ("公司财务因子", "fin__", 13),
    ("公司治理因子", "gov__", 4),
    ("公司市场因子", "mkt__", 8),
    ("公司情感因子", "sent__", 3),
    ("同行财务因子", "peer_fin__", 43),
    ("同行市场因子", "peer_mkt__", 27),
    ("同行治理因子", "peer_gov__", 35),
    ("同行情感因子", "peer_sent__", 9),
]


# ============================================================
# 2. 周频输入转季度末周面板
# ============================================================

def clean_stock_code(series):
    """提取股票代码中的数字并统一为 6 位，避免 000001 被读成 1。"""
    code = series.astype(str).str.extract(r"(\d+)", expand=False)
    if code.isna().any():
        raise ValueError(f"{STOCK_COL} 中存在无法识别的股票代码。")
    return code.str.zfill(6)


def clean_week(series):
    """把周编码规范成 YYYYWW；ISO 周是否合法由 iso_week_start 再校验。"""
    week = (
        series.astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
        .str.zfill(6)
    )
    if not week.str.fullmatch(r"\d{6}", na=False).all():
        raise ValueError(f"{WEEK_COL} 必须是 6 位 YYYYWW。")
    return week


def iso_week_start(week_value):
    """返回 ISO 周的周一；date.fromisocalendar 会拒绝非法年份/周号组合。"""
    week_text = str(week_value).zfill(6)
    return pd.Timestamp(
        date.fromisocalendar(
            int(week_text[:4]),
            int(week_text[4:6]),
            1,
        )
    )


def validate_target(series):
    """允许标签缺失，但所有非空标签必须严格属于整数集合 {0, 1, 2}。"""
    numeric = pd.to_numeric(series, errors="coerce")
    provided = series.notna() & series.astype(str).str.strip().ne("")
    unparseable = provided & numeric.isna()
    if unparseable.any():
        examples = (
            series.loc[unparseable]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"{SOURCE_TARGET_COL} 含无法解析的非空标签：{examples}"
        )

    non_finite = numeric.notna() & ~np.isfinite(numeric)
    if non_finite.any():
        raise ValueError(
            f"{SOURCE_TARGET_COL} 含无穷值，共 {int(non_finite.sum())} 行。"
        )

    non_integer = numeric.notna() & ~np.isclose(numeric, np.round(numeric))
    if non_integer.any():
        raise ValueError(
            f"{SOURCE_TARGET_COL} 含非整数标签，共 {int(non_integer.sum())} 行。"
        )

    observed = sorted(numeric.dropna().astype(int).unique().tolist())
    unexpected = sorted(set(observed) - set(LABELS))
    if unexpected:
        raise ValueError(
            f"{SOURCE_TARGET_COL} 含 0/1/2 之外的标签：{unexpected}"
        )
    return numeric, observed


def collapse_weekly_to_quarterly(weekly_df):
    """
    每只股票、每个季度只保留最后一个可用周的整行。

    YYYYWW 的季度划分固定为：
    Q1=01-13，Q2=14-26，Q3=27-39，Q4=40-53。
    因而特征和标签必定来自同一个季度末周记录。这里不做季度均值、
    求和或分别选取标签；整行保留可以避免特征与标签来自不同周。

    性能上只对分组键计算 idxmax，再抽取约 14 万条季度行；不再排序
    174 万行的整张 149 列宽表，从而显著降低峰值内存。
    """
    required = {STOCK_COL, WEEK_COL, SOURCE_TARGET_COL}
    missing = sorted(required - set(weekly_df.columns))
    if missing:
        raise ValueError(f"输入文件缺少必要字段：{missing}")

    df = weekly_df
    # read_csv 生成唯一 RangeIndex；仅在外部调用传入重复索引时才重建，
    # 保证下方 groupby.idxmax 返回的标签能唯一定位原始行。
    if not df.index.is_unique:
        df = df.reset_index(drop=True)

    df[STOCK_COL] = clean_stock_code(df[STOCK_COL])
    df[WEEK_COL] = clean_week(df[WEEK_COL])

    duplicate_count = int(
        df.duplicated([STOCK_COL, WEEK_COL], keep="first").sum()
    )
    if duplicate_count:
        raise ValueError(
            f"输入存在 {duplicate_count} 个重复 (code, week) 键。"
        )

    df["year"] = df[WEEK_COL].str[:4].astype(int)
    df["week_num"] = df[WEEK_COL].str[4:6].astype(int)
    if not df["week_num"].between(1, 53).all():
        bad = sorted(df.loc[
            ~df["week_num"].between(1, 53), WEEK_COL
        ].unique().tolist())
        raise ValueError(f"存在周号不在 1-53 的记录：{bad[:20]}")

    # ISO 日期转换只对约 522 个唯一周执行，而不是逐行调用 Python 函数。
    week_start_map = {
        week: iso_week_start(week)
        for week in df[WEEK_COL].drop_duplicates()
    }
    df["week_start"] = df[WEEK_COL].map(week_start_map)
    df["week_order"] = df["year"] * 100 + df["week_num"]

    df["quarter_num"] = ((df["week_num"] - 1) // 13 + 1).clip(upper=4)
    df["quarter"] = (
        df["year"].astype(str)
        + "Q"
        + df["quarter_num"].astype(str)
    )
    df["quarter_period"] = pd.PeriodIndex(df["quarter"], freq="Q")

    group_keys = [STOCK_COL, "quarter_period"]
    quarterly_groups = df.groupby(
        group_keys,
        sort=False,
        observed=True,
    )
    last_row_indices = quarterly_groups["week_order"].idxmax()
    expected_last_week = quarterly_groups["week_order"].max().sort_index()

    # 只排序抽取后的季度面板，后续窗口和输出因此仍具有确定的行序。
    quarterly = (
        df.loc[last_row_indices]
        .sort_values(["quarter_period", STOCK_COL])
        .reset_index(drop=True)
        .copy()
    )

    selected_last_week = quarterly.set_index(
        group_keys
    )["week_order"].sort_index()
    if not selected_last_week.equals(expected_last_week):
        raise RuntimeError("季度抽样未严格选中每只股票当季最后一周。")

    quarterly[TARGET_COL], observed_labels = validate_target(
        quarterly[SOURCE_TARGET_COL]
    )
    quarterly["target_quarter"] = quarterly["quarter"]
    quarterly["target_quarter_period"] = quarterly["quarter_period"]
    quarterly["target_week"] = quarterly[WEEK_COL]

    same_quarter = quarterly["target_quarter"].eq(quarterly["quarter"])
    target_alignment_audit = pd.DataFrame([{
        "frequency": "quarterly",
        "quarter_definition": (
            "Q1=W01-W13;Q2=W14-W26;Q3=W27-W39;Q4=W40-W53"
        ),
        "quarter_value_rule": "last_available_week_row_per_stock_quarter",
        "alignment_mode": (
            "same_quarter_last_week_features_to_same_quarter_last_week_fsfp"
        ),
        "weekly_input_rows": int(len(df)),
        "quarterly_rows": int(len(quarterly)),
        "quarterly_stock_count": int(quarterly[STOCK_COL].nunique()),
        "quarter_count": int(quarterly["quarter_period"].nunique()),
        "rows_with_target": int(quarterly[TARGET_COL].notna().sum()),
        "rows_missing_target": int(quarterly[TARGET_COL].isna().sum()),
        "same_quarter_alignment_rate": float(same_quarter.mean()),
        "observed_labels": ",".join(map(str, observed_labels)),
    }])

    quarter_sampling_audit = (
        quarterly.groupby(
            ["quarter_period", "quarter"],
            as_index=False,
            observed=True,
        )
        .agg(
            row_count=(STOCK_COL, "size"),
            stock_count=(STOCK_COL, "nunique"),
            earliest_selected_week=(WEEK_COL, "min"),
            latest_selected_week=(WEEK_COL, "max"),
            min_selected_week_num=("week_num", "min"),
            max_selected_week_num=("week_num", "max"),
        )
        .sort_values("quarter_period")
        .reset_index(drop=True)
    )
    quarter_sampling_audit["quarter_period"] = (
        quarter_sampling_audit["quarter_period"].astype(str)
    )

    target_alignment_by_quarter = (
        quarterly.assign(has_target=quarterly[TARGET_COL].notna())
        .groupby("quarter", as_index=False, observed=True)
        .agg(
            row_count=(STOCK_COL, "size"),
            target_rows=("has_target", "sum"),
        )
    )
    target_alignment_by_quarter["missing_target_rows"] = (
        target_alignment_by_quarter["row_count"]
        - target_alignment_by_quarter["target_rows"]
    )

    return {
        "quarterly_df": quarterly,
        "target_alignment_audit": target_alignment_audit,
        "target_alignment_by_quarter": target_alignment_by_quarter,
        "quarter_sampling_audit": quarter_sampling_audit,
    }


def load_quarterly_panel():
    """一次性读取周频宽表，并立即压缩成后续建模使用的季度面板。"""
    print(f"读取周频输入：{INPUT_FILE}")
    weekly_df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
        encoding="utf-8",
        dtype={STOCK_COL: str, WEEK_COL: str},
    )
    input_shape = tuple(weekly_df.shape)
    result = collapse_weekly_to_quarterly(weekly_df)
    del weekly_df
    gc.collect()

    quarterly_df = result["quarterly_df"]
    print(
        "季度化完成："
        f"{input_shape[0]:,} 周频行 -> {len(quarterly_df):,} 季度行；"
        f"{quarterly_df[STOCK_COL].nunique():,} 只股票；"
        f"{quarterly_df['quarter_period'].nunique()} 个季度。"
    )
    result["weekly_input_shape"] = input_shape
    return result


# ============================================================
# 3. 特征契约
# ============================================================

def get_feature_group(feature_name):
    """按唯一前缀把字段映射到论文定义的特征组。"""
    if feature_name == "Peer_FSFP":
        return "同行 FSFP"
    for group_name, prefix, _ in FEATURE_GROUP_SPECS:
        if feature_name.startswith(prefix):
            return group_name
    return None


def build_feature_contract(df):
    """
    固定模型输入字段并阻止未知列静默进入训练。

    只有 8 类前缀特征和 Peer_FSFP 可以进入模型；标签、时间索引以及
    历史遗留的 FSFP_next 均明确排除。分组数量和总数同时校验，避免
    上游改列名、少合并字段或意外增加字段后仍继续训练。
    """
    exclude_cols = {
        STOCK_COL,
        WEEK_COL,
        SOURCE_TARGET_COL,
        TARGET_COL,
        "FSFP_next",
        "Year",
        "year",
        "week_num",
        "week_order",
        "week_start",
        "quarter_num",
        "quarter",
        "quarter_period",
        "target_week",
        "target_quarter",
        "target_quarter_period",
        "oof_fold",
    }

    feature_group_by_col = {}
    unexpected = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        group_name = get_feature_group(col)
        if group_name is None:
            unexpected.append(col)
        else:
            feature_group_by_col[col] = group_name

    feature_cols = [
        col for col in df.columns
        if col in feature_group_by_col
    ]
    if unexpected:
        raise ValueError(f"发现未归类的候选特征列：{unexpected}")

    expected_counts = {
        group_name: expected
        for group_name, _, expected in FEATURE_GROUP_SPECS
    }
    expected_counts["同行 FSFP"] = 1

    group_count_df = pd.DataFrame([
        {
            "feature_group": group_name,
            "expected_count": expected,
            "actual_count": sum(
                feature_group_by_col.get(col) == group_name
                for col in feature_cols
            ),
        }
        for group_name, expected in expected_counts.items()
    ])
    group_count_df["matches_expected"] = (
        group_count_df["expected_count"]
        == group_count_df["actual_count"]
    )
    if not group_count_df["matches_expected"].all():
        raise ValueError(
            "特征分组数量与数据契约不一致：\n"
            + group_count_df.to_string(index=False)
        )
    if len(feature_cols) != 143:
        raise ValueError(f"特征总数应为 143，当前为 {len(feature_cols)}。")

    availability_rows = []
    for col in feature_cols:
        group_name = feature_group_by_col[col]
        if group_name in {"公司财务因子", "公司治理因子"}:
            status, risk = "unverified", "high"
        elif group_name in {"公司市场因子", "公司情感因子"}:
            status, risk = "conditional", "medium"
        elif group_name.startswith("同行"):
            status, risk = "unverified", "high"
        else:
            status, risk = "unverified", "medium"
        availability_rows.append({
            "feature": col,
            "feature_group": group_name,
            "availability_status": status,
            "leakage_risk": risk,
            "quarter_value_rule": "current_quarter_last_available_week",
            "included_in_model": True,
        })

    return {
        "feature_cols": feature_cols,
        "feature_group_by_col": feature_group_by_col,
        "feature_group_count_df": group_count_df,
        "feature_availability_audit_df": pd.DataFrame(availability_rows),
    }


def coerce_features_to_numeric(df, feature_cols):
    """
    在滚动开始前把 143 个特征统一转成数值类型。

    原代码在每个重叠的 20 季度窗口、甚至每个 OOF 预处理调用中重复
    pd.to_numeric。同一条历史记录会被转换多次。季度面板确定后只转换
    一次即可；无法解析的值仍变为 NaN，随后由严格折内规则填补，因此
    统计口径不变，也不会用到测试季度的填补统计量。

    同时返回逐列审计，记录有多少“原本非空”的值因无法解析而变成 NaN。
    该函数有意原位替换特征列，以避免再复制一份约 14 万 × 143 的面板。
    """
    non_missing_before = df.loc[:, feature_cols].notna().sum()
    numeric_features = df.loc[:, feature_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )
    non_missing_after = numeric_features.notna().sum()
    df[feature_cols] = numeric_features

    non_numeric = [
        col
        for col in feature_cols
        if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise TypeError(f"特征转数值后仍存在非数值列：{non_numeric}")

    coercion_audit = pd.DataFrame({
        "feature": feature_cols,
        "non_missing_before": non_missing_before.reindex(
            feature_cols
        ).to_numpy(),
        "non_missing_after": non_missing_after.reindex(
            feature_cols
        ).to_numpy(),
    })
    coercion_audit["coerced_to_missing"] = (
        coercion_audit["non_missing_before"]
        - coercion_audit["non_missing_after"]
    )
    return df, coercion_audit


# ============================================================
# 4. 二十季度训练、单季度测试的滚动窗口
# ============================================================

def parse_quarter(quarter_text):
    """解析 YYYYQn 环境变量，并把格式错误转换为可读异常。"""
    try:
        return pd.Period(quarter_text, freq="Q")
    except Exception as exc:
        raise ValueError(
            f"季度必须写成 YYYYQ1-YYYYQ4，当前为 {quarter_text!r}。"
        ) from exc


def build_rolling_windows(df_model):
    """
    构造固定长度、逐季度滚动的窗口定义。

    测试季度 t 只有在 t-20 至 t-1 的 20 个季度全部存在时才可用。
    因而窗口天然满足：
    2011Q1-2015Q4 -> 2016Q1，
    下一窗口 2011Q2-2016Q1 -> 2016Q2。
    """
    available_quarters = sorted(
        df_model["quarter_period"].drop_duplicates().tolist()
    )
    available_set = set(available_quarters)
    candidate_test_quarters = []

    for test_quarter in available_quarters:
        train_start = test_quarter - TRAIN_WINDOW_QUARTERS
        train_quarters = list(
            pd.period_range(
                start=train_start,
                periods=TRAIN_WINDOW_QUARTERS,
                freq="Q",
            )
        )
        if all(quarter in available_set for quarter in train_quarters):
            candidate_test_quarters.append(test_quarter)

    if SG_AL_RUN_MODE == "single_quarter_test":
        if SG_AL_SINGLE_TEST_QUARTER:
            requested = parse_quarter(SG_AL_SINGLE_TEST_QUARTER)
            if requested not in candidate_test_quarters:
                raise ValueError(
                    f"指定季度 {requested} 不存在，或此前没有连续 "
                    f"{TRAIN_WINDOW_QUARTERS} 个训练季度。"
                )
            candidate_test_quarters = [requested]
        else:
            candidate_test_quarters = candidate_test_quarters[:1]

    if not candidate_test_quarters:
        raise ValueError("没有可用的季度滚动窗口。")

    windows = []
    for window_id, test_quarter in enumerate(
        candidate_test_quarters,
        start=1,
    ):
        train_start = test_quarter - TRAIN_WINDOW_QUARTERS
        train_end = test_quarter - 1
        train_quarters = list(
            pd.period_range(
                start=train_start,
                end=train_end,
                freq="Q",
            )
        )
        windows.append({
            "window_id": window_id,
            "train_start_quarter": str(train_start),
            "train_end_quarter": str(train_end),
            "test_quarter": str(test_quarter),
            "target_quarter": str(test_quarter),
            "train_start_period": train_start,
            "train_end_period": train_end,
            "test_period_obj": test_quarter,
            "train_quarters": train_quarters,
            "train_period": f"{train_start}-{train_end}",
            "test_period": str(test_quarter),
            "prediction_asof": str(test_quarter),
            "alignment_mode": (
                "same_quarter_last_week_features_to_same_quarter_last_week_fsfp"
            ),
            "train_start_year": int(train_start.year),
            "train_end_year": int(train_end.year),
            "test_start_year": int(test_quarter.year),
            "test_end_year": int(test_quarter.year),
        })

    return windows


def assign_quarter_folds(train_df, train_quarters):
    """
    把训练季度按时间顺序切成 OOF_FOLDS 个连续验证块。

    同季度的所有股票必须落入同一折，避免同一时点的截面样本同时出现在
    拟合集和验证集中。每折模型使用其余季度拟合；这不是随机 K 折，也不是
    只允许过去数据的 expanding-window 验证。
    """
    if len(train_quarters) != TRAIN_WINDOW_QUARTERS:
        raise ValueError(
            f"训练窗口必须包含 {TRAIN_WINDOW_QUARTERS} 个季度。"
        )
    quarter_to_fold = {
        quarter: position // QUARTERS_PER_FOLD
        for position, quarter in enumerate(train_quarters)
    }
    result = train_df.copy()
    result["oof_fold"] = result["quarter_period"].map(quarter_to_fold)
    if result["oof_fold"].isna().any():
        raise RuntimeError("训练样本中存在无法分配 OOF 折的季度。")
    result["oof_fold"] = result["oof_fold"].astype(int)
    if sorted(result["oof_fold"].unique().tolist()) != list(range(OOF_FOLDS)):
        raise ValueError("季度训练样本没有完整覆盖 5 个时间 OOF 折。")
    return result


def stratified_training_cap(train_df, max_rows):
    """
    仅为单季度冒烟测试压缩训练集。

    按 OOF 折 × 标签分层分配配额，至少保留每个已存在分组的一条记录；
    正式 full 模式完全不调用该抽样。
    """
    if max_rows is None or max_rows <= 0 or len(train_df) <= max_rows:
        return train_df

    groups = list(
        train_df.groupby(
            ["oof_fold", TARGET_COL],
            sort=True,
            observed=True,
        )
    )
    if max_rows < len(groups):
        raise ValueError(
            f"训练样本上限 {max_rows} 小于折×标签分组数 {len(groups)}。"
        )

    sizes = np.asarray([len(group) for _, group in groups], dtype=int)
    raw_quotas = max_rows * sizes / sizes.sum()
    quotas = np.maximum(1, np.floor(raw_quotas).astype(int))
    quotas = np.minimum(quotas, sizes)

    while quotas.sum() > max_rows:
        reducible = np.where(quotas > 1)[0]
        if len(reducible) == 0:
            raise RuntimeError("无法在保留每个折×标签组的同时压缩训练集。")
        index = reducible[np.argmax(quotas[reducible])]
        quotas[index] -= 1

    fractional = raw_quotas - np.floor(raw_quotas)
    while quotas.sum() < max_rows:
        addable = np.where(quotas < sizes)[0]
        if len(addable) == 0:
            break
        index = addable[np.argmax(fractional[addable])]
        quotas[index] += 1
        fractional[index] = -1

    rng = np.random.default_rng(SG_AL_RANDOM_STATE)
    selected_indices = []
    for quota, (_, group) in zip(quotas, groups):
        selected_indices.extend(
            rng.choice(
                group.index.to_numpy(),
                size=int(quota),
                replace=False,
            ).tolist()
        )

    return (
        train_df.loc[selected_indices]
        .sort_values(["quarter_period", STOCK_COL])
        .reset_index(drop=True)
    )


def prepare_window_data(df_model, feature_cols, window):
    """
    从季度面板切出一个 20 季度训练集和紧随其后的单季度测试集。

    此处只做行选择和元数据拆分。特征已经在滚动前统一转为数值；缺失值
    仍原样保留，必须等到第一层每个 OOF 拟合时才能使用对应训练子集填补。
    """
    train_mask = (
        (df_model["quarter_period"] >= window["train_start_period"])
        & (df_model["quarter_period"] <= window["train_end_period"])
    )
    test_mask = df_model["quarter_period"].eq(window["test_period_obj"])

    train_df = df_model.loc[train_mask].copy()
    test_df = df_model.loc[test_mask].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"窗口 {window['window_id']} 的训练集或测试集为空。"
        )
    if train_df["quarter_period"].nunique() != TRAIN_WINDOW_QUARTERS:
        raise ValueError(
            f"窗口 {window['window_id']} 未覆盖完整 "
            f"{TRAIN_WINDOW_QUARTERS} 个训练季度。"
        )

    train_df = assign_quarter_folds(
        train_df,
        window["train_quarters"],
    )
    test_df["oof_fold"] = OOF_FOLDS - 1
    rows_before_cap = len(train_df)

    if SG_AL_RUN_MODE == "single_quarter_test":
        train_df = stratified_training_cap(
            train_df,
            SG_AL_SINGLE_QUARTER_TRAIN_MAX_ROWS,
        )

    # df_model 在季度化后已按 quarter_period、code 全局排序；布尔筛选、
    # 折号映射和 dropna 都保持行序。单季度抽样函数也会自行恢复该顺序，
    # 因此这里不再为每个高度重叠的窗口重复排序整张宽表。
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    meta_cols = [
        STOCK_COL,
        WEEK_COL,
        "week_start",
        "week_num",
        "quarter",
        "quarter_num",
        "quarter_period",
        "target_week",
        "target_quarter",
        "oof_fold",
    ]

    window_info = {
        key: value
        for key, value in window.items()
        if key != "train_quarters"
        and key not in {
            "train_start_period",
            "train_end_period",
            "test_period_obj",
        }
    }
    window_info.update({
        "train_quarter_count": int(
            train_df["quarter_period"].nunique()
        ),
        "train_rows_before_cap": int(rows_before_cap),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_stock_count": int(train_df[STOCK_COL].nunique()),
        "test_stock_count": int(test_df[STOCK_COL].nunique()),
        "oof_fold_count": int(train_df["oof_fold"].nunique()),
        "quarters_per_oof_fold": QUARTERS_PER_FOLD,
        "is_capped_trial_sample": (
            SG_AL_RUN_MODE == "single_quarter_test"
        ),
        "feature_count": int(len(feature_cols)),
    })

    return {
        "window_info": window_info,
        "train_X_raw": (
            train_df[feature_cols]
            .reset_index(drop=True)
        ),
        "test_X_raw": (
            test_df[feature_cols]
            .reset_index(drop=True)
        ),
        "y_train": (
            train_df[TARGET_COL].astype(int).reset_index(drop=True)
        ),
        "y_test": (
            test_df[TARGET_COL].astype(int).reset_index(drop=True)
        ),
        "train_meta": train_df[meta_cols].reset_index(drop=True),
        "test_meta": test_df[meta_cols].reset_index(drop=True),
    }


# ============================================================
# 5. 折内预处理、17 模型和评价指标
# ============================================================

MODEL_FACTORIES = required_model_factories()
MODEL_NAMES = validate_required_model_names(REQUIRED_MODEL_NAMES)


def fit_fold_preprocessor(X_fit, fit_stocks, X_apply, apply_stocks):
    """
    只用当前拟合子集建立“公司中位数 + 全局中位数 + 标准化”参数。

    X_apply 可以是 OOF 验证块，也可以是测试季度。公司在拟合子集中出现
    时优先使用该公司的历史中位数；新公司或公司中位数仍缺失时退回全局
    中位数。均值和标准差同样只来自 X_fit，防止验证折或测试季度泄漏。

    输入已由 coerce_features_to_numeric 统一转为数值，因此这里不再在每个
    折、每次 SG-AL 迭代重复执行字符串解析。
    """
    X_fit = pd.DataFrame(X_fit).reset_index(drop=True)
    X_apply = pd.DataFrame(X_apply).reset_index(drop=True)
    fit_stocks = pd.Series(fit_stocks).reset_index(drop=True)
    apply_stocks = pd.Series(apply_stocks).reset_index(drop=True)

    # groupby 只使用拟合样本。同一股票跨季度的历史记录共同给出公司基准。
    company_medians = X_fit.groupby(fit_stocks).median()
    global_medians = X_fit.median().fillna(0.0)

    # reindex 按每一行的股票代码广播对应公司中位数；测试期新股票会得到
    # 全 NaN 行，随后自然退回 global_medians。
    fit_company_fill = company_medians.reindex(
        fit_stocks
    ).reset_index(drop=True)
    apply_company_fill = company_medians.reindex(
        apply_stocks
    ).reset_index(drop=True)
    fit_company_fill.columns = X_fit.columns
    apply_company_fill.columns = X_apply.columns

    X_fit_imputed = (
        X_fit.fillna(fit_company_fill).fillna(global_medians)
    )
    X_apply_imputed = (
        X_apply.fillna(apply_company_fill).fillna(global_medians)
    )

    # 标准差为 0 的常量列除数改为 1，使其标准化结果为 0，而不是 NaN。
    means = X_fit_imputed.mean()
    stds = X_fit_imputed.std(ddof=0).replace(0, 1).fillna(1)
    return (
        (X_fit_imputed - means) / stds,
        (X_apply_imputed - means) / stds,
    )


def wall_time_text():
    """返回便于终端定位的本地墙钟时间。"""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


def available_physical_memory_gib():
    """在 Windows 返回当前可用物理内存；查询失败时返回 None。"""
    if os.name != "nt":
        return None
    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(
            ctypes.byref(status)
        ):
            return None
        return status.ullAvailPhys / (1024 ** 3)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def choose_effective_parallel_jobs(requested_jobs):
    """
    根据批次开始时的可用物理内存下调普通模型并发数。

    NCA 不进入该进程池；这里的阈值只用于避免普通模型在用户同时运行其他
    高内存程序时继续固定启动 6 个进程。
    """
    requested_jobs = max(1, min(int(requested_jobs), AVAILABLE_CPU_COUNT))
    available_gib = available_physical_memory_gib()
    if available_gib is None or requested_jobs == 1:
        return requested_jobs, available_gib
    if available_gib >= 16:
        memory_cap = 6
    elif available_gib >= 12:
        memory_cap = 4
    elif available_gib >= 8:
        memory_cap = 2
    else:
        memory_cap = 1
    return min(requested_jobs, memory_cap), available_gib


def shutdown_reusable_loky_workers():
    """
    强制结束当前批次的 loky 工作进程。

    joblib 默认会保留空闲工作进程供下一次 Parallel 复用；这里必须主动结束，
    才能保证下一折或下一层的 NCA 启动前没有普通模型子进程继续驻留。
    """
    try:
        from joblib.externals.loky import reusable_executor

        executor = reusable_executor._executor
        if executor is not None:
            executor.shutdown(wait=True, kill_workers=True)
    except Exception as exc:
        print(
            f"[{wall_time_text()}] loky 工作进程清理警告 | "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


def safe_model_fit_predict(
    model_name,
    X_fit,
    y_fit,
    X_predict,
    scheduler_mode="serial",
    requested_parallel_jobs=1,
    effective_parallel_jobs=1,
    model_position=None,
    model_count=None,
):
    """
    创建一个全新模型实例，完成一次拟合和预测，并把异常转成结构化日志。

    每次都从工厂新建实例，防止不同 OOF 折或 SG-AL 轮次共享已拟合状态。
    不在单个模型结束后强制 gc.collect；局部引用退出函数即释放，集中在
    折、迭代和窗口边界回收可避免上万次无必要的全量垃圾回收。
    """
    started_at = time.perf_counter()
    diagnostics = {
        "fit_rows": int(len(X_fit)),
        "predict_rows": int(len(X_predict)),
        "feature_count": int(pd.DataFrame(X_fit).shape[1]),
        "model_parallel_jobs": int(requested_parallel_jobs),
        "effective_model_parallel_jobs": int(effective_parallel_jobs),
        "model_inner_max_threads": MODEL_INNER_MAX_THREADS,
        "scheduler_mode": scheduler_mode,
        "model_position": model_position,
        "model_count": model_count,
        "worker_pid": os.getpid(),
        "started_at": wall_time_text(),
    }
    try:
        model = MODEL_FACTORIES[model_name]()
        model.fit(X_fit, y_fit)
        prediction = np.asarray(model.predict(X_predict)).reshape(-1)
        if len(prediction) != len(X_predict):
            raise ValueError(
                "模型预测长度与待预测样本数不一致："
                f"{len(prediction)} != {len(X_predict)}"
            )
        prediction = prediction.astype(int)
        if not np.isin(prediction, LABELS).all():
            raise ValueError("模型输出包含 0/1/2 之外的类别。")
        diagnostics["elapsed_seconds"] = (
            time.perf_counter() - started_at
        )
        diagnostics["finished_at"] = wall_time_text()
        return prediction, "success", "", diagnostics
    except Exception as exc:
        diagnostics["elapsed_seconds"] = (
            time.perf_counter() - started_at
        )
        diagnostics["finished_at"] = wall_time_text()
        return (
            None,
            "failed",
            f"{type(exc).__name__}: {exc}",
            diagnostics,
        )


def execute_model_task(
    model_name,
    X_fit,
    y_fit,
    X_predict,
    stage_label,
    scheduler_mode,
    requested_parallel_jobs,
    effective_parallel_jobs,
    model_position,
    model_count,
):
    """由实际执行模型的进程打印开始位置，再返回带模型名的结构化结果。"""
    print(
        f"[{wall_time_text()}] [{stage_label}] 模型开始 | "
        f"模型 {model_position}/{model_count} {model_name} | "
        f"调度={scheduler_mode} | pid={os.getpid()} | "
        f"fit={len(X_fit):,}, predict={len(X_predict):,}",
        flush=True,
    )
    result = safe_model_fit_predict(
        model_name,
        X_fit,
        y_fit,
        X_predict,
        scheduler_mode=scheduler_mode,
        requested_parallel_jobs=requested_parallel_jobs,
        effective_parallel_jobs=effective_parallel_jobs,
        model_position=model_position,
        model_count=model_count,
    )
    return model_name, result


def print_model_result(
    stage_label,
    model_name,
    result,
    completed_count,
    model_count,
    stage_started_at,
):
    """模型返回后立即打印成功/失败、样本规模和训练预测总耗时。"""
    _, status, error_message, diagnostics = result
    elapsed_seconds = float(diagnostics["elapsed_seconds"])
    elapsed_minutes = elapsed_seconds / 60
    stage_elapsed_minutes = (
        time.perf_counter() - stage_started_at
    ) / 60
    location = (
        f"模型 {diagnostics.get('model_position')}/{model_count} "
        f"{model_name} | 阶段完成 {completed_count}/{model_count}"
    )
    if status == "success":
        print(
            f"[{wall_time_text()}] [{stage_label}] 模型成功 | "
            f"{location} | "
            f"fit={diagnostics['fit_rows']:,}, "
            f"predict={diagnostics['predict_rows']:,} | "
            f"耗时 {elapsed_seconds:.2f} 秒 "
            f"({elapsed_minutes:.2f} 分钟) | "
            f"阶段累计 {stage_elapsed_minutes:.2f} 分钟 | "
            f"pid={diagnostics.get('worker_pid')}",
            flush=True,
        )
    else:
        print(
            f"[{wall_time_text()}] [{stage_label}] 模型失败 | "
            f"{location} | "
            f"耗时 {elapsed_seconds:.2f} 秒 "
            f"({elapsed_minutes:.2f} 分钟) | "
            f"阶段累计 {stage_elapsed_minutes:.2f} 分钟 | "
            f"{error_message}",
            flush=True,
        )


def build_failed_process_result(
    model_name,
    X_fit,
    X_predict,
    started_at,
    error,
    scheduler_mode,
    effective_parallel_jobs,
    model_position,
    model_count,
):
    """把独占子进程崩溃转换为与普通模型一致的失败结果。"""
    diagnostics = {
        "fit_rows": int(len(X_fit)),
        "predict_rows": int(len(X_predict)),
        "feature_count": int(pd.DataFrame(X_fit).shape[1]),
        "model_parallel_jobs": MODEL_PARALLEL_JOBS,
        "effective_model_parallel_jobs": int(effective_parallel_jobs),
        "model_inner_max_threads": MODEL_INNER_MAX_THREADS,
        "scheduler_mode": scheduler_mode,
        "model_position": model_position,
        "model_count": model_count,
        "worker_pid": None,
        "started_at": None,
        "finished_at": wall_time_text(),
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    return (
        None,
        "failed",
        f"{type(error).__name__}: {error}",
        diagnostics,
    )


def run_required_model_batch(
    X_fit,
    y_fit,
    X_predict,
    stage_label,
):
    """
    拟合固定 17 个模型，并始终按注册表顺序返回结果。

    正式脚本的默认串行模式保持原顺序和失败即停。首窗口安全并行模式把
    NCA 放入独立 spawn 子进程并等待其完全退出，再用最多 6 个 loky 进程
    运行其余模型；OOF 折和 SG-AL 迭代仍由调用方串行执行。普通模型按预计
    耗时从长到短提交，但最终结果重新按 MODEL_NAMES 排列，不改变 stacking
    列契约。
    """
    stage_started_at = time.perf_counter()
    X_fit_array = np.ascontiguousarray(
        pd.DataFrame(X_fit).to_numpy(dtype=float, copy=False)
    )
    X_predict_array = np.ascontiguousarray(
        pd.DataFrame(X_predict).to_numpy(dtype=float, copy=False)
    )
    y_fit_array = np.asarray(y_fit, dtype=int).reshape(-1)

    if len(X_fit_array) != len(y_fit_array):
        raise ValueError("模型批次的 X_fit 和 y_fit 长度不一致。")

    model_count = len(MODEL_NAMES)
    model_positions = {
        model_name: position
        for position, model_name in enumerate(MODEL_NAMES, start=1)
    }
    print(
        f"[{wall_time_text()}] [{stage_label}] 阶段开始 | "
        f"fit={len(X_fit_array):,}, predict={len(X_predict_array):,}, "
        f"features={X_fit_array.shape[1]:,} | "
        f"请求普通模型并发={MODEL_PARALLEL_JOBS} | "
        f"NCA独占={MODEL_PARALLEL_JOBS > 1 or NCA_EXCLUSIVE_PROCESS}",
        flush=True,
    )

    use_resource_groups = (
        MODEL_PARALLEL_JOBS > 1 or NCA_EXCLUSIVE_PROCESS
    )
    if not use_resource_groups:
        results_by_name = {}
        completed_count = 0
        for model_name in MODEL_NAMES:
            returned_name, result = execute_model_task(
                model_name,
                X_fit_array,
                y_fit_array,
                X_predict_array,
                stage_label,
                "serial",
                MODEL_PARALLEL_JOBS,
                1,
                model_positions[model_name],
                model_count,
            )
            results_by_name[returned_name] = result
            completed_count += 1
            print_model_result(
                stage_label,
                returned_name,
                result,
                completed_count,
                model_count,
                stage_started_at,
            )

            # 固定 17 模型策略不允许忽略失败模型。串行时立即返回，
            # 让上层保存日志并终止当前折，不再浪费时间运行后续模型。
            if result[1] != "success":
                break
        print(
            f"[{wall_time_text()}] [{stage_label}] 阶段结束 | "
            f"完成={len(results_by_name)}/{model_count} | "
            f"状态={'success' if len(results_by_name) == model_count else 'failed'} | "
            f"阶段耗时={(time.perf_counter() - stage_started_at) / 60:.2f} 分钟",
            flush=True,
        )
        return [
            (model_name, results_by_name[model_name])
            for model_name in MODEL_NAMES
            if model_name in results_by_name
        ]

    results_by_name = {}
    completed_count = 0
    nca_position = model_positions["NCA"]
    nca_core_gib = (
        len(X_fit_array) ** 2 * (4 * 8 + 1) / (1024 ** 3)
    )
    nca_available_gib = available_physical_memory_gib()
    memory_text = (
        "当前可用物理内存=未知"
        if nca_available_gib is None
        else f"当前可用物理内存={nca_available_gib:.2f} GiB"
    )
    warning_text = ""
    if (
        nca_available_gib is not None
        and nca_core_gib + 4 > nca_available_gib
    ):
        warning_text = " | 警告：NCA核心矩阵加运行余量超过当前可用内存"
    print(
        f"[{wall_time_text()}] [{stage_label}] NCA独占屏障 | "
        f"核心矩阵粗估={nca_core_gib:.2f} GiB | {memory_text}"
        f"{warning_text}",
        flush=True,
    )

    nca_process_started_at = time.perf_counter()
    try:
        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=get_context("spawn"),
        ) as executor:
            returned_name, nca_result = executor.submit(
                execute_model_task,
                "NCA",
                X_fit_array,
                y_fit_array,
                X_predict_array,
                stage_label,
                "exclusive_spawn",
                MODEL_PARALLEL_JOBS,
                1,
                nca_position,
                model_count,
            ).result()
    except Exception as exc:
        returned_name = "NCA"
        nca_result = build_failed_process_result(
            "NCA",
            X_fit_array,
            X_predict_array,
            nca_process_started_at,
            exc,
            "exclusive_spawn",
            1,
            nca_position,
            model_count,
        )

    results_by_name[returned_name] = nca_result
    completed_count += 1
    print_model_result(
        stage_label,
        returned_name,
        nca_result,
        completed_count,
        model_count,
        stage_started_at,
    )
    gc.collect()
    if nca_result[1] != "success":
        print(
            f"[{wall_time_text()}] [{stage_label}] 阶段中止 | "
            "NCA 独占任务失败，普通模型队列未启动 | "
            f"阶段耗时={(time.perf_counter() - stage_started_at) / 60:.2f} 分钟",
            flush=True,
        )
        return [("NCA", nca_result)]

    normal_execution_order = [
        "SVM-Lin",
        "SVM-Poly",
        "SVM-RBF",
        "FNN3",
        "FNN2",
        "FNN1",
        "Probit",
        "Logit",
        "ITML",
        "KNN",
        "LDA",
        "QDA",
        "NaiveBayes",
        "Tree-ID3",
        "Tree-C4.5",
        "Tree-CART",
    ]
    if (
        len(normal_execution_order) != model_count - 1
        or set(normal_execution_order) != set(MODEL_NAMES) - {"NCA"}
    ):
        raise RuntimeError("普通模型执行队列与固定 17 模型注册表不一致。")

    effective_jobs, normal_available_gib = choose_effective_parallel_jobs(
        MODEL_PARALLEL_JOBS
    )
    available_text = (
        "未知"
        if normal_available_gib is None
        else f"{normal_available_gib:.2f} GiB"
    )
    print(
        f"[{wall_time_text()}] [{stage_label}] 普通模型队列开始 | "
        f"模型数={len(normal_execution_order)} | "
        f"请求并发={MODEL_PARALLEL_JOBS}, 实际并发={effective_jobs} | "
        f"可用物理内存={available_text} | 内部线程=1",
        flush=True,
    )

    if effective_jobs == 1:
        for model_name in normal_execution_order:
            returned_name, result = execute_model_task(
                model_name,
                X_fit_array,
                y_fit_array,
                X_predict_array,
                stage_label,
                "resource_queue_serial",
                MODEL_PARALLEL_JOBS,
                1,
                model_positions[model_name],
                model_count,
            )
            results_by_name[returned_name] = result
            completed_count += 1
            print_model_result(
                stage_label,
                returned_name,
                result,
                completed_count,
                model_count,
                stage_started_at,
            )
            if result[1] != "success":
                break
    else:
        try:
            with parallel_config(
                backend="loky",
                n_jobs=effective_jobs,
                inner_max_num_threads=MODEL_INNER_MAX_THREADS,
            ):
                completion_stream = Parallel(
                    n_jobs=effective_jobs,
                    backend="loky",
                    return_as="generator_unordered",
                    batch_size=1,
                    pre_dispatch=effective_jobs,
                    max_nbytes="1M",
                    mmap_mode="r",
                )(
                    delayed(execute_model_task)(
                        model_name,
                        X_fit_array,
                        y_fit_array,
                        X_predict_array,
                        stage_label,
                        "parallel_loky",
                        MODEL_PARALLEL_JOBS,
                        effective_jobs,
                        model_positions[model_name],
                        model_count,
                    )
                    for model_name in normal_execution_order
                )
                for returned_name, result in completion_stream:
                    results_by_name[returned_name] = result
                    completed_count += 1
                    print_model_result(
                        stage_label,
                        returned_name,
                        result,
                        completed_count,
                        model_count,
                        stage_started_at,
                    )
        finally:
            shutdown_reusable_loky_workers()

    gc.collect()
    stage_elapsed_minutes = (
        time.perf_counter() - stage_started_at
    ) / 60
    print(
        f"[{wall_time_text()}] [{stage_label}] 阶段结束 | "
        f"完成={len(results_by_name)}/{model_count} | "
        f"阶段耗时={stage_elapsed_minutes:.2f} 分钟",
        flush=True,
    )
    return [
        (model_name, results_by_name[model_name])
        for model_name in MODEL_NAMES
        if model_name in results_by_name
    ]


def persist_required_model_failure(logs, window_id, iteration):
    """在抛出终止异常前保存截至失败点的全部模型诊断。"""
    failure_df = pd.DataFrame(logs)
    failure_df["window_id"] = window_id
    failure_df["iteration"] = iteration
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failure_path = RUN_OUTPUT_DIR / "sg_al_required_model_failure.csv"
    failure_df.to_csv(
        failure_path,
        mode="a",
        header=not failure_path.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def run_first_layer_with_oof(
    X_train,
    y_train,
    X_test,
    fold_ids,
    train_stocks,
    test_stocks,
    window_id=None,
    iteration=None,
):
    """
    生成第一层无折内拟合泄漏的训练特征 Z_train 和测试特征 Z_test。

    对每个基础模型：
    - OOF_FOLDS 个 OOF 拟合分别填充 Z_train 中对应验证块；
    - 再用完整当前训练池拟合一次，预测当前剩余测试池形成 Z_test。

    原始 143 维特征的填补和标准化必须在每个拟合内部重新估计；返回的
    Z_train/Z_test 均为固定模型顺序的 17 列硬标签。
    """
    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    X_test = pd.DataFrame(X_test).reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True).astype(int)
    fold_ids = pd.Series(fold_ids).reset_index(drop=True).astype(int)
    train_stocks = pd.Series(train_stocks).reset_index(drop=True)
    test_stocks = pd.Series(test_stocks).reset_index(drop=True)
    layer_name = "layer_1"

    if not (
        len(X_train)
        == len(y_train)
        == len(fold_ids)
        == len(train_stocks)
    ):
        raise ValueError(
            "X_train、y_train、fold_ids 和 train_stocks 长度不一致。"
        )
    if len(X_test) != len(test_stocks):
        raise ValueError("X_test 和 test_stocks 长度不一致。")
    if sorted(fold_ids.unique().tolist()) != list(range(OOF_FOLDS)):
        raise ValueError(
            f"{layer_name} 必须具有完整的 {OOF_FOLDS} 个 OOF 折。"
        )

    validate_required_model_names(MODEL_NAMES)
    oof_predictions = {
        model_name: np.full(len(X_train), np.nan)
        for model_name in MODEL_NAMES
    }
    logs = []

    def abort(message):
        persist_required_model_failure(
            logs,
            window_id,
            iteration,
        )
        raise RuntimeError(message)

    for fold_id in range(OOF_FOLDS):
        validation_mask = fold_ids.eq(fold_id).to_numpy()
        fit_mask = ~validation_mask
        if not validation_mask.any() or not fit_mask.any():
            raise ValueError(
                f"{layer_name} 的第 {fold_id} 折为空。"
            )

        X_fit, X_validation = fit_fold_preprocessor(
            X_train.loc[fit_mask],
            train_stocks.loc[fit_mask],
            X_train.loc[validation_mask],
            train_stocks.loc[validation_mask],
        )

        fold_failures = []
        for model_name, result in run_required_model_batch(
            X_fit,
            y_train.loc[fit_mask],
            X_validation,
            stage_label=(
                f"window={window_id} | iter={iteration} | "
                f"layer_1 OOF fold={fold_id + 1}/{OOF_FOLDS}"
            ),
        ):
            prediction, status, error_message, diagnostics = result
            logs.append({
                "layer": layer_name,
                "model_name": model_name,
                "fit_stage": "oof",
                "fold_id": fold_id,
                "status": status,
                "error_message": error_message,
                **diagnostics,
            })
            if status != "success":
                fold_failures.append(
                    f"{layer_name} 的必需模型 {model_name} "
                    f"在 OOF 折 {fold_id} 失败：{error_message}"
                )
            else:
                oof_predictions[model_name][validation_mask] = prediction

        if fold_failures:
            abort(fold_failures[0])

        del X_fit, X_validation
        gc.collect()

    X_full_fit, X_full_test = fit_fold_preprocessor(
        X_train,
        train_stocks,
        X_test,
        test_stocks,
    )

    train_prediction_df = pd.DataFrame(index=np.arange(len(X_train)))
    test_prediction_df = pd.DataFrame(index=np.arange(len(X_test)))

    for model_name in MODEL_NAMES:
        if np.isnan(oof_predictions[model_name]).any():
            logs.append({
                "layer": layer_name,
                "model_name": model_name,
                "fit_stage": "oof_coverage_check",
                "fold_id": None,
                "status": "failed",
                "error_message": "OOF 预测未完整覆盖训练集。",
            })
            abort(
                f"{layer_name} 的 {model_name} 未完整生成 OOF 预测。"
            )

    full_fit_failures = []
    full_fit_results = run_required_model_batch(
        X_full_fit,
        y_train,
        X_full_test,
        stage_label=(
            f"window={window_id} | iter={iteration} | "
            "layer_1 full_train"
        ),
    )
    for model_name, result in full_fit_results:
        prediction, status, error_message, diagnostics = result
        logs.append({
            "layer": layer_name,
            "model_name": model_name,
            "fit_stage": "full_train",
            "fold_id": None,
            "status": status,
            "error_message": error_message,
            **diagnostics,
        })
        if status != "success":
            full_fit_failures.append(
                f"{layer_name} 的必需模型 {model_name} "
                f"在完整训练集拟合失败：{error_message}"
            )
        else:
            train_prediction_df[model_name] = (
                oof_predictions[model_name].astype(int)
            )
            test_prediction_df[model_name] = prediction.astype(int)

    if full_fit_failures:
        abort(full_fit_failures[0])

    if list(train_prediction_df.columns) != list(REQUIRED_MODEL_NAMES):
        raise RuntimeError(f"{layer_name} 未生成固定顺序的 17 列输出。")

    return (
        train_prediction_df,
        test_prediction_df,
        pd.DataFrame(logs),
    )


def run_second_layer_full_fit(
    X_train,
    y_train,
    X_test,
    window_id=None,
    iteration=None,
):
    """
    第二层只在完整 Z_train 上拟合 17 个模型并预测 Z_test。

    当前没有第三层，也不消费第二层训练预测，因此无需再次做 OOF。
    这样每轮第二层从 102 次拟合降为 17 次，同时不改变最终 P_test 定义。
    """
    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    X_test = pd.DataFrame(X_test).reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True).astype(int)
    layer_name = "layer_2"

    if len(X_train) != len(y_train):
        raise ValueError("X_train 和 y_train 长度不一致。")

    validate_required_model_names(MODEL_NAMES)
    test_prediction_df = pd.DataFrame(index=np.arange(len(X_test)))
    logs = []

    def abort(message):
        persist_required_model_failure(
            logs,
            window_id,
            iteration,
        )
        raise RuntimeError(message)

    failures = []
    for model_name, result in run_required_model_batch(
        X_train,
        y_train,
        X_test,
        stage_label=(
            f"window={window_id} | iter={iteration} | "
            "layer_2 full_train"
        ),
    ):
        prediction, status, error_message, diagnostics = result
        logs.append({
            "layer": layer_name,
            "model_name": model_name,
            "fit_stage": "full_train",
            "fold_id": None,
            "status": status,
            "error_message": error_message,
            **diagnostics,
        })
        if status != "success":
            failures.append(
                f"{layer_name} 的必需模型 {model_name} "
                f"在完整训练集拟合失败：{error_message}"
            )
        else:
            test_prediction_df[model_name] = prediction.astype(int)

    if failures:
        abort(failures[0])

    if list(test_prediction_df.columns) != list(REQUIRED_MODEL_NAMES):
        raise RuntimeError(f"{layer_name} 未生成固定顺序的 17 列输出。")

    return test_prediction_df, pd.DataFrame(logs)


def ensemble_vote(prediction_df):
    """
    汇总第二层 17 个硬标签，并以模型间标准差衡量分歧。

    pred_label 不是多数票，而是 17 个类别编号均值最接近的 0/1/2；
    agreement_rate 才是众数占比。NumPy 按列向量化计算，避免逐行
    DataFrame.apply/value_counts。
    """
    prediction_array = prediction_df.to_numpy(dtype=int, copy=False)
    p_mean_array = prediction_array.mean(axis=1)
    p_std_array = prediction_array.std(axis=1, ddof=0)
    label_array = np.asarray(LABELS, dtype=float)
    nearest = np.abs(
        p_mean_array[:, None] - label_array[None, :]
    ).argmin(axis=1)

    index = prediction_df.index
    pred_label = pd.Series(
        label_array[nearest].astype(int),
        index=index,
    )
    p_mean = pd.Series(p_mean_array, index=index)
    p_std = pd.Series(p_std_array, index=index)
    label_counts = np.column_stack([
        (prediction_array == label).sum(axis=1)
        for label in LABELS
    ])
    agreement_rate = pd.Series(
        label_counts.max(axis=1) / prediction_array.shape[1],
        index=index,
    )
    return pred_label, p_mean, p_std, agreement_rate


def compute_multiclass_metrics(y_true, y_pred):
    """计算逐类、总体和混淆矩阵指标；固定输出缺失类别的零值指标。"""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )

    metrics = {}
    for index, label in enumerate(LABELS):
        metrics[f"precision_{label}"] = float(precision[index])
        metrics[f"recall_{label}"] = float(recall[index])
        metrics[f"f1_{label}"] = float(f1[index])
        metrics[f"support_{label}"] = int(support[index])
        metrics[f"acc_{label}"] = float(
            accuracy_score(
                (y_true == label).astype(int),
                (y_pred == label).astype(int),
            )
        )

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["macro_f1"] = float(
        f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
    )
    metrics["weighted_f1"] = float(
        f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="weighted",
            zero_division=0,
        )
    )
    metrics["confusion_matrix"] = json.dumps(
        confusion_matrix(
            y_true,
            y_pred,
            labels=LABELS,
        ).tolist(),
        ensure_ascii=False,
    )
    return metrics


# ============================================================
# 6. 单季度 SG-AL
# ============================================================

def run_sg_al_single_window(df_model, feature_cols, window):
    """
    执行一个季度窗口内的完整 SG-AL 伪标签反馈。

    真实测试标签只在样本最终出池时写入评价字段，绝不参与模型拟合或
    不确定性排序。反馈标签始终来自第二层模型共识。
    """
    data = prepare_window_data(df_model, feature_cols, window)
    current_train_X = data["train_X_raw"]
    current_test_X = data["test_X_raw"]
    current_train_y = data["y_train"]
    current_test_y = data["y_test"]
    current_train_meta = data["train_meta"]
    current_test_meta = data["test_meta"]
    window_info = data["window_info"]
    del data
    gc.collect()

    initial_train_size = len(current_train_X)
    initial_test_size = len(current_test_X)
    # 反馈批量以“初始”测试池计算，保证每轮数量固定、结果便于审计。
    d_per_round = SG_AL_D_PER_ROUND
    if d_per_round is None:
        d_per_round = max(
            1,
            int(np.ceil(
                initial_test_size * SG_AL_CONSENSUS_FRACTION
            )),
        )
    max_iterations = SG_AL_MAX_ITERATIONS
    if max_iterations is None:
        max_iterations = int(
            np.ceil(initial_test_size / d_per_round)
        )

    final_predictions = []
    iteration_logs = []
    layer_logs = []

    if SG_AL_VERBOSE:
        print(
            f"\n[{wall_time_text()}] 窗口开始 | "
            f"window={window_info['window_id']} | "
            f"训练 {window_info['train_period']} | "
            f"测试 {window_info['test_period']} | "
            f"训练行 {initial_train_size:,} | "
            f"测试行 {initial_test_size:,} | "
            f"每轮反馈 {d_per_round:,} | "
            f"预计迭代 {max_iterations}"
        )

    iteration = 1
    while (
        len(current_test_X) > 0
        and iteration <= max_iterations
    ):
        iteration_started_at = time.perf_counter()
        if SG_AL_VERBOSE:
            print(
                f"\n[{wall_time_text()}] 迭代开始 | "
                f"window={window_info['window_id']} | "
                f"iter={iteration}/{max_iterations} | "
                f"train={len(current_train_X):,}, "
                f"remaining_test={len(current_test_X):,}"
            )

        # 第一层 OOF 训练列供第二层拟合；完整拟合列用于当前测试池。
        Z_train, Z_test, first_logs = run_first_layer_with_oof(
            current_train_X,
            current_train_y,
            current_test_X,
            current_train_meta["oof_fold"],
            current_train_meta[STOCK_COL],
            current_test_meta[STOCK_COL],
            window_id=window_info["window_id"],
            iteration=iteration,
        )
        first_logs["window_id"] = window_info["window_id"]
        first_logs["iteration"] = iteration
        layer_logs.append(first_logs)

        # 第二层没有后续 stacking 层，故只需要完整拟合后的测试预测。
        P_test, second_logs = run_second_layer_full_fit(
            Z_train,
            current_train_y,
            Z_test,
            window_id=window_info["window_id"],
            iteration=iteration,
        )
        second_logs["window_id"] = window_info["window_id"]
        second_logs["iteration"] = iteration
        layer_logs.append(second_logs)

        pred_label, p_mean, p_std, agreement_rate = ensemble_vote(
            P_test
        )
        round_prediction = current_test_meta.copy()
        round_prediction["window_id"] = window_info["window_id"]
        round_prediction["train_period"] = window_info["train_period"]
        round_prediction["test_period"] = window_info["test_period"]
        round_prediction["prediction_asof"] = (
            window_info["prediction_asof"]
        )
        round_prediction["prediction_target_quarter"] = (
            window_info["target_quarter"]
        )
        round_prediction["alignment_mode"] = (
            window_info["alignment_mode"]
        )
        round_prediction["iteration"] = iteration
        round_prediction["pred_label"] = pred_label.to_numpy()
        round_prediction["p_mean"] = p_mean.to_numpy()
        round_prediction["p_std"] = p_std.to_numpy()
        round_prediction["s_tilde"] = p_std.to_numpy()
        round_prediction["agreement_rate"] = agreement_rate.to_numpy()
        round_prediction["requested_model_count"] = len(MODEL_NAMES)
        round_prediction["layer_1_active_model_count"] = (
            Z_test.shape[1]
        )
        round_prediction["layer_2_active_model_count"] = (
            P_test.shape[1]
        )
        round_prediction["selection_order"] = np.arange(
            len(round_prediction)
        )

        selected_n = min(d_per_round, len(round_prediction))
        # lexsort 的最后一个键为主键：先按 p_std 升序，同分时保持当前
        # 测试池原始顺序。只排序两个 NumPy 向量，避免复制整张元数据表。
        selected_positions = np.lexsort((
            round_prediction["selection_order"].to_numpy(),
            round_prediction["p_std"].to_numpy(),
        ))[:selected_n]
        selected_indices = pd.Index(
            round_prediction.index.to_numpy()[selected_positions]
        )
        selected_rows = round_prediction.loc[
            selected_indices
        ].copy()
        selected_rows["true_label"] = current_test_y.loc[
            selected_indices
        ].to_numpy()
        final_predictions.append(selected_rows)

        # 这里只读取 true_label 用于离线评价；进入训练池的是 pred_label。
        pseudo_X = current_test_X.loc[selected_indices].copy()
        pseudo_y = (
            selected_rows["pred_label"]
            .astype(int)
            .reset_index(drop=True)
        )
        pseudo_meta = (
            current_test_meta.loc[selected_indices]
            .copy()
            .reset_index(drop=True)
        )
        # 所有测试季度伪样本归入最后一折，保证同一季度不会被拆到多折。
        pseudo_meta["oof_fold"] = OOF_FOLDS - 1

        current_train_X = pd.concat(
            [current_train_X, pseudo_X],
            ignore_index=True,
        )
        current_train_y = pd.concat(
            [current_train_y, pseudo_y],
            ignore_index=True,
        )
        current_train_meta = pd.concat(
            [current_train_meta, pseudo_meta],
            ignore_index=True,
        )

        # 删除已反馈样本并重置索引，使下一轮 X/y/meta 继续逐行严格对齐。
        keep_mask = ~current_test_X.index.isin(selected_indices)
        current_test_X = (
            current_test_X.loc[keep_mask].reset_index(drop=True)
        )
        current_test_y = (
            current_test_y.loc[keep_mask].reset_index(drop=True)
        )
        current_test_meta = (
            current_test_meta.loc[keep_mask].reset_index(drop=True)
        )

        iteration_logs.append({
            "window_id": window_info["window_id"],
            "test_period": window_info["test_period"],
            "iteration": iteration,
            "selected_n": int(selected_n),
            "remaining_test_after": int(len(current_test_X)),
            "mean_p_mean": float(selected_rows["p_mean"].mean()),
            "mean_p_std": float(selected_rows["p_std"].mean()),
            "mean_s_tilde": float(selected_rows["s_tilde"].mean()),
            "mean_agreement_rate": float(
                selected_rows["agreement_rate"].mean()
            ),
            "requested_model_count": len(MODEL_NAMES),
            "layer_1_active_model_count": int(Z_test.shape[1]),
            "layer_2_active_model_count": int(P_test.shape[1]),
        })
        if SG_AL_VERBOSE:
            print(
                f"[{wall_time_text()}] 迭代完成 | "
                f"window={window_info['window_id']} | "
                f"iter={iteration}/{max_iterations} | "
                f"selected={selected_n:,} | "
                f"remaining={len(current_test_X):,} | "
                f"耗时={(time.perf_counter() - iteration_started_at) / 60:.2f} 分钟",
                flush=True,
            )

        del (
            Z_train,
            Z_test,
            P_test,
            first_logs,
            second_logs,
            round_prediction,
            selected_rows,
            pseudo_X,
            pseudo_y,
            pseudo_meta,
        )
        gc.collect()
        iteration += 1

    predictions_df = (
        pd.concat(final_predictions, ignore_index=True)
        if final_predictions
        else pd.DataFrame()
    )
    if len(predictions_df) != initial_test_size:
        raise RuntimeError(
            f"窗口 {window_info['window_id']} 只预测了 "
            f"{len(predictions_df)}/{initial_test_size} 条测试记录。"
        )

    metrics = compute_multiclass_metrics(
        predictions_df["true_label"],
        predictions_df["pred_label"],
    )
    metrics_row = {
        **window_info,
        "initial_train_rows": int(initial_train_size),
        "test_rows": int(initial_test_size),
        "predicted_rows": int(len(predictions_df)),
        "feature_version": (
            "quarterly_numeric_fold_local_impute_standardize"
        ),
        "model_names": ",".join(MODEL_NAMES),
        "requested_model_count": len(MODEL_NAMES),
        "d_per_round": int(d_per_round),
        "consensus_fraction": SG_AL_CONSENSUS_FRACTION,
        **metrics,
    }

    if SG_AL_VERBOSE:
        print(
            f"[{wall_time_text()}] 窗口完成 | "
            f"window={window_info['window_id']} | "
            f"iterations={iteration - 1} | "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

    return {
        "window_info": window_info,
        "predictions_df": predictions_df,
        "iteration_logs_df": pd.DataFrame(iteration_logs),
        "model_logs_df": (
            pd.concat(layer_logs, ignore_index=True)
            if layer_logs
            else pd.DataFrame()
        ),
        "metrics_row": metrics_row,
    }


# ============================================================
# 7. 落盘与主程序
# ============================================================

def append_frame(frame, path, write_state, state_key):
    """按窗口增量写顶层汇总，避免把全部预测和逐拟合日志常驻内存。"""
    if frame.empty:
        return
    first_write = not write_state.get(state_key, False)
    frame.to_csv(
        path,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
        encoding="utf-8-sig",
    )
    write_state[state_key] = True


def serializable_window(window):
    """移除仅供内存筛选使用的 Period 对象和季度列表。"""
    return {
        key: value
        for key, value in window.items()
        if key != "train_quarters"
        and key not in {
            "train_start_period",
            "train_end_period",
            "test_period_obj",
        }
    }


def run_all_windows(df_model, feature_cols, windows):
    """
    串行执行并立即落盘每个季度窗口。

    17 模型中部分算法本身占用大量内存，因此窗口级串行执行比并发更稳妥。
    每个窗口同时写独立分区和顶层增量汇总；窗口对象释放后再显式回收。
    """
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partition_root = RUN_OUTPUT_DIR / "window_partitions"
    partition_root.mkdir(parents=True, exist_ok=True)

    failure_path = RUN_OUTPUT_DIR / "sg_al_required_model_failure.csv"
    if failure_path.exists():
        failure_path.unlink()

    aggregate_paths = {
        "predictions": RUN_OUTPUT_DIR / "sg_al_predictions.csv",
        "iterations": RUN_OUTPUT_DIR / "sg_al_iteration_logs.csv",
        "models": RUN_OUTPUT_DIR / "sg_al_model_logs.csv",
    }
    write_state = {}
    metrics_rows = []
    completed_windows = []
    model_status_parts = []
    pooled_true = []
    pooled_pred = []

    started_at = time.perf_counter()
    for window in windows:
        result = run_sg_al_single_window(
            df_model,
            feature_cols,
            window,
        )
        window_info = result["window_info"]
        quarter_dir = (
            partition_root / window_info["test_period"]
        )
        quarter_dir.mkdir(parents=True, exist_ok=True)

        result["predictions_df"].to_csv(
            quarter_dir / "predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        result["iteration_logs_df"].to_csv(
            quarter_dir / "iteration_log.csv",
            index=False,
            encoding="utf-8-sig",
        )
        result["model_logs_df"].to_csv(
            quarter_dir / "model_log.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame([result["metrics_row"]]).to_csv(
            quarter_dir / "metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        with open(
            quarter_dir / "window_manifest.json",
            "w",
            encoding="utf-8",
        ) as manifest_file:
            json.dump(
                window_info,
                manifest_file,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        append_frame(
            result["predictions_df"],
            aggregate_paths["predictions"],
            write_state,
            "predictions",
        )
        append_frame(
            result["iteration_logs_df"],
            aggregate_paths["iterations"],
            write_state,
            "iterations",
        )
        append_frame(
            result["model_logs_df"],
            aggregate_paths["models"],
            write_state,
            "models",
        )

        metrics_rows.append(result["metrics_row"])
        completed_windows.append(window_info)
        pooled_true.extend(
            result["predictions_df"]["true_label"].astype(int).tolist()
        )
        pooled_pred.extend(
            result["predictions_df"]["pred_label"].astype(int).tolist()
        )
        if not result["model_logs_df"].empty:
            model_status_parts.append(
                result["model_logs_df"]
                .groupby(
                    ["layer", "model_name", "status"],
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    count=("status", "size"),
                    elapsed_seconds=("elapsed_seconds", "sum"),
                )
            )

        del result
        gc.collect()

    total_seconds = time.perf_counter() - started_at
    metrics_df = pd.DataFrame(metrics_rows)
    completed_df = pd.DataFrame(completed_windows)
    pooled_metrics = compute_multiclass_metrics(
        pooled_true,
        pooled_pred,
    )
    pooled_df = pd.DataFrame([{
        "sample_count": len(pooled_true),
        "window_count": len(windows),
        "frequency": "quarterly",
        "train_window_quarters": TRAIN_WINDOW_QUARTERS,
        "test_window_quarters": 1,
        "step_quarters": 1,
        "feature_count": len(feature_cols),
        "model_names": ",".join(MODEL_NAMES),
        "consensus_fraction": SG_AL_CONSENSUS_FRACTION,
        "total_seconds": total_seconds,
        "total_minutes": total_seconds / 60,
        **pooled_metrics,
    }])

    metric_cols = [
        col for col in metrics_df.columns
        if col.startswith(
            ("precision_", "recall_", "f1_", "support_", "acc_")
        )
        or col in {"accuracy", "macro_f1", "weighted_f1"}
    ]
    window_mean_df = (
        metrics_df[metric_cols].mean().to_frame().T
    )
    window_mean_df.insert(0, "model_name", "SG-AL")
    window_mean_df.insert(1, "window_count", len(metrics_df))
    window_mean_df.insert(2, "total_seconds", total_seconds)

    if model_status_parts:
        model_status_summary_df = (
            pd.concat(model_status_parts, ignore_index=True)
            .groupby(
                ["layer", "model_name", "status"],
                as_index=False,
                dropna=False,
            )[["count", "elapsed_seconds"]]
            .sum()
        )
    else:
        model_status_summary_df = pd.DataFrame()

    metrics_df.to_csv(
        RUN_OUTPUT_DIR / "sg_al_window_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pooled_df.to_csv(
        RUN_OUTPUT_DIR / "sg_al_pooled_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    window_mean_df.to_csv(
        RUN_OUTPUT_DIR / "sg_al_window_mean_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    completed_df.to_csv(
        RUN_OUTPUT_DIR / "sg_al_completed_window_definitions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model_status_summary_df.to_csv(
        RUN_OUTPUT_DIR / "sg_al_model_status_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "total_seconds": total_seconds,
        "metrics_df": metrics_df,
        "pooled_df": pooled_df,
        "window_mean_df": window_mean_df,
        "completed_df": completed_df,
        "model_status_summary_df": model_status_summary_df,
    }


def main():
    """按“季度化→契约→窗口→训练→汇总清单”的顺序运行完整流程。"""
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel_result = load_quarterly_panel()
    quarterly_df = panel_result["quarterly_df"]
    feature_result = build_feature_contract(quarterly_df)
    feature_cols = feature_result["feature_cols"]

    # 特征只在这里统一解析一次。NaN 不在全局填补，真正的填补和标准化
    # 仍由每个第一层 OOF 拟合仅依据其训练子集完成。
    quarterly_df, feature_numeric_coercion_audit = (
        coerce_features_to_numeric(
            quarterly_df,
            feature_cols,
        )
    )
    df_model = (
        quarterly_df.dropna(subset=[TARGET_COL])
        .reset_index(drop=True)
    )
    windows = build_rolling_windows(df_model)
    window_definitions_df = pd.DataFrame([
        serializable_window(window)
        for window in windows
    ])

    panel_result["target_alignment_audit"].to_csv(
        RUN_OUTPUT_DIR / "sg_al_target_alignment_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    panel_result["target_alignment_by_quarter"].to_csv(
        RUN_OUTPUT_DIR / "sg_al_target_alignment_by_quarter.csv",
        index=False,
        encoding="utf-8-sig",
    )
    panel_result["quarter_sampling_audit"].to_csv(
        RUN_OUTPUT_DIR / "sg_al_quarter_sampling_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_result["feature_group_count_df"].to_csv(
        RUN_OUTPUT_DIR / "sg_al_feature_group_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_result["feature_availability_audit_df"].to_csv(
        RUN_OUTPUT_DIR / "sg_al_feature_availability_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_numeric_coercion_audit.to_csv(
        RUN_OUTPUT_DIR / "sg_al_feature_numeric_coercion_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    window_definitions_df.to_csv(
        RUN_OUTPUT_DIR / "sg_al_window_definitions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"特征契约通过：{len(feature_cols)} 个特征；"
        f"滚动窗口：{len(windows)} 个；"
        f"运行模式：{SG_AL_RUN_MODE}。"
    )
    print(
        "固定 17 个基础学习器："
        + ", ".join(MODEL_NAMES)
    )
    if MODEL_PARALLEL_JOBS == 1:
        print(
            "模型执行：逐模型串行；失败后立即停止当前折；"
            f"NCA独立进程={NCA_EXCLUSIVE_PROCESS}。"
        )
    else:
        print(
            "模型安全并行：NCA 独占 spawn 子进程；"
            f"其余 16 模型最多 {MODEL_PARALLEL_JOBS} 个 loky 进程；"
            f"每进程内部线程上限 {MODEL_INNER_MAX_THREADS}；"
            "OOF 折、SG-AL 迭代和季度窗口串行。"
        )

    run_result = run_all_windows(
        df_model,
        feature_cols,
        windows,
    )

    feature_group_counts = {
        row.feature_group: int(row.actual_count)
        for row in feature_result[
            "feature_group_count_df"
        ].itertuples(index=False)
    }
    with open(
        RUN_OUTPUT_DIR / "sg_al_run_manifest.json",
        "w",
        encoding="utf-8",
    ) as manifest_file:
        json.dump(
            {
                "input_file": str(INPUT_FILE),
                "weekly_input_rows": int(
                    panel_result["weekly_input_shape"][0]
                ),
                "quarterly_rows": int(len(quarterly_df)),
                "model_rows": int(len(df_model)),
                "stock_count": int(df_model[STOCK_COL].nunique()),
                "frequency": "quarterly",
                "quarter_definition": (
                    "Q1=W01-W13;Q2=W14-W26;"
                    "Q3=W27-W39;Q4=W40-W53"
                ),
                "quarter_value_rule": (
                    "last_available_week_row_per_stock_quarter"
                ),
                "target_alignment": (
                    "same_quarter_last_week_features_to_"
                    "same_quarter_last_week_fsfp"
                ),
                "source_target_col": SOURCE_TARGET_COL,
                "target_col": TARGET_COL,
                "train_window_quarters": TRAIN_WINDOW_QUARTERS,
                "test_window_quarters": 1,
                "step_quarters": 1,
                "oof_folds": OOF_FOLDS,
                "quarters_per_oof_fold": QUARTERS_PER_FOLD,
                "run_mode": SG_AL_RUN_MODE,
                "single_test_quarter": (
                    windows[0]["test_quarter"]
                    if len(windows) == 1
                    else None
                ),
                "single_quarter_train_max_rows": (
                    SG_AL_SINGLE_QUARTER_TRAIN_MAX_ROWS
                ),
                "feature_count": len(feature_cols),
                "feature_group_counts": feature_group_counts,
                "feature_values_coerced_to_missing": int(
                    feature_numeric_coercion_audit[
                        "coerced_to_missing"
                    ].sum()
                ),
                "Peer_FSFP_included": "Peer_FSFP" in feature_cols,
                "model_names": MODEL_NAMES,
                "model_parallel_jobs": MODEL_PARALLEL_JOBS,
                "model_parallel_backend": (
                    "serial"
                    if MODEL_PARALLEL_JOBS == 1
                    else "nca_exclusive_spawn_plus_loky"
                ),
                "model_inner_max_threads": MODEL_INNER_MAX_THREADS,
                "nca_exclusive_process": (
                    MODEL_PARALLEL_JOBS > 1
                    or NCA_EXCLUSIVE_PROCESS
                ),
                "oof_fold_parallelism": 1,
                "sg_al_iteration_parallelism": 1,
                "window_parallelism": 1,
                "base_learner_policy": (
                    "all_17_required_lmnn_and_rca_removed_no_fallback"
                ),
                "layer_1_policy": "5_fold_oof_plus_full_fit",
                "layer_2_policy": "full_fit_only_on_layer_1_oof_features",
                "implemented_optimizations": [
                    "groupby_idxmax_without_full_wide_table_sort",
                    "single_global_quarter_panel_sort",
                    "one_time_feature_numeric_coercion",
                    "vectorized_second_layer_vote",
                    "nca_exclusive_spawn_barrier",
                    "memory_adaptive_bounded_model_level_loky_parallelism",
                    "unordered_live_model_completion_reporting",
                    "garbage_collection_at_stage_boundaries",
                    "incremental_window_output",
                ],
                "consensus_fraction": SG_AL_CONSENSUS_FRACTION,
                "total_seconds": run_result["total_seconds"],
                "output_dir": str(RUN_OUTPUT_DIR),
            },
            manifest_file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"季度 SG-AL 完成，用时 "
        f"{run_result['total_seconds'] / 60:.2f} 分钟。"
    )
    print(f"结果目录：{RUN_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
