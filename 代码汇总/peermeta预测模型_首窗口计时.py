"""
PeerMeta / SG-AL 首个季度窗口的全样本计时入口。

本脚本只运行一个窗口：
- 训练集：数据集中最早、连续的 12 个季度（3 年）；
- 测试集：紧接训练集的第 13 个季度；
- OOF：按时间顺序分为 3 折，每折包含连续 4 个季度；
- 窗口内不抽样，测试池按正式 SG-AL 规则逐轮清空。

当前数据对应：
2015Q1—2017Q4 训练，2018Q1 测试。

季度化、143 特征契约、折内预处理、固定 17 个模型、两层 stacking 和
伪标签反馈全部复用正式脚本，避免计时版本与正式版本产生实现偏差。
LMNN 和 RCA 均已删除。NCA 保持全样本算法不变并在独立子进程中独占
运行；其余 16 个模型最多 6 进程并发。OOF 折、SG-AL 迭代和季度窗口
不并行，避免多层并行叠乘内存。
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

# 必须在 pandas/numpy/sklearn 导入前限制数值库线程，避免 6 个模型进程
# 各自再创建一组 BLAS/OpenMP 线程。
for thread_env_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[thread_env_name] = "1"

import pandas as pd


# 必须在导入正式脚本前固定为 full。prepare_window_data 只有在
# single_quarter_test 模式下才会把训练集压缩到 5000 行。
os.environ["SG_AL_RUN_MODE"] = "full"

import peermeta预测模型 as pipeline


# 这些覆盖只作用于当前计时进程。正式脚本单独运行时仍保持 5 年、5 折。
BENCHMARK_TRAIN_YEARS = 3
BENCHMARK_TRAIN_QUARTERS = BENCHMARK_TRAIN_YEARS * 4
BENCHMARK_OOF_FOLDS = 3
if BENCHMARK_TRAIN_QUARTERS % BENCHMARK_OOF_FOLDS != 0:
    raise ValueError("首窗口训练季度数必须能被 OOF 折数整除。")

pipeline.TRAIN_WINDOW_QUARTERS = BENCHMARK_TRAIN_QUARTERS
pipeline.OOF_FOLDS = BENCHMARK_OOF_FOLDS
pipeline.QUARTERS_PER_FOLD = (
    BENCHMARK_TRAIN_QUARTERS // BENCHMARK_OOF_FOLDS
)
BENCHMARK_NORMAL_MODEL_JOBS = min(6, pipeline.AVAILABLE_CPU_COUNT)
pipeline.MODEL_PARALLEL_JOBS = BENCHMARK_NORMAL_MODEL_JOBS
pipeline.MODEL_INNER_MAX_THREADS = 1
pipeline.NCA_EXCLUSIVE_PROCESS = True

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "sg_al_current_quarter_first_window_3year_benchmark_model17_nca_full"
)


def select_first_full_window(df_model):
    """
    选择“数据集最早 12 个季度 -> 第 13 个季度”，并验证季度连续。

    不直接写死 2015Q1 或 2018Q1，使输入表整体平移后仍按相同规则工作；
    若最早 13 个季度不连续，则拒绝悄悄跳到更晚窗口。
    """
    available_quarters = sorted(
        df_model["quarter_period"].drop_duplicates().tolist()
    )
    required_quarter_count = pipeline.TRAIN_WINDOW_QUARTERS + 1
    if len(available_quarters) < required_quarter_count:
        raise ValueError(
            f"至少需要连续 {required_quarter_count} 个季度，才能构造"
            f"{BENCHMARK_TRAIN_YEARS}年训练和下一季度测试。"
        )

    expected_quarters = list(
        pd.period_range(
            start=available_quarters[0],
            periods=required_quarter_count,
            freq="Q",
        )
    )
    if available_quarters[:required_quarter_count] != expected_quarters:
        raise ValueError(
            f"数据集最早 {required_quarter_count} 个季度不连续，"
            "无法执行首窗口计时。"
        )

    first_window = pipeline.build_rolling_windows(df_model)[0]
    if (
        first_window["train_quarters"]
        != expected_quarters[:pipeline.TRAIN_WINDOW_QUARTERS]
        or first_window["test_period_obj"] != expected_quarters[-1]
    ):
        raise RuntimeError("正式窗口生成器与首窗口定义不一致。")
    return first_window


def write_json(payload, path):
    """以 UTF-8 写入便于人工读取的 JSON。"""
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(
            payload,
            output_file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def main():
    """准备首窗口、运行完整识别、分别记录数据处理和模型耗时。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline.RUN_OUTPUT_DIR = OUTPUT_DIR

    # 防止上一次失败日志与本次计时混淆；其余成功产物会由本次直接覆盖。
    failure_path = OUTPUT_DIR / "sg_al_required_model_failure.csv"
    if failure_path.exists():
        failure_path.unlink()

    total_started_at = time.perf_counter()
    preparation_started_at = time.perf_counter()

    panel_result = pipeline.load_quarterly_panel()
    quarterly_df = panel_result["quarterly_df"]
    feature_result = pipeline.build_feature_contract(quarterly_df)
    feature_cols = feature_result["feature_cols"]
    quarterly_df, numeric_coercion_audit = (
        pipeline.coerce_features_to_numeric(
            quarterly_df,
            feature_cols,
        )
    )
    df_model = (
        quarterly_df.dropna(subset=[pipeline.TARGET_COL])
        .reset_index(drop=True)
    )
    first_window = select_first_full_window(df_model)
    preparation_seconds = (
        time.perf_counter() - preparation_started_at
    )

    # 原始季度面板和审计表不再参与拟合，先释放以降低首窗口训练峰值内存。
    weekly_input_rows = int(panel_result["weekly_input_shape"][0])
    quarterly_rows = int(len(quarterly_df))
    coerced_to_missing = int(
        numeric_coercion_audit["coerced_to_missing"].sum()
    )
    del (
        panel_result,
        quarterly_df,
        feature_result,
        numeric_coercion_audit,
    )
    gc.collect()

    print(
        f"\n[{pipeline.wall_time_text()}] "
        f"首窗口{BENCHMARK_TRAIN_YEARS}年训练全样本计时开始："
        f"{first_window['train_period']} -> "
        f"{first_window['test_period']}"
    )
    print(
        f"OOF：{pipeline.OOF_FOLDS} 折；"
        f"每折 {pipeline.QUARTERS_PER_FOLD} 个连续季度。"
    )
    print(
        "模型执行：NCA 保持全样本并由独立 spawn 子进程独占运行；"
        f"其余 16 模型最多 {pipeline.MODEL_PARALLEL_JOBS} 个 loky 进程；"
        "可用内存不足时普通模型自动降为 4/2/1 并发；"
        "所有模型内部线程为 1。"
    )
    print(
        "外层执行：3 个 OOF 折串行、SG-AL 迭代串行、季度窗口串行；"
        "终端将打印 window、iteration、layer、fold、模型序号、墙钟时间和耗时。"
    )

    model_started_at = time.perf_counter()
    result = pipeline.run_sg_al_single_window(
        df_model,
        feature_cols,
        first_window,
    )
    model_seconds = time.perf_counter() - model_started_at

    output_started_at = time.perf_counter()
    result["predictions_df"].to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result["iteration_logs_df"].to_csv(
        OUTPUT_DIR / "iteration_log.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result["model_logs_df"].to_csv(
        OUTPUT_DIR / "model_log.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([result["metrics_row"]]).to_csv(
        OUTPUT_DIR / "metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_json(
        result["window_info"],
        OUTPUT_DIR / "window_definition.json",
    )
    output_seconds = time.perf_counter() - output_started_at

    layer_counts = (
        result["model_logs_df"]
        .groupby("layer", observed=True)
        .size()
        .to_dict()
    )
    observed_effective_jobs = sorted(
        result["model_logs_df"]["effective_model_parallel_jobs"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    observed_scheduler_modes = sorted(
        result["model_logs_df"]["scheduler_mode"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    benchmark_summary = {
        "scope": "first_window_3year_full_sample",
        "input_file": str(pipeline.INPUT_FILE),
        "output_dir": str(OUTPUT_DIR),
        "weekly_input_rows": weekly_input_rows,
        "quarterly_rows": quarterly_rows,
        "feature_count": len(feature_cols),
        "feature_values_coerced_to_missing": coerced_to_missing,
        "train_window_years": BENCHMARK_TRAIN_YEARS,
        "train_window_quarters": pipeline.TRAIN_WINDOW_QUARTERS,
        "oof_folds": pipeline.OOF_FOLDS,
        "quarters_per_oof_fold": pipeline.QUARTERS_PER_FOLD,
        "train_period": first_window["train_period"],
        "test_period": first_window["test_period"],
        "train_rows": int(result["window_info"]["train_rows"]),
        "test_rows": int(result["window_info"]["test_rows"]),
        "training_sample_cap": None,
        "sg_al_iterations": int(len(result["iteration_logs_df"])),
        "model_parallel_jobs": pipeline.MODEL_PARALLEL_JOBS,
        "effective_model_parallel_jobs_observed": (
            observed_effective_jobs
        ),
        "scheduler_modes_observed": observed_scheduler_modes,
        "model_inner_max_threads": pipeline.MODEL_INNER_MAX_THREADS,
        "nca_exclusive_process": pipeline.NCA_EXCLUSIVE_PROCESS,
        "normal_model_parallelism_policy": (
            "available_ram_gib>=16:6;>=12:4;>=8:2;<8:1"
        ),
        "oof_fold_parallelism": 1,
        "sg_al_iteration_parallelism": 1,
        "window_parallelism": 1,
        "model_contract": "17_models_lmnn_and_rca_removed_nca_full_sample",
        "layer_1_fit_count": int(layer_counts.get("layer_1", 0)),
        "layer_2_fit_count": int(layer_counts.get("layer_2", 0)),
        "total_model_fit_count": int(len(result["model_logs_df"])),
        "data_preparation_seconds": preparation_seconds,
        "model_seconds": model_seconds,
        "output_seconds": output_seconds,
        "total_wall_seconds": time.perf_counter() - total_started_at,
        "accuracy": float(result["metrics_row"]["accuracy"]),
        "macro_f1": float(result["metrics_row"]["macro_f1"]),
        "weighted_f1": float(result["metrics_row"]["weighted_f1"]),
    }
    write_json(
        benchmark_summary,
        OUTPUT_DIR / "benchmark_summary.json",
    )

    print(
        f"\n[{pipeline.wall_time_text()}] 首窗口识别完成："
        f"数据处理 {preparation_seconds / 60:.2f} 分钟；"
        f"模型识别 {model_seconds / 60:.2f} 分钟；"
        f"总耗时 {benchmark_summary['total_wall_seconds'] / 60:.2f} 分钟。"
    )
    print(f"计时结果：{OUTPUT_DIR / 'benchmark_summary.json'}")


if __name__ == "__main__":
    main()
