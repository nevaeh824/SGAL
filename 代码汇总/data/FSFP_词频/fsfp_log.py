"""
FSFP (Financial Statement Fraud Propensity) 计算脚本
基于论文: "Unearthing Financial Statement Fraud: Insights from News Coverage Analysis"

数据结构:
  all_news_dir/     全量新闻文件夹
    000003.csv   →  列: id, 标题, 来源, 数据库, 栏目, 地区, 日期, ...
                    文件名 = 6位股票代码，无正文，仅用于计数 N_it

  fraud_news_dir/   欺诈新闻文件夹（已按词表过滤）
    3.csv        →  列: stock_code, publish_time, news_id, title, source,
                         industry, area, article_content
                    文件名 = 不含前导零的股票代码，有正文，用于计算 ATF×IIF

公式:
  FSFP_it = (1 / N_it) × Σ_j (ATF_j × IIF_j)
  N_it    = 公司 i 在窗口 t 内的全量新闻数（来自 all_news_dir）
  ATF_j   = 文章 j 中欺诈词的总词频
  IIF_j   = TotalFirmCount_year / max(NewsFirmCount_j, 1)

用法:
  python compute_fsfp.py \\
      --all_news_dir   ./all_news \\
      --fraud_news_dir ./merged_news \\
      --output         fsfp_weekly.csv
"""

import os, re, glob, argparse, warnings
from datetime import timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── 欺诈词表（论文 Table 1，源代码中用于 ATF 计算）──
FRAUD_TERMS = [
    "造假", "作假", "涉嫌", "指控", "失实", "爆出", "虚假",
    "弄虚作假", "捏造", "炒作", "不实", "隐瞒", "利益输送",
    "爆料", "起底", "查出", "不属实", "篡改", "黑幕", "舞弊",
    "欺诈", "疑点", "违反", "虚报", "揭露", "纠纷", "举报",
    "偷工减料", "虚增", "虚减"
]

# A 股上市公司总数（近似，用于 IIF 分母中的 TotalFirmCount）
TOTAL_FIRMS_BY_YEAR = {
    2001: 1166, 2002: 1223, 2003: 1293, 2004: 1379, 2005: 1374,
    2006: 1457, 2007: 1571, 2008: 1625, 2009: 1774, 2010: 2128,
    2011: 2366, 2012: 2494, 2013: 2543, 2014: 2652, 2015: 2842,
    2016: 3136, 2017: 3512, 2018: 3607, 2019: 3814, 2020: 4264,
    2021: 4847, 2022: 5146, 2023: 5346, 2024: 5392, 2025: 5469,
}


# ─── 工具函数 ─────────────────────────────────────────────────────

def normalize_code(code):
    """去掉前导零: '000003' → '3'，'600519' → '600519'"""
    return str(code).lstrip("0") or "0"


def compute_atf(text):
    """ATF = 文章中所有欺诈词的总出现次数"""
    if not isinstance(text, str) or not text:
        return 0.0
    return float(sum(text.count(t) for t in FRAUD_TERMS))


def compute_new_firm_count(text, stock_name_set):
    """
    统计文章中出现了多少个已知股票代码/简称（IIF 分子的 NewsFirmCount）
    最小返回 1（文章至少与本公司相关）
    """
    if not isinstance(text, str) or not stock_name_set:
        return 1
    cnt = sum(1 for name in stock_name_set if name and len(name) >= 2 and name in text)
    return max(cnt, 1)


# ─── 数据加载 ─────────────────────────────────────────────────────

def load_all_news(csv_path):
    """
    加载全量新闻，只取日期列，用于统计 N_it。
    支持 utf-8-sig / gbk 编码。
    """
    for enc in ("utf-8-sig", "gbk"):
        try:
            df = pd.read_csv(csv_path, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    df.columns = [c.strip() for c in df.columns]
    date_col = next((c for c in df.columns if "日期" in c or "date" in c.lower()), None)
    if date_col is None:
        raise ValueError(f"找不到日期列，实际列: {df.columns.tolist()}")
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    return df.dropna(subset=["date"])[["date"]].reset_index(drop=True)


def load_fraud_news(csv_path):
    """
    加载欺诈新闻，需要 publish_time 和 article_content 列。
    """
    for enc in ("utf-8-sig", "gbk"):
        try:
            df = pd.read_csv(csv_path, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    if "publish_time" not in df.columns or "article_content" not in df.columns:
        raise ValueError(f"缺少必要列，实际列: {df.columns.tolist()}")
    df["date"] = pd.to_datetime(df["publish_time"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["article_content"] = df["article_content"].fillna("").astype(str)
    return df[["date", "article_content"]].reset_index(drop=True)


# ─── 代码映射 ─────────────────────────────────────────────────────

def build_code_mapping(all_news_dir, fraud_news_dir):
    """
    将两个文件夹的 CSV 按 normalize_code 对齐，返回映射字典。
    结构: { normalized_code: {display_code, all_news_path, fraud_news_path} }
    """
    all_map   = {normalize_code(os.path.splitext(os.path.basename(f))[0]): f
                 for f in glob.glob(os.path.join(all_news_dir,   "*.csv"))}
    fraud_map = {normalize_code(os.path.splitext(os.path.basename(f))[0]): f
                 for f in glob.glob(os.path.join(fraud_news_dir, "*.csv"))}

    mapping = {}
    for nc, ap in all_map.items():
        mapping[nc] = {
            "display_code":    os.path.splitext(os.path.basename(ap))[0].strip(),
            "all_news_path":   ap,
            "fraud_news_path": fraud_map.get(nc),
        }

    matched = sum(1 for v in mapping.values() if v["fraud_news_path"])
    print(f"全量新闻文件 : {len(all_map)}  家")
    print(f"欺诈新闻文件 : {len(fraud_map)} 家")
    print(f"成功匹配     : {matched} 家（以全量新闻为准）")
    no_fraud = [v["display_code"] for v in mapping.values() if not v["fraud_news_path"]]
    if no_fraud:
        print(f"无欺诈新闻   : {len(no_fraud)} 家（FSFP 将为 0）："
              f"{no_fraud[:5]}{'...' if len(no_fraud)>5 else ''}")
    return mapping


def build_stock_name_set(all_news_dir, company_info_path=None):
    """
    构建用于 IIF NewsFirmCount 的股票名称集合。

    优先使用 company_info_path（AF_Co.xlsx）中的股票简称和公司全称，
    能让"文章提及了多少家公司"的计数更接近论文逻辑。
    若未提供，则退化为仅用股票代码字符串（精度较低）。

    清洗规则：去掉 *ST / ST / PT 前缀（如"PT 金田A"→"金田A"），
    同时保留原始简称，确保覆盖率。
    """
    names = set()

    if company_info_path and os.path.exists(company_info_path):
        try:
            df = pd.read_excel(company_info_path, skiprows=2)
            df.columns = ['Stkcd','Stknmec','Updt','Listdt','Conme',
                          'Conmee','IndClaCd','Indus','Indnme','Udwnm','Sponsor','Http']
            df = df.dropna(subset=['Stknmec'])

            for _, row in df.iterrows():
                raw_name = str(row['Stknmec']).strip()
                names.add(raw_name)                                      # 原始简称
                clean = re.sub(r'^[\*\s]*(ST|PT)\s*', '', raw_name).strip()
                if clean:
                    names.add(clean)                                     # 去前缀核心名
                if pd.notna(row.get('Conme', None)):
                    names.add(str(row['Conme']).strip())               # 公司全称

            names = {n for n in names if len(n) >= 2}
            print(f"股票名称集合 : {len(names)} 个（来自公司信息文件）")
        except Exception as e:
            print(f"⚠ 读取公司信息文件失败，退化为代码匹配: {e}")
            company_info_path = None

    if not company_info_path:
        for f in glob.glob(os.path.join(all_news_dir, "*.csv")):
            raw = os.path.splitext(os.path.basename(f))[0].strip()
            names.add(raw)
            names.add(normalize_code(raw))
        print(f"股票名称集合 : {len(names)} 个（仅股票代码，精度较低）")

    return names


# ─── 核心计算：单公司周频 FSFP ────────────────────────────────────

def compute_fsfp_weekly(all_df, fraud_df, stock_code,
                        window_days, stock_name_set, total_firms_by_year):
    """
    对单家公司输出周频 FSFP DataFrame。
    列: stock_code, week_end, fsfp, n_total_news, n_fraud_news
    """
    if all_df.empty:
        return pd.DataFrame()

    # 预计算每篇欺诈新闻的 ATF 和 NewsFirmCount（与窗口无关，计算一次）
    if not fraud_df.empty:
        fraud_df = fraud_df.copy()
        fraud_df["atf"]     = fraud_df["article_content"].apply(compute_atf)
        fraud_df["n_firms"] = fraud_df["article_content"].apply(
            lambda t: compute_new_firm_count(t, stock_name_set))
        fraud_df.sort_values("date", inplace=True)

    all_df = all_df.sort_values("date").reset_index(drop=True)

    # 生成周频节点（每周一），从 min_date + window_days 开始
    min_date = all_df["date"].min()
    max_date = all_df["date"].max()
    first_node = max(min_date + timedelta(days=window_days),
                     min_date + timedelta(days=7))
    date_range = pd.date_range(start=first_node, end=max_date, freq="W-MON")
    if len(date_range) == 0:
        return pd.DataFrame()

    rows = []
    for week_end in date_range:
        window_start = week_end - timedelta(days=window_days)
        year = week_end.year
        total_firms = total_firms_by_year.get(year, 5000)

        # N_it：窗口内全量新闻计数
        n_total = int(
            ((all_df["date"] > window_start) & (all_df["date"] <= week_end)).sum()
        )

        if n_total == 0:
            rows.append({"stock_code": stock_code, "week_end": week_end.date(),
                         "fsfp": 0.0, "n_total_news": 0, "n_fraud_news": 0})
            continue

        # sum(ATF × IIF)：窗口内欺诈新闻的加权和
        if fraud_df.empty:
            atf_iif_sum, n_fraud = 0.0, 0
        else:
            mask = (fraud_df["date"] > window_start) & (fraud_df["date"] <= week_end)
            win = fraud_df[mask]
            n_fraud = len(win)
            # ATF_IIF = ATF × (TotalFirms / NewsFirmCount)
            atf_iif_sum = float(
                (win["atf"] * np.log(total_firms / win["n_firms"])).sum()
            ) if n_fraud > 0 else 0.0

        rows.append({
            "stock_code": stock_code,
            "week_end":   week_end.date(),
            "fsfp":       atf_iif_sum / n_total,
            "n_total_news": n_total,
            "n_fraud_news": n_fraud,
        })

    return pd.DataFrame(rows)


# ─── 批量处理 ─────────────────────────────────────────────────────

def run_batch(all_news_dir, fraud_news_dir, output_path,
              stock_list=None, start_date=None, end_date=None,
              window_days=365, total_firms_override=None,
              company_info_path=None):

    mapping = build_code_mapping(all_news_dir, fraud_news_dir)

    if stock_list:
        wanted  = {normalize_code(s) for s in stock_list}
        mapping = {k: v for k, v in mapping.items() if k in wanted}
        if not mapping:
            raise ValueError(f"未找到指定股票: {stock_list}")
        print(f"筛选后处理 {len(mapping)} 家")

    stock_name_set   = build_stock_name_set(all_news_dir, company_info_path)
    firms_map        = {**TOTAL_FIRMS_BY_YEAR, **(total_firms_override or {})}
    items            = list(mapping.items())
    total            = len(items)
    print(f"\n开始计算 {total} 家公司的周频 FSFP ...\n")

    all_results = []
    for i, (nc, paths) in enumerate(items):
        code = paths["display_code"]
        step = max(1, total // 10)
        if (i + 1) % step == 0 or i == total - 1:
            print(f"  进度: {i+1}/{total}  当前: {code}")
        try:
            all_df   = load_all_news(paths["all_news_path"])
            fraud_df = (load_fraud_news(paths["fraud_news_path"])
                        if paths["fraud_news_path"] else
                        pd.DataFrame(columns=["date", "article_content"]))

            res = compute_fsfp_weekly(
                all_df, fraud_df, code,
                window_days, stock_name_set, firms_map)
            if not res.empty:
                all_results.append(res)
        except Exception as e:
            print(f"  ⚠ 跳过 {code}: {e}")

    if not all_results:
        raise RuntimeError("所有公司计算失败，请检查路径与格式")

    final = pd.concat(all_results, ignore_index=True)
    final["week_end"] = pd.to_datetime(final["week_end"])
    if start_date:
        final = final[final["week_end"] >= pd.to_datetime(start_date)]
    if end_date:
        final = final[final["week_end"] <= pd.to_datetime(end_date)]
    final = final.sort_values(["stock_code","week_end"]).reset_index(drop=True)

    # 统计摘要
    nz = final["fsfp"] > 0
    print(f"\n✅ 完成")
    print(f"   公司数        : {final['stock_code'].nunique()}")
    print(f"   总行数        : {len(final)}")
    print(f"   FSFP 非零比例 : {nz.mean()*100:.1f}%")
    if nz.any():
        print(f"   FSFP 均值(非零): {final.loc[nz,'fsfp'].mean():.4f}")
        print(f"   FSFP 最大值    : {final['fsfp'].max():.4f}")

    final.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n📄 已保存: {output_path}")
    print("\n输出列:")
    print("  stock_code    股票代码")
    print("  week_end      窗口截止日（每周一）")
    print("  fsfp          FSFP 值")
    print("  n_total_news  窗口内全量新闻数（N_it）")
    print("  n_fraud_news  窗口内欺诈新闻数（M_it）")
    return final


# ─── 命令行 ───────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="计算周频 FSFP")
    p.add_argument("--all_news_dir",   required=True,
                   help="全量新闻文件夹（文件名=6位股票代码，如 000003.csv）")
    p.add_argument("--fraud_news_dir", required=True,
                   help="欺诈新闻文件夹（文件名=无前导零代码，如 3.csv）")
    p.add_argument("--output",         default="fsfp_weekly.csv")
    p.add_argument("--stock_list",     default=None,
                   help="仅计算指定股票，逗号分隔，如 '000003,600519'")
    p.add_argument("--start_date",     default=None)
    p.add_argument("--end_date",       default=None)
    p.add_argument("--window_days",    type=int, default=365)
    return p.parse_args()


if __name__ == "__main__":
    run_batch(
        all_news_dir      = r"all_news_dir",
        fraud_news_dir    = r"fraud_news_dir",
        output_path       = r"fsfp_weekly_log(2700-3600).csv",
        company_info_path = r"AF_Co.xlsx",
        window_days       = 365,
    )