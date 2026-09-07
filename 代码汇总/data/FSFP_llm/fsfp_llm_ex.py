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


# ─── 工具函数（严格保持原始逻辑） ───────────────────────────────────

def normalize_code(code):
    return str(code).lstrip("0") or "0"

def compute_atf(text):
    if not isinstance(text, str) or not text:
        return 0.0
    return float(sum(text.count(t) for t in FRAUD_TERMS))

def compute_new_firm_count(text, stock_name_set):
    if not isinstance(text, str) or not stock_name_set:
        return 1
    cnt = sum(1 for name in stock_name_set if name and len(name) >= 2 and name in text)
    return max(cnt, 1)

# ─── 数据加载（仅更新列名以匹配打标后的文件，其余逻辑不动） ───────────

def load_all_news(csv_path):
    for enc in ("utf-8-sig", "gbk"):
        try:
            df = pd.read_csv(csv_path, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError: 
            continue
    df.columns = [c.strip() for c in df.columns]
    date_col = next((c for c in df.columns if "日期" in c or "date" in c.lower()), None)
    if date_col is None:
        raise ValueError(f"找不到日期列: {csv_path}")
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    return df.dropna(subset=["date"])[["date"]].reset_index(drop=True)

def load_fraud_news(csv_path):
    """
    加载欺诈新闻及 LLM 标签（risk_score）。
    """
    for enc in ("utf-8-sig", "gbk"):
        try:
            df = pd.read_csv(csv_path, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError: 
            continue
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    
    # 兼容原始列名和打标后的列名
    date_col = next((c for c in df.columns if c in ["pub_date", "publish_time"]), "pub_date")
    content_col = next((c for c in df.columns if c in ["article", "article_content"]), "article_content")
    
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])
    df["article_content"] = df[content_col].fillna("").astype(str)
    
    # 获取 LLM 分数，如果不存在则默认为 0
    if "risk_score" in df.columns:
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0)
    else:
        df["risk_score"] = 0.0
        
    return df[["date", "article_content", "risk_score"]].reset_index(drop=True)

# ─── 代码映射───────────────────────────────

def build_code_mapping(all_news_dir, fraud_news_dir):
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

# ─── 股票名称集合（严格还原您的原始函数） ─────────────────────────────

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
                names.add(raw_name)                                      
                clean = re.sub(r'^[\*\s]*(ST|PT)\s*', '', raw_name).strip()
                if clean: 
                    names.add(clean)                                     
                if pd.notna(row.get('Conme', None)):
                    names.add(str(row['Conme']).strip())  

            names = {n for n in names if len(n) >= 2}
            print(f"股票名称集合 : {len(names)} 个（来自公司信息文件）")
        except Exception as e:
            print(f"⚠ 读取公司信息文件失败: {e}")
            company_info_path = None

    if not company_info_path:
        for f in glob.glob(os.path.join(all_news_dir, "*.csv")):
            raw = os.path.splitext(os.path.basename(f))[0].strip()
            names.add(raw)
            names.add(normalize_code(raw))
        print(f"股票名称集合 : {len(names)} 个（仅股票代码）")
    return names

# ─── 核心计算：加入 LLM 激活函数修正公式 ──────────────────────────────

def compute_fsfp_weekly(all_df, fraud_df, stock_code,
                        window_days, stock_name_set, total_firms_by_year):
    """
    对单家公司输出周频 FSFP DataFrame。
    列: stock_code, week_end, fsfp, n_total_news, n_fraud_news
    """
    if all_df.empty:
        return pd.DataFrame()

    if not fraud_df.empty:
        fraud_df = fraud_df.copy()
        fraud_df["atf"]     = fraud_df["article_content"].apply(compute_atf)
        fraud_df["n_firms"] = fraud_df["article_content"].apply(
            lambda t: compute_new_firm_count(t, stock_name_set))
        # 新增逻辑：计算 LLM 修正系数 log(1 + exp(S))
        fraud_df["llm_factor"] = np.log(1 + np.exp(fraud_df["risk_score"]))
        fraud_df.sort_values("date", inplace=True)

    all_df = all_df.sort_values("date").reset_index(drop=True)
    min_date, max_date = all_df["date"].min(), all_df["date"].max()
    first_node = max(min_date + timedelta(days=window_days), min_date + timedelta(days=7))
    date_range = pd.date_range(start=first_node, end=max_date, freq="W-MON")
    if len(date_range) == 0:
        return pd.DataFrame()

    rows = []
    for week_end in date_range:
        window_start = week_end - timedelta(days=window_days)
        year, total_firms = week_end.year, total_firms_by_year.get(week_end.year, 5000)
        n_total = int(((all_df["date"] > window_start) & (all_df["date"] <= week_end)).sum())

        if n_total == 0:
            rows.append({"stock_code": stock_code, "week_end": week_end.date(),
                         "fsfp": 0.0, "n_total_news": 0, "n_fraud_news": 0})
            continue

        if fraud_df.empty:
            atf_iif_sum, n_fraud = 0.0, 0
        else:
            mask = (fraud_df["date"] > window_start) & (fraud_df["date"] <= week_end)
            win = fraud_df[mask]
            n_fraud = len(win)
            if n_fraud > 0:
                # 按照新公式计算修正后的分子：ATF * IIF * log(1 + exp(S))
                iif = np.log(total_firms / win["n_firms"])
                atf_iif_sum = float((win["atf"] * iif * win["llm_factor"]).sum())
            else:
                atf_iif_sum = 0.0

        rows.append({
            "stock_code": stock_code, 
            "week_end": week_end.date(),
            "fsfp": atf_iif_sum / n_total,
            "n_total_news": n_total,
            "n_fraud_news": n_fraud,
        })
    return pd.DataFrame(rows)

# ─── 批量处理 ──────────────────────────────

def run_batch(all_news_dir, fraud_news_dir, output_path,
              stock_list=None, start_date=None, end_date=None,
              window_days=365, total_firms_override=None,
              company_info_path=None):

    mapping = build_code_mapping(all_news_dir, fraud_news_dir)

    if stock_list:
        wanted  = {normalize_code(s) for s in stock_list}
        mapping = {k: v for k, v in mapping.items() if k in wanted}
        if not mapping: raise ValueError(f"未找到指定股票: {stock_list}")
        print(f"筛选后处理 {len(mapping)} 家")

    stock_name_set   = build_stock_name_set(all_news_dir, company_info_path)
    firms_map        = {**TOTAL_FIRMS_BY_YEAR, **(total_firms_override or {})}
    items, total     = list(mapping.items()), len(mapping)
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
                        pd.DataFrame(columns=["date", "article_content", "risk_score"]))

            res = compute_fsfp_weekly(
                all_df, fraud_df, code,
                window_days, stock_name_set, firms_map)
            if not res.empty: all_results.append(res)
        except Exception as e: 
            print(f"  ⚠ 跳过 {code}: {e}")

    if not all_results: 
        raise RuntimeError("所有公司计算失败")

    final = pd.concat(all_results, ignore_index=True)
    final["week_end"] = pd.to_datetime(final["week_end"])
    if start_date: final = final[final["week_end"] >= pd.to_datetime(start_date)]
    if end_date:   final = final[final["week_end"] <= pd.to_datetime(end_date)]
    final = final.sort_values(["stock_code","week_end"]).reset_index(drop=True)

    # 统计摘要（严格保持您的原始输出）
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
    return final

if __name__ == "__main__":
    run_batch(
        all_news_dir      = r"all_news_dir",
        fraud_news_dir    = r"fraud_news_processed_dir",
        output_path       = r"fsfp_weekly_corrected.csv",
        company_info_path = r"AF_Co.xlsx",
        window_days       = 365,
    )