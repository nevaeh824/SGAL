import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SIMILARITY_DIR = BASE_DIR / "similarity_matrices"
DEFAULT_OUTPUT_FILE = BASE_DIR / "data" / "business_communities_all_years.csv"
DEFAULT_THRESHOLD = 0.5


def load_networkx():
    try:
        import networkx as nx
        import networkx.algorithms.community as nxcom
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "缺少 networkx，请先安装后再运行分组脚本: python -m pip install networkx"
        ) from exc

    return nx, nxcom


def panel_to_matrix(data):
    """兼容旧版 Symbol1/Symbol2/Similarity 面板数据。"""
    symbols = sorted(set(data["Symbol1"]).union(set(data["Symbol2"])))
    similarity_matrix = pd.DataFrame(0.0, index=symbols, columns=symbols)

    for _, row in data.iterrows():
        symbol1 = row["Symbol1"]
        symbol2 = row["Symbol2"]
        similarity = row["Similarity"]
        similarity_matrix.loc[symbol1, symbol2] = similarity
        similarity_matrix.loc[symbol2, symbol1] = similarity

    return similarity_matrix


def extract_year(file_path):
    match = re.search(r"(?:19|20)\d{2}", file_path.stem)
    if not match:
        raise ValueError(f"无法从文件名中识别年份: {file_path.name}")
    return int(match.group(0))


def normalize_similarity_payload(payload):
    """
    将 pkl 内容统一转换成 (matrix, companies)。

    当前 similarity_matrices 下的 pkl 是:
    {
        "matrix": ndarray,
        "companies": list,
        "company_to_idx": dict
    }
    """
    if isinstance(payload, dict):
        if "matrix" not in payload or "companies" not in payload:
            raise ValueError("pkl 字典必须包含 matrix 和 companies")
        matrix = np.asarray(payload["matrix"], dtype=float)
        companies = list(payload["companies"])
    elif isinstance(payload, pd.DataFrame):
        if {"Symbol1", "Symbol2", "Similarity"}.issubset(payload.columns):
            matrix_df = panel_to_matrix(payload)
            matrix = matrix_df.to_numpy(dtype=float)
            companies = list(matrix_df.index)
        else:
            matrix = payload.to_numpy(dtype=float)
            companies = list(payload.index)
    else:
        raise TypeError(f"不支持的 pkl 内容类型: {type(payload)}")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"相似度矩阵必须是方阵，当前形状: {matrix.shape}")
    if matrix.shape[0] != len(companies):
        raise ValueError(
            f"矩阵维度和 companies 数量不一致: {matrix.shape[0]} vs {len(companies)}"
        )

    return matrix, companies


def load_similarity_matrix(file_path):
    with open(file_path, "rb") as f:
        payload = pickle.load(f)
    return normalize_similarity_payload(payload)


def build_similarity_graph(matrix, threshold=DEFAULT_THRESHOLD):
    """按阈值建无向图；只取非对角上三角，避免 pkl 对角线 1 形成自环。"""
    nx, _ = load_networkx()
    node_count = matrix.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))

    edge_mask = np.triu(matrix > threshold, k=1)
    rows, cols = np.where(edge_mask)
    graph.add_edges_from(zip(rows.tolist(), cols.tolist()))

    return graph


def detect_communities(graph):
    _, nxcom = load_networkx()
    communities = nxcom.greedy_modularity_communities(graph)
    return sorted(communities, key=lambda community: (-len(community), min(community)))


def to_python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def bus_net_all_years(
    sim_folder=DEFAULT_SIMILARITY_DIR,
    output_file=DEFAULT_OUTPUT_FILE,
    threshold=DEFAULT_THRESHOLD,
):
    all_groupings = []
    sim_folder = Path(sim_folder)
    output_file = Path(output_file)

    sim_files = sorted(sim_folder.glob("*.pkl"), key=extract_year)
    print(f"找到 {len(sim_files)} 个年份的相似度矩阵文件")

    for file_path in tqdm(sim_files, desc="处理各年份网络分组"):
        year = extract_year(file_path)

        try:
            matrix, companies = load_similarity_matrix(file_path)
            graph = build_similarity_graph(matrix, threshold)

            if graph.number_of_edges() == 0:
                print(f"年份 {year}: 网络为空，跳过")
                continue

            communities = detect_communities(graph)

            for community_id, community in enumerate(communities, 1):
                for node_index in community:
                    all_groupings.append(
                        {
                            "Year": year,
                            "Community_ID": community_id,
                            "Symbol": to_python_scalar(companies[node_index]),
                            "Community_Size": len(community),
                        }
                    )

            print(
                f"年份 {year}: 公司数 {len(companies)}, "
                f"边数 {graph.number_of_edges()}, 社区数 {len(communities)}"
            )

        except Exception as e:
            print(f"处理年份 {year} 时出错: {e}")
            continue

    if not all_groupings:
        print("没有生成任何分组数据")
        return None

    grouping_df = pd.DataFrame(all_groupings)
    grouping_df = grouping_df.sort_values(["Year", "Community_ID", "Symbol"])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    grouping_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    total_communities = grouping_df[["Year", "Community_ID"]].drop_duplicates().shape[0]
    print(f"\n分组结果已保存到: {output_file}")
    print(f"总记录数: {len(grouping_df)}")
    print(f"覆盖年份: {grouping_df['Year'].nunique()} 年")
    print(f"总社区数: {total_communities}")

    year_stats = (
        grouping_df.groupby("Year")
        .agg({"Symbol": "nunique", "Community_ID": "nunique"})
        .rename(columns={"Symbol": "Company_Count", "Community_ID": "Community_Count"})
    )

    print("\n各年份统计:")
    print(year_stats)

    return grouping_df


if __name__ == "__main__":
    print("=== 处理所有年份的业务网络分组 ===")
    bus_net_all_years()
