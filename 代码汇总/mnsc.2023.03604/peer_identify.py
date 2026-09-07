import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nxcom
from tqdm import tqdm
import logging
import os

# Set up logging
os.makedirs('log', exist_ok=True)
logging.basicConfig(
    filename='log/peer_identify.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Script started.")

def set_node_community(G, communities):
    '''Add community to node attributes'''
    for c, v_c in enumerate(communities):
        for v in v_c:
            G.nodes[v]['community'] = c + 1

def set_edge_community(G):
    '''Find internal edges and add their community to their attributes'''
    for v, w in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            G.edges[v, w]['community'] = 0

def get_color(i, r_off=1, g_off=1, b_off=1):
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)

peer_data = []

for year in tqdm(list(range(2001, 2023))):
    logging.info(f"Processing year: {year}")
    try:
        df = pd.read_csv(r"output/bus_sim/bus_sim" + str(year) + ".csv")
        logging.info(f"Loaded data for year {year}.")
    except Exception as e:
        logging.error(f"Failed to load data for year {year}: {e}")
        continue

    nodes = list(df.columns)
    sim_mat = np.asarray(df)

    G = nx.Graph()
    G.add_nodes_from(nodes)

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if 0.5 < sim_mat[i][j] < 1:
                G.add_edge(nodes[i], nodes[j])

    for v, w in G.edges:
        G.edges[v, w]["weight"] = sim_mat[nodes.index(v)][nodes.index(w)]

    try:
        communities = sorted(nxcom.greedy_modularity_communities(G), key=len, reverse=True)
        set_node_community(G, communities)
        set_edge_community(G)

        for node in G.nodes:
            peer_data.append({'Symbol': node, 'Year': year, 'PeerID': G.nodes[node]['community']})

        node_color = [get_color(G.nodes[v]['community']) for v in G.nodes]
        external = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] == 0]
        internal = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] > 0]
        internal_color = [get_color(G.edges[e]['community']) for e in internal]
        pos = nx.spring_layout(G, k=0.5, pos=nx.circular_layout(G))

        nx.draw_networkx(
            G, pos=pos, node_size=1, with_labels=False, edgelist=internal, alpha=0.3,
            width=0.8, node_color=node_color, edge_color=internal_color
        )
        nx.draw_networkx(
            G, pos=pos, node_size=0, with_labels=False, edgelist=external, alpha=0.3,
            edge_color="#333333", width=0.8
        )

        plt.rcParams.update({'figure.figsize': (20, 20)})
        plt.tight_layout()
        plt.savefig(f'output/fig-network{year}.png')
        plt.close()
        logging.info(f"Network graph for year {year} saved.")
    except Exception as e:
        logging.error(f"Error processing year {year}: {e}")

peer_df = pd.DataFrame(peer_data)
peer_df.to_csv('output/peer_identified.csv', index=False)
logging.info("Peer identification results saved to output/peer_identified.csv.")
logging.info("Script finished.")
