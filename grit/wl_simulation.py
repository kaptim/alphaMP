import os
import csv
import networkx as nx
import networkx.algorithms.graph_hashing as nxgh
import numpy as np
from torch_geometric.utils import to_networkx
from grit.utils import get_data_dir


def save_simulation(results, cfg):
    sim_f = "/".join(
        [
            "./simulations",
            "_".join(
                [
                    cfg.wandb.name.split("-")[0],
                    str(cfg.async_update.alpha),
                    str(cfg.async_update.sim_rounds),
                    str(cfg.async_update.async_runs),
                ]
            )
            + ".csv",
        ]
    )
    if not os.path.exists("./simulations"):
        os.mkdir("./simulations")
    if os.path.exists(sim_f):
        with open(sim_f, "a", newline="") as f:
            writer = csv.writer(f)
            for r in results:
                writer.writerow(r)
    else:
        with open(sim_f, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Synchronous", "Asynchronous"])
            for r in results:
                writer.writerow(r)


def simulate_1wl_sync(c_g, cfg):
    # run synchronous updates for the same number of times as asynchronous
    hashes = nx.weisfeiler_lehman_subgraph_hashes(
        c_g, iterations=cfg.async_update.sim_rounds
    )
    final_hashes = [hashes[k][-1] for k in hashes.keys()]
    # unique ID for every node?
    return len(set(final_hashes)) == c_g.number_of_nodes()


def simulate_1wl_async(c_g, cfg):
    # initial node attributes: degree of the nodes
    node_attr_name = "wl"
    nx.set_node_attributes(c_g, nxgh._init_node_labels(c_g, None, None), node_attr_name)
    # convert from node number to index in nodes object
    node_list = list(c_g.nodes())
    node_to_idx = {node_list[i]: i for i in range(len(node_list))}

    for i in range(cfg.async_update.sim_rounds):
        sync_update = nx.weisfeiler_lehman_subgraph_hashes(
            c_g, node_attr=node_attr_name, iterations=1
        )
        async_update = {}
        alphas = np.random.choice(
            [0, 1],
            c_g.number_of_nodes(),
            p=[1 - cfg.async_update.alpha, cfg.async_update.alpha],
        )
        for node in c_g.nodes():
            async_update[node] = (1 - alphas[node_to_idx[node]]) * c_g.nodes()[node][
                node_attr_name
            ] + alphas[node_to_idx[node]] * sync_update[node][0]
        nx.set_node_attributes(c_g, async_update, node_attr_name)

    return (
        len(set(nx.get_node_attributes(c_g, node_attr_name).values()))
        == c_g.number_of_nodes()
    )


def simulate_1wl(batch, cfg):
    # for each graph do 1 synchronous run (deterministic) and 5 asynchronous ones
    g = to_networkx(batch, to_undirected=True)
    # multiple graphs in one batch => split connected components
    results = []
    for c in nx.connected_components(g):
        c_g = g.subgraph(c)
        component_async_results = []
        component_sync_result = simulate_1wl_sync(c_g, cfg)
        for i in range(cfg.async_update.async_runs):
            component_async_results.append(simulate_1wl_async(c_g, cfg))
        results.append(
            [
                int(component_sync_result is True),
                sum(component_async_results) / len(component_async_results),
            ]
        )

    save_simulation(results, cfg)
