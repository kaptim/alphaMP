import json
import torch
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class NetworkAnalysis(BaseTransform):
    # used as pre_transform

    def __init__(self, dataset_dir):
        super().__init__()
        self.min_max_dict = {}
        self.dataset_dir = dataset_dir

    def __call__(self, data: Data) -> Data:
        g = to_networkx(data, to_undirected=True)
        self.update_json = False
        self.get_coloring(g, data)
        self.get_betweenness_centrality(g, data)
        self.get_degree_centrality(g, data)
        self.get_closeness_centrality(g, data)
        self.get_ev_centrality(g, data)
        self.get_local_efficiency(g, data)
        self.get_nb_homophily(g, data)
        self.get_articulation_points(g, data)
        if self.update_json:
            with open(self.dataset_dir + "/min_max.json", "w") as f:
                json.dump(self.min_max_dict, f)
        return data

    def save_metric_output(self, metric_name, data, output):
        data[metric_name] = output
        # save max and min for normalisation during training
        if output.max() > self.min_max_dict.get(metric_name + "_max", -float("inf")):
            self.min_max_dict[metric_name + "_max"] = output.max().item()
            self.update_json = True
        if output.min() < self.min_max_dict.get(metric_name + "_min", float("inf")):
            self.min_max_dict[metric_name + "_min"] = output.min().item()
            self.update_json = True

    def get_coloring(self, g, data):
        # compute the coloring for a graph
        coloring = nx.coloring.greedy_color(g)
        # create color mask for faster updating
        # there may be some disconnected nodes
        # (e.g., https://github.com/snap-stanford/ogb/issues/109)
        # which are not included in the graph so we color those nodes with the first color 0
        color_mask = torch.tensor(
            [coloring[i] for i in range(len(coloring.keys()))]
            + [0 for i in range(len(coloring.keys()), data.x.shape[0])]
        )
        # save as data attribute
        data.coloring = color_mask

    def get_betweenness_centrality(self, g, data):
        # compute the centrality for each node of a graph
        centrality = nx.betweenness_centrality(g)
        # insert 0 for each disconnected node
        centrality_mask = torch.tensor(
            [centrality[i] for i in range(len(centrality.keys()))]
            + [0 for i in range(len(centrality.keys()), data.x.shape[0])]
        )
        self.save_metric_output("centrality", data, centrality_mask)

    def get_degree_centrality(self, g, data):
        centrality = nx.degree_centrality(g)
        centrality_mask = torch.tensor(
            [centrality[i] for i in range(len(centrality.keys()))]
            + [0 for i in range(len(centrality.keys()), data.x.shape[0])]
        )
        self.save_metric_output("degree_centrality", data, centrality_mask)

        # compute relative degree centrality
        neighbor_degree = nx.average_neighbor_degree(g)
        neighbor_degree_centrality = {
            k: v / (g.number_of_nodes() - 1) for k, v in neighbor_degree.items()
        }  # divide by maximum possible degree n-1 to get degree centrality
        centrality_mask_rel = torch.tensor(
            [
                centrality[i] / neighbor_degree_centrality[i]
                for i in range(len(centrality.keys()))
            ]
            + [0 for i in range(len(centrality.keys()), data.x.shape[0])]
        )
        self.save_metric_output("degree_centrality_rel", data, centrality_mask_rel)

    def get_closeness_centrality(self, g, data):
        centrality = nx.closeness_centrality(g)
        centrality_mask = torch.tensor(
            [centrality[i] for i in range(len(centrality.keys()))]
            + [0 for i in range(len(centrality.keys()), data.x.shape[0])]
        )
        self.save_metric_output("closeness_centrality", data, centrality_mask)

    def get_ev_centrality(self, g, data):
        # need larger max_iter because of convergence issues
        centrality = nx.eigenvector_centrality(g, max_iter=5000)
        centrality_mask = torch.tensor(
            [centrality[i] for i in range(len(centrality.keys()))]
            + [0 for i in range(len(centrality.keys()), data.x.shape[0])]
        )
        self.save_metric_output("eigenvector_centrality", data, centrality_mask)

    def get_local_efficiency(self, g, data):
        # local efficiency: average global efficiency
        # of the subgraph induced by the neighbors of the node
        local_efficiency = {v: nx.global_efficiency(g.subgraph(g[v])) for v in g}
        # taken from nx.local_efficiency()
        efficiency_mask = torch.tensor(
            [local_efficiency[i] for i in range(len(local_efficiency.keys()))]
            + [0 for i in range(len(local_efficiency.keys()), data.x.shape[0])]
        )
        self.save_metric_output("local_efficiency", data, efficiency_mask)

    def get_nb_homophily(self, g, data):
        # taken from "LSGNN: Towards General Graph Neural Network in Node Classification by Local Similarity"
        nb_homophily = torch.zeros(data.x.shape[0])
        for edge in g.edges:
            # similarity metric: negative euclidean distance
            sim = -torch.cdist(
                data.x[edge[0]].float().unsqueeze(-1),
                data.x[edge[1]].float().unsqueeze(-1),
            ).squeeze()
            nb_homophily[edge[0]] += sim
            nb_homophily[edge[1]] += sim
        nb_homophily_normalised = nb_homophily / torch.tensor(
            [len(nbrs) if len(nbrs) != 0 else 1 for nbrs in g.adj.values()]
        )  # if statement: avoid div by 0 error in case of disconnected nodes
        self.save_metric_output("nb_homophily", data, nb_homophily_normalised)

    def get_articulation_points(self, g, data):
        # mark all articulation points in the graph
        articulation_points = nx.articulation_points(g)
        articulation_points_mask = [0 for i in range(g.number_of_nodes())]
        for pt in list(articulation_points):
            articulation_points_mask[pt] = 1

        data.articulation_points = torch.tensor(articulation_points_mask)
