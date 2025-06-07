import torch
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class NetworkAnalysis(BaseTransform):
    # used as pre_transform

    def __call__(self, data: Data) -> Data:
        g = to_networkx(data, to_undirected=True)
        self.get_coloring(g, data)
        self.get_centrality(g, data)
        return data

    def get_coloring(self, g, data):
        # compute the coloring for a graph
        coloring = nx.coloring.greedy_color(g)
        # create color mask for faster updating
        # there may be some isolated nodes
        # (e.g., https://github.com/snap-stanford/ogb/issues/109)
        # which are not included in the graph so we color those nodes with the first color 0
        color_mask = torch.tensor(
            [coloring[i] for i in range(len(coloring.keys()))]
            + [0 for i in range(len(coloring.keys()), data.x.shape[0])]
        )
        # save as data attribute
        data.coloring = color_mask

    def get_centrality(self, g, data):
        # compute the centrality for each node of a graph
        centrality = nx.betweenness_centrality(g)
        # insert 0 for each disconnected node
        centrality_mask = torch.tensor(
            [centrality[i] for i in range(len(centrality.keys()))]
            + [0 for i in range(len(centrality.keys()), data.x.shape[0])]
        )
        # save as data attribute
        data.centrality = centrality_mask
