import networkx as nx
import numpy as np
import torch
import torch_geometric.nn as gnn
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from ..modules.head import PredictionHead
from ..modules.feature_encoder import FeatureEncoder
from ..modules.mpnn_layer import MpnnLayer


class Mpnn(torch.nn.Module):
    def __init__(
        self,
        in_node_dim,
        num_class,
        hidden_size=64,
        num_layers=2,
        in_edge_dim=None,
        node_embed=True,
        edge_embed=True,
        dropout=0.0,
        global_pool=None,
        head="mlp",
        pad_idx=-1,
        alpha=0.5,
        alpha_eval_flag="a",
        recurrent=False,
        use_coloring=False,
        **kwargs,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.alpha = alpha
        self.alpha_eval_flag = alpha_eval_flag
        # only use one layer multiple times
        self.recurrent = recurrent
        # use coloring to update iteratively
        self.use_coloring = use_coloring
        self.num_layers = num_layers

        self.feature_encoder = FeatureEncoder(
            hidden_size=hidden_size,
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            node_embed=node_embed,
            edge_embed=edge_embed,
        )

        global_mp_type = (
            None
            if kwargs.get("global_mp_type") == "None"
            else kwargs.get("global_mp_type", "vn")
        )
        # self.recurrent == True: one layer
        # self.recurrent == False: num_layers different layers
        self.blocks = (
            torch.nn.ModuleList(
                [
                    MpnnLayer(
                        hidden_size=hidden_size,
                        local_gnn_type=kwargs.get("local_mp_type", "gin"),
                        global_model_type=(
                            None
                            if (global_mp_type == "vn" and i == num_layers - 1)
                            or (global_mp_type is None)
                            else global_mp_type
                        ),
                        dropout=dropout,
                        vn_norm_first=kwargs.get("vn_norm_first", True),
                        vn_norm_type=kwargs.get("vn_norm_type", "batchnorm"),
                        vn_pooling=kwargs.get("vn_pooling", "sum"),
                    )
                    for i in range(num_layers)
                ]
            )
            if not self.recurrent
            else MpnnLayer(
                hidden_size=hidden_size,
                local_gnn_type=kwargs.get("local_mp_type", "gin"),
                # TODO: not sure how to integrate virtual node in this case
                global_model_type=None,
                dropout=dropout,
                vn_norm_first=kwargs.get("vn_norm_first", True),
                vn_norm_type=kwargs.get("vn_norm_type", "batchnorm"),
                vn_pooling=kwargs.get("vn_pooling", "sum"),
            )
        )

        self.node_out = None
        if kwargs.get("node_out", True):
            if global_mp_type is None or global_mp_type == "vn":
                self.node_out = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.BatchNorm1d(hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_size, hidden_size),
                )

        if global_pool == "mean":
            self.global_pool = gnn.global_mean_pool
        elif global_pool == "sum":
            self.global_pool = gnn.global_add_pool
        else:
            self.global_pool = None

        self.out_head = PredictionHead(
            hidden_size=hidden_size,
            num_class=num_class,
            head=head,
            dropout=dropout,
        )

    def forward(self, batch):
        batch = self.feature_encoder(batch)
        self.masked_update(batch)

        h = batch.x
        if self.node_out is not None:
            h = self.node_out(h)

        # Readout
        if self.global_pool is not None:
            h = self.global_pool(h, batch.batch)

        # final prediction head (from hidden_size to num_class)
        return self.out_head(h, batch)

    def get_params(self):
        return self.parameters()

    def get_mask(self, num_nodes):
        """Training: returns a probabilistic mask tensor of size (num_nodes, 1)
            where each value is 1 with probability self.alpha, 0 otherwise
           Evaluation: depends on self.alpha_eval_flag
            'a': mask tensor of size (num_nodes, 1) containing alpha
            'p': probabilistic mask tensor of size (num_nodes, 1) (same as for training)
            'n': no mask during inference

        Args:
            num_nodes (int): Number of nodes in the batch
            alpha (float, optional): Probability of a node update. Defaults to 0.5.

        Returns:
            torch.tensor: (num_nodes, 1) mask to be multiplied with the updated batch
        """
        alphas = torch.full((num_nodes, 1), fill_value=self.alpha)
        if self.training:
            return torch.bernoulli(alphas).int()
        else:
            if self.alpha_eval_flag == "a":
                return alphas
            elif self.alpha_eval_flag == "p":
                # same procedure for training and inference
                return torch.bernoulli(alphas).int()
            elif self.alpha_eval_flag == "n":
                # no mask during inference
                return torch.full((num_nodes, 1), fill_value=1)
            else:
                raise RuntimeError(f"Unexpected alpha flag: {self.alpha_eval_flag}")

    def get_coloring(self, batch):
        # compute the coloring for a graph
        # return: colors (to loop through, one update per color), mask
        if not self.use_coloring:
            # coloring not used => return one-colored mask
            return {0}, np.zeros(batch.x.shape[0], dtype=int)
        if not batch.is_directed():
            # undirected graph
            g = nx.from_numpy_array(to_scipy_sparse_matrix(batch.edge_index).todense())
        else:
            # directed graph
            # TODO: test with directed dataset, e.g., MNIST, CIFAR10
            g = nx.from_numpy_array(
                to_scipy_sparse_matrix(batch.edge_index).todense(),
                create_using=nx.DiGraph,
            )
        # returns a dict node:color
        coloring = nx.coloring.greedy_color(g)
        # create color mask for faster updating
        color_mask = np.array([coloring[i] for i in range(batch.x.shape[0])])
        return set(coloring.values()), color_mask

    def masked_update(self, batch):
        # performs the update on this batch including all necessary masking
        # training: some old values, some new values
        # evaluation: depends on self.alpha_evaluation_flag
        # .to(device=batch.x.device)
        print("hello")
        mask = self.get_mask(batch.x.shape[0])
        colors, color_mask = self.get_coloring(batch)
        # TODO: store coloring in the batch to avoid recomputation
        if not self.recurrent:
            for block in self.blocks:
                # block() calls forward (after registering hooks), modifies batch in place
                # i.e., you should read this as "keep some values of the original batch,
                # update batch (by passing it through the layer) and keep some (mask) of the new values"
                batch.x = (1 - mask) * batch.x + mask * block(batch).x
                # TODO: coloring, loop through colors, combine color_mask with mask, do one update
                for color in colors:
                    combined_mask = (
                        torch.as_tensor(
                            np.logical_not(color_mask - color).astype(int)
                        ).unsqueeze(-1)
                        * mask
                    )
        else:
            # apply one layer recurrently
            for i in range(self.num_layers):
                batch.x = (1 - mask) * batch.x + mask * self.blocks(batch).x
