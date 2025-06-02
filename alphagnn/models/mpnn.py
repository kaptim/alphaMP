import torch
import torch_geometric.nn as gnn
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
        centrality_range=0,
        recurrent=False,
        use_coloring=False,
        **kwargs,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.num_layers = num_layers
        self.alpha = alpha
        # inference: a: use alpha values, p: bernoulli (as in training), n: nothing
        self.alpha_eval_flag = alpha_eval_flag
        # only use one layer multiple times
        self.recurrent = recurrent
        # use coloring to update iteratively
        self.use_coloring = use_coloring
        # alpha values in [alpha-centrality_range, alpha]; 0 => no centrality used
        self.centrality_range = centrality_range
        # set up running centrality min, max for min-max normalisation
        self.min_running = 1
        self.max_running = 0
        # change in_edge_dim for dummy attributes
        if in_edge_dim is None:
            in_edge_dim = 1

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

    def get_node_mask(self, batch):
        # return mask for the batch based on self.alpha and self.centrality_range
        # training: alpha is used as the probability for a bernoulli distribution
        # inference: depends on self.alpha_eval_flag
        if self.alpha > 1.0 or self.alpha < 0:
            raise RuntimeError(f"Alpha value should be in [0,1]")
        if self.centrality_range > self.alpha or self.centrality_range < 0:
            raise RuntimeError(f"Centrality range value should be in [0,alpha]")
        # update running min, max
        # TODO: clamping
        if batch.centrality.max() > self.max_running:
            self.max_running = batch.centrality.max()
        if batch.centrality.min() < self.min_running:
            self.min_running = batch.centrality.min()
        alphas = torch.full(
            (batch.x.shape[0], 1), fill_value=self.alpha - self.centrality_range
        ).to(device=batch.x.device)
        # adapt alphas using the normalised centrality information of the nodes
        alphas += (
            (batch.centrality - self.min_running)
            / (self.max_running - self.min_running)
        ).unsqueeze(-1) * self.centrality_range
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
                return torch.full((batch.x.shape[0], 1), fill_value=1).to(
                    device=batch.x.device
                )
            else:
                raise RuntimeError(f"Unexpected alpha flag: {self.alpha_eval_flag}")

    def get_edge_mask(self, batch, node_mask):
        # TODO: outdated, if you want to run this, need to update compare edge_mask implementation with the one in benchmarking-PEs
        # mask updates for edges which are between two masked nodes
        edge_mask = torch.empty(
            (batch.edge_index.shape[1], 1), device=batch.edge_attr.device
        )
        for i in range(batch.edge_index.shape[1]):
            edge_mask[i] = (
                node_mask[batch.edge_index[0][i]] | node_mask[batch.edge_index[1][i]]
            )
        return edge_mask.int()

    def color_update(self, batch):
        # update using coloring of the graph
        colors = batch.coloring.unique().tolist()
        if not self.recurrent:
            for block in self.blocks:
                mask = self.get_node_mask(batch)
                for color in colors:
                    # only update nodes with color=color in this iteration
                    node_combined_mask = (
                        torch.logical_not(batch.coloring - color).int().unsqueeze(-1)
                        * mask
                    )
                    edge_mask = self.get_edge_mask(batch, node_combined_mask)
                    batch_old_x = batch.x.detach().clone()
                    # block() calls forward (after registering hooks), modifies batch in place
                    # i.e., you should read this as "keep some values of the original batch,
                    # update batch (by passing it through the layer) and keep some (mask) of the new values"
                    batch.edge_attr = (
                        1 - edge_mask
                    ) * batch.edge_attr + edge_mask * block(batch).edge_attr
                    batch.x = (
                        1 - node_combined_mask
                    ) * batch_old_x + node_combined_mask * batch.x

        else:
            # apply one layer recurrently
            for i in range(self.num_layers):
                mask = self.get_node_mask(batch)
                for color in colors:
                    # only update nodes with color=color in this iteration
                    node_combined_mask = (
                        torch.logical_not(batch.coloring - color).int().unsqueeze(-1)
                        * mask
                    )
                    edge_mask = self.get_edge_mask(batch, node_combined_mask)
                    batch_old_x = batch.x.detach().clone()
                    # difference to above: self.blocks instead of block
                    batch.edge_attr = (
                        1 - edge_mask
                    ) * batch.edge_attr + edge_mask * self.blocks(batch).edge_attr
                    batch.x = (
                        1 - node_combined_mask
                    ) * batch_old_x + node_combined_mask * batch.x

    def masked_update(self, batch):
        # TODO: implement edge masking
        # performs the update on this batch including all necessary masking
        # training: some old values, some new values
        # evaluation: depends on self.alpha_evaluation_flag
        if self.use_coloring:
            self.color_update(batch)
            return
        if not self.recurrent:
            for block in self.blocks:
                node_mask = self.get_node_mask(batch)
                # edge_mask: only update edges where at least one of the end nodes is updated
                edge_mask = self.get_edge_mask(batch, node_mask)
                batch_old_x = batch.x.detach().clone()
                # block() calls forward (after registering hooks), modifies batch in place
                # i.e., you should read this as "keep some values of the original batch,
                # update batch (by passing it through the layer) and keep some (mask) of the new values"
                batch.edge_attr = (1 - edge_mask) * batch.edge_attr + edge_mask * block(
                    batch
                ).edge_attr
                batch.x = (1 - node_mask) * batch_old_x + node_mask * batch.x

        else:
            # apply one layer recurrently
            for i in range(self.num_layers):
                node_mask = self.get_node_mask(batch)
                edge_mask = self.get_edge_mask(batch, node_mask)
                batch_old_x = batch.x.detach().clone()
                # difference to above: self.blocks instead of block
                batch.edge_attr = (
                    1 - edge_mask
                ) * batch.edge_attr + edge_mask * self.blocks(batch).edge_attr
                batch.x = (1 - node_mask) * batch_old_x + node_mask * batch.x
