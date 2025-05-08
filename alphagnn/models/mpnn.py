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
        self.alpha = alpha
        # inference: a: use alpha values, p: bernoulli (as in training), n: nothing
        self.alpha_eval_flag = alpha_eval_flag
        # only use one layer multiple times
        self.recurrent = recurrent
        # use coloring to update iteratively
        self.use_coloring = use_coloring
        # alpha values in [alpha-centrality_range, alpha], 0 => no centrality used
        self.centrality_range = centrality_range
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

    def get_mask(self, batch):
        # return mask for the batch based on self.alpha and self.centrality_range
        # training: alpha is used as the probability for a bernoulli distribution
        # inference: depends on self.alpha_eval_flag
        alphas = torch.full(
            (batch.x.shape[0], 1), fill_value=self.alpha - self.centrality_range
        ).to(device=batch.x.device)
        if alphas.max() + self.centrality_range * batch.centrality.max() > 1:
            raise RuntimeError(
                f"Alpha value too high: {self.alpha + self.centrality_range * batch.centrality.max()}"
            )
        # adapt alphas using the centrality information of the nodes
        alphas += batch.centrality.unsqueeze(-1) * self.centrality_range
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

    def color_update(self, batch, mask):
        # update using coloring of the graph
        colors = batch.coloring.unique().tolist()
        if not self.recurrent:
            for block in self.blocks:
                for color in colors:
                    # only update nodes with color=color in this iteration
                    combined_mask = (
                        torch.logical_not(batch.coloring - color).int().unsqueeze(-1)
                        * mask
                    )
                    # block() calls forward (after registering hooks), modifies batch in place
                    # i.e., you should read this as "keep some values of the original batch,
                    # update batch (by passing it through the layer) and keep some (mask) of the new values"
                    batch.x = (1 - combined_mask) * batch.x + combined_mask * block(
                        batch
                    ).x

        else:
            # apply one layer recurrently
            for i in range(self.num_layers):
                for color in colors:
                    # only update nodes with color=color in this iteration
                    combined_mask = (
                        torch.logical_not(batch.coloring - color).int().unsqueeze(-1)
                        * mask
                    )
                    # difference to above: self.blocks instead of block
                    batch.x = (
                        1 - combined_mask
                    ) * batch.x + combined_mask * self.blocks(batch).x

    def masked_update(self, batch):
        # performs the update on this batch including all necessary masking
        # training: some old values, some new values
        # evaluation: depends on self.alpha_evaluation_flag
        mask = self.get_mask(batch)
        if self.use_coloring:
            self.color_update(batch, mask)
            return
        if not self.recurrent:
            for block in self.blocks:
                # block() calls forward (after registering hooks), modifies batch in place
                # i.e., you should read this as "keep some values of the original batch,
                # update batch (by passing it through the layer) and keep some (mask) of the new values"
                batch.x = (1 - mask) * batch.x + mask * block(batch).x

        else:
            # apply one layer recurrently
            for i in range(self.num_layers):
                # difference to above: self.blocks instead of block
                batch.x = (1 - mask) * batch.x + mask * self.blocks(batch).x
