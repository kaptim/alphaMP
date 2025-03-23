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
        **kwargs,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.alpha = alpha

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
        print("hello")
        self.blocks = torch.nn.ModuleList(
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
                    num_heads=kwargs.get("num_heads", 4),
                    dropout=dropout,
                    attn_dropout=kwargs.get("attn_dropout", 0.0),
                    vn_norm_first=kwargs.get("vn_norm_first", True),
                    vn_norm_type=kwargs.get("vn_norm_type", "batchnorm"),
                    vn_pooling=kwargs.get("vn_pooling", "sum"),
                )
                for i in range(num_layers)
            ]
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

        for i, block in enumerate(self.blocks):
            # TODO: alpha
            # block() basically calls forward (after registering hooks)
            # first idea: keep old batch, compute mask, only change
            # subset of the batch
            # !!! different number of features per node => watch out
            # (layers might rely on this)
            # only need a mask for the rows
            mask = self.get_mask(batch.x.shape[0]).to(device=batch.x.device)
            batch.x = (1 - mask) * batch.x + mask * block(batch).x

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
        """Returns a probabilistic mask tensor (num_nodes, 1)

        Args:
            num_nodes (int): Number of nodes in the batch
            alpha (float, optional): Probability of a node update. Defaults to 0.5.

        Returns:
            torch.tensor: (num_nodes, 1) mask to be multiplied with the updated batch
        """
        alphas = torch.full((num_nodes, 1), fill_value=self.alpha)
        return torch.bernoulli(alphas).int()
