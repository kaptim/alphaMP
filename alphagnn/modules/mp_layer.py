from torch import nn
from .gnn_layers import get_gnn_layer
from .virtual_node_layer import VirtualNodeLayer


class MessagePassingLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        local_gnn_type="gin",
        global_model_type=None,
        dropout=0.0,
        vn_norm_first=True,
        vn_norm_type="batchnorm",
        vn_pooling="sum",
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.local_mp = get_gnn_layer(local_gnn_type, hidden_size, dropout)

        self.global_model_type = global_model_type
        self.global_mp = get_global_layer(
            global_model_type,
            hidden_size,
            dropout,
            vn_norm_first=vn_norm_first,
            vn_norm_type=vn_norm_type,
            vn_pooling=vn_pooling,
        )

    def forward(self, batch):
        h = batch.x
        edge_attr = batch.edge_attr

        if self.local_mp is not None:
            h, edge_attr = self.local_mp(h, batch.edge_index, edge_attr)

        if self.global_model_type is not None:
            if self.global_model_type == "vn":
                h = self.global_mp(h, batch)
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

        batch.x = h
        batch.edge_attr = edge_attr
        return batch


def get_global_layer(
    global_model_type,
    hidden_size,
    dropout,
    vn_norm_first=True,
    vn_norm_type="batchnorm",
    vn_pooling="sum",
):
    if global_model_type == "vn":
        return VirtualNodeLayer(
            hidden_size,
            dropout,
            norm_first=vn_norm_first,
            norm_type=vn_norm_type,
            pooling=vn_pooling,
        )
    else:
        return None
