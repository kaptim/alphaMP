from torch import nn
from .mp_layer import MessagePassingLayer


class MpnnLayer(nn.Module):
    """MPNN layer."""

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

        self.mp_layer = MessagePassingLayer(
            hidden_size=hidden_size,
            local_gnn_type=local_gnn_type,
            global_model_type=global_model_type,
            dropout=dropout,
            vn_norm_first=vn_norm_first,
            vn_norm_type=vn_norm_type,
            vn_pooling=vn_pooling,
        )

    def forward(self, batch):
        batch = self.mp_layer(batch)
        return batch
