import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network
from grit.layer.gatedgcn_layer import ResGatedGCNConvLayer
from grit.layer.gatedgcn_layer import GatedGCNLayer
from grit.layer.gine_conv_layer import GINEConvLayer
from grit.layer.gin_layer import GINConvLayer


@register_network("custom_gnn")
class CustomGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, "The inner and hidden dims must match."

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        layers = []
        for _ in range(cfg.gnn.layers_mp):
            layers.append(
                conv_model(
                    dim_in, dim_in, dropout=cfg.gnn.dropout, residual=cfg.gnn.residual
                )
            )
        self.gnn_layers = torch.nn.Sequential(*layers)

        # get the prediction head
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == "gatedgcnconv":
            return GatedGCNLayer
        elif model_type == "gineconv":
            return GINEConvLayer
        elif model_type == "resgatedgcnconv":
            return ResGatedGCNConvLayer
        elif model_type == "gin_conv":
            return GINConvLayer
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        # self.children(): torch function to return an iterator over child
        # modules, i.e., the feature encoder, the layers etc.
        for name, module in self.named_children():
            if name == "gnn_layers":
                # alpha-asynchronous updates of each layer
                self.masked_update(batch)
            else:
                batch = module(batch)
        return batch

    def get_mask(self, batch):
        # return mask for the batch based on cfg.model.alpha and cfg.model.centrality_range
        # training: alpha is used as the probability for a bernoulli distribution
        # inference: depends on cfg.model.alpha_eval_flag
        if cfg.model.alpha > 1.0 or cfg.model.alpha < 0:
            raise RuntimeError(f"Alpha value should be in [0,1]")
        if (
            cfg.model.centrality_range > cfg.model.alpha
            or cfg.model.centrality_range < 0
        ):
            raise RuntimeError(f"Centrality range value should be in [0,alpha]")
        alphas = torch.full(
            (batch.x.shape[0], 1),
            fill_value=cfg.model.alpha - cfg.model.centrality_range,
        ).to(device=batch.x.device)
        # adapt alphas using the normalised centrality information of the nodes
        # clamp centrality values to ensure that the normalised values are in [0,1]
        alphas += (
            (
                torch.clamp(
                    batch.centrality,
                    min=cfg.dataset.centrality_min,
                    max=cfg.dataset.centrality_max,
                )
                - cfg.dataset.centrality_min
            )
            / (cfg.dataset.centrality_max - cfg.dataset.centrality_min)
        ).unsqueeze(-1) * cfg.model.centrality_range
        if self.training:
            return torch.bernoulli(alphas).int()
        else:
            if cfg.model.alpha_eval_flag == "a":
                return alphas
            elif cfg.model.alpha_eval_flag == "p":
                # same procedure for training and inference
                return torch.bernoulli(alphas).int()
            elif cfg.model.alpha_eval_flag == "n":
                # no mask during inference
                return torch.full((batch.x.shape[0], 1), fill_value=1).to(
                    device=batch.x.device
                )
            else:
                raise RuntimeError(
                    f"Unexpected alpha flag: {cfg.model.alpha_eval_flag}"
                )

    def color_update(self, batch):
        # update using coloring of the graph
        colors = batch.coloring.unique().tolist()
        for layer in self.gnn_layers:
            mask = self.get_mask(batch)
            for color in colors:
                # only update nodes with color=color in this iteration
                combined_mask = (
                    torch.logical_not(batch.coloring - color).int().unsqueeze(-1) * mask
                )
                # layer() calls forward (after registering hooks), modifies batch in place
                # i.e., you should read this as "keep some values of the original batch,
                # update batch (by passing it through the layer) and keep some (mask) of the new values"
                batch.x = (1 - combined_mask) * batch.x + combined_mask * layer(batch).x

    def masked_update(self, batch):
        # performs the update on this batch including all necessary masking
        # evaluation: depends on alpha_evaluation_flag
        if cfg.model.use_coloring:
            self.color_update(batch)
            return
        for layer in self.gnn_layers:
            mask = self.get_mask(batch)
            print("viper")
            # layer() calls forward (after registering hooks), modifies batch in place
            # i.e., you should read this as "keep some values of the original batch,
            # update batch (by passing it through the layer) and keep some (mask) of the new values"
            batch.x = (1 - mask) * batch.x + mask * layer(batch).x
