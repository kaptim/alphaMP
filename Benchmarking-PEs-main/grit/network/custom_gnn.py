import json
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

        # get the min and max values for custom metrics
        with open(dataset_dir + "/min_max.json", "r") as f:
            self.min_max_dict = json.load(f)

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

    def get_node_mask(self, batch):
        # return mask for the batch based on cfg.async_update.alpha and cfg.async_update.metric_range
        # training: alpha is used as the probability for a bernoulli distribution
        # inference: depends on cfg.async_update.alpha_node_flag
        if cfg.async_update.alpha > 1.0 or cfg.async_update.alpha < 0:
            raise RuntimeError(f"Alpha value should be in [0,1]")
        if (
            cfg.async_update.metric_range > cfg.async_update.alpha
            or cfg.async_update.metric_range < 0
            or (cfg.async_update.metric_range != 0 and cfg.async_update.metric is None)
        ):
            raise RuntimeError(f"Metric range value should be in [0,alpha]")
        alphas = torch.full(
            (batch.x.shape[0], 1),
            fill_value=(
                cfg.async_update.alpha - cfg.async_update.metric_range
                if cfg.async_update.metric_pos
                else cfg.async_update.alpha
            ),
            device=batch.x.device,
        )
        # adapt alphas using the normalised metric information of the nodes
        # clamp metric values to ensure that the normalised values are in [0,1]
        print("hu?")
        if cfg.async_update.metric is not None:
            normalised_metric = (
                (
                    torch.clamp(
                        batch.get(cfg.async_update.metric),
                        min=cfg.async_update.metric_min,
                        max=cfg.async_update.metric_max,
                    )
                    - cfg.async_update.metric_min
                )
                / (cfg.async_update.metric_max - cfg.async_update.metric_min)
            ).unsqueeze(-1) * cfg.async_update.metric_range
            alphas += (
                normalised_metric if cfg.async_update.metric_pos else -normalised_metric
            )
        if self.training:
            return torch.bernoulli(alphas).int()
        else:
            if cfg.async_update.alpha_node_flag == "a":
                return alphas
            elif cfg.async_update.alpha_node_flag == "p":
                # same procedure for training and inference
                return torch.bernoulli(alphas).int()
            elif cfg.async_update.alpha_node_flag == "n":
                # no mask during inference
                return torch.full(
                    (batch.x.shape[0], 1), fill_value=1, device=batch.x.device
                )
            else:
                raise RuntimeError(
                    f"Unexpected node alpha flag: {cfg.async_update.alpha_node_flag}"
                )

    def get_edge_mask(self, batch, node_mask):
        # mask updates for edges which are between two masked nodes
        if self.training or cfg.async_update.alpha_node_flag == "p":
            return torch.logical_or(
                node_mask[batch.edge_index[0]], node_mask[batch.edge_index[1]]
            ).int()
        else:
            if cfg.async_update.alpha_node_flag == "n":
                # all nodes are used => all edges are used (in the same way)
                return torch.full(
                    (batch.edge_index.shape[1], 1),
                    fill_value=1,
                    device=batch.x.device,
                )
            elif cfg.async_update.alpha_node_flag == "a":
                if cfg.async_update.metric_range == 0:
                    return torch.full(
                        (batch.edge_index.shape[1], 1),
                        fill_value=cfg.async_update.alpha,
                        device=batch.x.device,
                    )
                # metric considered => need to handle different alpha values at the endpoints
                if cfg.async_update.alpha_edge_flag == "a":
                    return (
                        node_mask[batch.edge_index[0]] + node_mask[batch.edge_index[1]]
                    ) / 2
                elif cfg.async_update.alpha_edge_flag == "m":
                    return torch.maximum(
                        node_mask[batch.edge_index[0]], node_mask[batch.edge_index[1]]
                    )
                else:
                    raise RuntimeError(
                        f"Unexpected edge alpha flag: {cfg.async_update.alpha_edge_flag}"
                    )

    def color_update(self, batch):
        # update using coloring of the graph
        colors = batch.coloring.unique().tolist()
        for layer in self.gnn_layers:
            mask = self.get_node_mask(batch)
            for color in colors:
                # only update nodes with color=color in this iteration
                node_combined_mask = (
                    torch.logical_not(batch.coloring - color).int().unsqueeze(-1) * mask
                )
                edge_mask = self.get_edge_mask(batch, node_combined_mask)
                batch_old_x = batch.x.detach().clone()
                # layer() calls forward (after registering hooks), modifies batch in place
                # i.e., you should read this as "keep some values of the original batch,
                # update batch (by passing it through the layer) and keep some (mask) of the new values"
                batch.edge_attr = (1 - edge_mask) * batch.edge_attr + edge_mask * layer(
                    batch
                ).edge_attr
                batch.x = (
                    1 - node_combined_mask
                ) * batch_old_x + node_combined_mask * batch.x

    def masked_update(self, batch):
        # performs the update on this batch including all necessary masking
        # evaluation: depends on alpha_node_flag
        if cfg.async_update.use_coloring:
            self.color_update(batch)
            return
        for layer in self.gnn_layers:
            node_mask = self.get_node_mask(batch)
            # edge_mask: only update edges where at least one of the end nodes is updated
            edge_mask = self.get_edge_mask(batch, node_mask)
            batch_old_x = batch.x.detach().clone()
            # layer() calls forward (after registering hooks), modifies batch in place
            # i.e., you should read this as "keep some values of the original batch,
            # update batch (by passing it through the layer) and keep some (mask) of the new values"
            batch.edge_attr = (1 - edge_mask) * batch.edge_attr + edge_mask * layer(
                batch
            ).edge_attr
            batch.x = (1 - node_mask) * batch_old_x + node_mask * batch.x
