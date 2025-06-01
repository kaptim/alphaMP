import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import new_layer_config, BatchNorm1dNode
from torch_geometric.graphgym.register import register_network


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """

    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # print(register.node_encoder_dict)
            # Encode integer node features via nn.Embeddings
            # print(len(register.node_encoder_dict))
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_inner,
                        -1,
                        -1,
                        has_act=False,
                        has_bias=False,
                        cfg=cfg,
                    )
                )
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if "PNA" in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_edge, -1, -1, has_act=False, has_bias=False, cfg=cfg
                    )
                )

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network("GritTransformer")
class GritTransformer(torch.nn.Module):
    """
    The proposed GritTransformer (Graph Inductive Bias Transformer)
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        self.ablation = True
        self.ablation = False

        if cfg.posenc_RRWP.enable:
            self.rrwp_abs_encoder = register.node_encoder_dict["rrwp_linear"](
                cfg.posenc_RRWP.ksteps, cfg.gnn.dim_inner
            )
            rel_pe_dim = cfg.posenc_RRWP.ksteps
            self.rrwp_rel_encoder = register.edge_encoder_dict["rrwp_linear"](
                rel_pe_dim,
                cfg.gnn.dim_edge,
                pad_to_full_graph=cfg.gt.attn.full_attn,
                add_node_attr_as_self_loop=False,
                fill_value=0.0,
            )

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert (
            cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in
        ), "The inner and hidden dims must match."

        global_model_type = cfg.gt.get("layer_type", "GritTransformer")
        # global_model_type = "GritTransformer"

        TransformerLayer = register.layer_dict.get(global_model_type)

        layers = []
        for l in range(cfg.gt.layers):
            layers.append(
                TransformerLayer(
                    in_dim=cfg.gt.dim_hidden,
                    out_dim=cfg.gt.dim_hidden,
                    num_heads=cfg.gt.n_heads,
                    dropout=cfg.gt.dropout,
                    sparse=cfg.gt.sparse,
                    act=cfg.gnn.act,
                    attn_dropout=cfg.gt.attn_dropout,
                    layer_norm=cfg.gt.layer_norm,
                    batch_norm=cfg.gt.batch_norm,
                    residual=True,
                    norm_e=cfg.gt.attn.norm_e,
                    O_e=cfg.gt.attn.O_e,
                    cfg=cfg.gt,
                )
            )

        self.layers = torch.nn.Sequential(*layers)
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        # self.children(): torch function to return an iterator over child
        # modules, i.e., the feature encoder, the layers etc.
        for name, module in self.named_children():
            if name == "layers":
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
        for layer in self.layers:
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
        for layer in self.layers:
            mask = self.get_mask(batch)
            # layer() calls forward (after registering hooks), modifies batch in place
            # i.e., you should read this as "keep some values of the original batch,
            # update batch (by passing it through the layer) and keep some (mask) of the new values"
            print("viper")
            batch.x = (1 - mask) * batch.x + mask * layer(batch).x
