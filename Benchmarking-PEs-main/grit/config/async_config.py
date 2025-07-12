from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("async")
def set_cfg_posenc(cfg):
    """Extend configuration with asynchronous update options."""

    cfg.async_update = CN()

    cfg.async_update.metric = None  # centrality, coloring, degree
    cfg.async_update.metric_min = 0
    cfg.async_update.metric_max = 1.0
    cfg.async_update.metric_range = 0.0  # 0: no metric information considered
    cfg.async_update.metric_pos = True  # correlation of alpha and this metric (True: higher value of the metric => higher alpha value)

    # alpha itself
    cfg.async_update.alpha = (
        1.0  # synchronous: 1.0, higher alpha -> update more nodes in one iteration
    )
    cfg.async_update.alpha_node_flag = "a"  # inference (nodes): a: use alpha values, p: bernoulli (as in training), n: synchronous update
    cfg.async_update.alpha_edge_flag = "a"  # inference (edges): a: average of end nodes, m: maximum alpha of the two end nodes
