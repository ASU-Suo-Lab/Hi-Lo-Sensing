import torch

try:
    from . import lion_map_ops_cuda
except ImportError:
    lion_map_ops_cuda = None


def fused_build_group_mappings(batch_ids: torch.Tensor, batch_size: int, group_size: int):
    if lion_map_ops_cuda is None:
        raise ImportError('lion_map_ops_cuda is not built')

    return lion_map_ops_cuda.build_group_mappings(
        batch_ids.contiguous(),
        int(batch_size),
        int(group_size)
    )
