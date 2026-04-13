import torch

try:
    from . import dsvt_set_ops_cuda
except ImportError:
    dsvt_set_ops_cuda = None


def fused_get_set_single_shift(contiguous_win_inds, coors_in_win, select_idx, win_num, window_shape):
    if dsvt_set_ops_cuda is None:
        raise ImportError('dsvt_set_ops_cuda is not built')

    if len(window_shape) == 2:
        win_shape_x, win_shape_y = window_shape
        win_shape_z = 1
    else:
        win_shape_x, win_shape_y, win_shape_z = window_shape

    return dsvt_set_ops_cuda.forward(
        contiguous_win_inds.contiguous(),
        coors_in_win.contiguous(),
        select_idx.contiguous(),
        int(win_num),
        int(win_shape_x),
        int(win_shape_y),
        int(win_shape_z)
    )


def fused_build_packed_metadata(set_voxel_inds, set_voxel_mask, row_offsets, partition_offsets, total_token_num, total_voxel_num):
    if dsvt_set_ops_cuda is None:
        raise ImportError('dsvt_set_ops_cuda is not built')

    return dsvt_set_ops_cuda.build_packed_metadata(
        set_voxel_inds.contiguous(),
        set_voxel_mask.contiguous(),
        row_offsets.contiguous(),
        partition_offsets.contiguous(),
        int(total_token_num),
        int(total_voxel_num)
    )
