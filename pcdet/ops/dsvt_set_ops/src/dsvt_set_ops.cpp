#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous")
#define CHECK_LONG(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Long, #x, " must be int64")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x); \
  CHECK_LONG(x)

void dsvt_get_set_single_shift_launcher(
    const int64_t* contiguous_win_inds,
    const int64_t* coors_in_win,
    const int64_t* select_idx,
    int64_t* out_set_voxel_inds,
    int n,
    int set_num,
    int voxel_num_set,
    int win_num,
    int win_shape_x,
    int win_shape_y,
    int win_shape_z
);

void dsvt_build_packed_metadata_launcher(
    const int64_t* set_voxel_inds,
    const bool* set_voxel_mask,
    const int32_t* row_offsets,
    const int32_t* partition_offsets,
    int64_t* token_indices,
    int64_t* restore_perm_full,
    int num_partitions,
    int set_num,
    int voxel_num_set,
    int total_voxel_num
);


at::Tensor dsvt_get_set_single_shift_gpu(
    at::Tensor contiguous_win_inds,
    at::Tensor coors_in_win,
    at::Tensor select_idx,
    int win_num,
    int win_shape_x,
    int win_shape_y,
    int win_shape_z
) {
    CHECK_INPUT(contiguous_win_inds);
    CHECK_INPUT(coors_in_win);
    CHECK_INPUT(select_idx);
    TORCH_CHECK(coors_in_win.dim() == 2 && coors_in_win.size(1) == 3, "coors_in_win must have shape (N, 3)");
    TORCH_CHECK(select_idx.dim() == 2, "select_idx must have shape (set_num, voxel_num_set)");

    int n = contiguous_win_inds.size(0);
    int set_num = select_idx.size(0);
    int voxel_num_set = select_idx.size(1);

    auto out = torch::full(
        {2, set_num, voxel_num_set},
        -1,
        torch::TensorOptions().dtype(select_idx.dtype()).device(select_idx.device())
    );

    dsvt_get_set_single_shift_launcher(
        contiguous_win_inds.data_ptr<int64_t>(),
        coors_in_win.data_ptr<int64_t>(),
        select_idx.data_ptr<int64_t>(),
        out.data_ptr<int64_t>(),
        n,
        set_num,
        voxel_num_set,
        win_num,
        win_shape_x,
        win_shape_y,
        win_shape_z
    );

    return out;
}


std::vector<at::Tensor> dsvt_build_packed_metadata_gpu(
    at::Tensor set_voxel_inds,
    at::Tensor set_voxel_mask,
    at::Tensor row_offsets,
    at::Tensor partition_offsets,
    int total_token_num,
    int total_voxel_num
) {
    CHECK_INPUT(set_voxel_inds);
    CHECK_CUDA(set_voxel_mask);
    CHECK_CONTIGUOUS(set_voxel_mask);
    TORCH_CHECK(set_voxel_mask.scalar_type() == at::ScalarType::Bool, "set_voxel_mask must be bool");
    TORCH_CHECK(set_voxel_inds.dim() == 3, "set_voxel_inds must have shape (num_partitions, set_num, voxel_num_set)");
    TORCH_CHECK(set_voxel_mask.sizes() == set_voxel_inds.sizes(), "set_voxel_mask must match set_voxel_inds");
    TORCH_CHECK(row_offsets.scalar_type() == at::ScalarType::Int, "row_offsets must be int32");
    TORCH_CHECK(partition_offsets.scalar_type() == at::ScalarType::Int, "partition_offsets must be int32");
    TORCH_CHECK(row_offsets.dim() == 2, "row_offsets must have shape (num_partitions, set_num)");
    TORCH_CHECK(partition_offsets.dim() == 1, "partition_offsets must have shape (num_partitions)");
    CHECK_CUDA(row_offsets);
    CHECK_CONTIGUOUS(row_offsets);
    CHECK_CUDA(partition_offsets);
    CHECK_CONTIGUOUS(partition_offsets);

    int num_partitions = set_voxel_inds.size(0);
    int set_num = set_voxel_inds.size(1);
    int voxel_num_set = set_voxel_inds.size(2);
    auto restore_perm_full = torch::full(
        {num_partitions, total_voxel_num},
        -1,
        torch::TensorOptions().dtype(set_voxel_inds.dtype()).device(set_voxel_inds.device())
    );
    auto token_indices = torch::full(
        {total_token_num},
        -1,
        torch::TensorOptions().dtype(set_voxel_inds.dtype()).device(set_voxel_inds.device())
    );

    dsvt_build_packed_metadata_launcher(
        set_voxel_inds.data_ptr<int64_t>(),
        set_voxel_mask.data_ptr<bool>(),
        row_offsets.data_ptr<int32_t>(),
        partition_offsets.data_ptr<int32_t>(),
        token_indices.data_ptr<int64_t>(),
        restore_perm_full.data_ptr<int64_t>(),
        num_partitions,
        set_num,
        voxel_num_set,
        total_voxel_num
    );

    return {token_indices, restore_perm_full};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &dsvt_get_set_single_shift_gpu, "DSVT fused get_set_single_shift (CUDA)");
    m.def("build_packed_metadata", &dsvt_build_packed_metadata_gpu, "DSVT fused packed metadata batched (CUDA)");
}
