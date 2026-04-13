#include <torch/serialize/tensor.h>
#include <torch/types.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define CHECK_CALL(call)                                                  \
do {                                                                      \
    const cudaError_t error_code = call;                                  \
    if (error_code != cudaSuccess) {                                      \
        printf("CUDA Error:\n");                                          \
        printf("    File:       %s\n", __FILE__);                         \
        printf("    Line:       %d\n", __LINE__);                         \
        printf("    Error code: %d\n", error_code);                       \
        printf("    Error text: %s\n", cudaGetErrorString(error_code));   \
        exit(1);                                                          \
    }                                                                     \
} while (0)

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void fill_axis_occupancy_kernel(
    const int64_t* contiguous_win_inds,
    const int64_t* coors_in_win,
    int64_t* occupancy_y,
    int64_t* occupancy_x,
    int n,
    int max_voxel,
    int win_shape_x,
    int win_shape_y,
    int win_shape_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    int64_t win_idx = contiguous_win_inds[idx];
    int64_t z = coors_in_win[idx * 3 + 0];
    int64_t y = coors_in_win[idx * 3 + 1];
    int64_t x = coors_in_win[idx * 3 + 2];

    int64_t axis_idx_y = win_idx * max_voxel + y * win_shape_x * win_shape_z + x * win_shape_z + z;
    int64_t axis_idx_x = win_idx * max_voxel + x * win_shape_y * win_shape_z + y * win_shape_z + z;

    occupancy_y[axis_idx_y] = idx;
    occupancy_x[axis_idx_x] = idx;
}


__global__ void compact_axis_occupancy_kernel(
    int64_t* occupancy,
    int win_num,
    int max_voxel
) {
    int win_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (win_idx >= win_num) {
        return;
    }

    int64_t base = static_cast<int64_t>(win_idx) * max_voxel;
    int write_idx = 0;
    for (int read_idx = 0; read_idx < max_voxel; ++read_idx) {
        int64_t value = occupancy[base + read_idx];
        if (value >= 0) {
            occupancy[base + write_idx] = value;
            ++write_idx;
        }
    }
}


__global__ void gather_set_voxel_inds_kernel(
    const int64_t* occupancy_y,
    const int64_t* occupancy_x,
    const int64_t* select_idx,
    int64_t* out_set_voxel_inds,
    int total_select
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_select) {
        return;
    }

    int64_t gather_idx = select_idx[idx];
    out_set_voxel_inds[idx] = occupancy_y[gather_idx];
    out_set_voxel_inds[total_select + idx] = occupancy_x[gather_idx];
}


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
) {
    int max_voxel = win_shape_x * win_shape_y * win_shape_z;
    int64_t occupancy_size = static_cast<int64_t>(win_num) * max_voxel;
    int total_select = set_num * voxel_num_set;

    int64_t* occupancy_y = nullptr;
    int64_t* occupancy_x = nullptr;
    CHECK_CALL(cudaMalloc(&occupancy_y, occupancy_size * sizeof(int64_t)));
    CHECK_CALL(cudaMalloc(&occupancy_x, occupancy_size * sizeof(int64_t)));
    CHECK_CALL(cudaMemset(occupancy_y, 0xFF, occupancy_size * sizeof(int64_t)));
    CHECK_CALL(cudaMemset(occupancy_x, 0xFF, occupancy_size * sizeof(int64_t)));

    dim3 fill_blocks(DIVUP(n, THREADS_PER_BLOCK));
    dim3 fill_threads(THREADS_PER_BLOCK);
    fill_axis_occupancy_kernel<<<fill_blocks, fill_threads>>>(
        contiguous_win_inds,
        coors_in_win,
        occupancy_y,
        occupancy_x,
        n,
        max_voxel,
        win_shape_x,
        win_shape_y,
        win_shape_z
    );

    dim3 compact_blocks(DIVUP(win_num, THREADS_PER_BLOCK));
    dim3 compact_threads(THREADS_PER_BLOCK);
    compact_axis_occupancy_kernel<<<compact_blocks, compact_threads>>>(occupancy_y, win_num, max_voxel);
    compact_axis_occupancy_kernel<<<compact_blocks, compact_threads>>>(occupancy_x, win_num, max_voxel);

    dim3 gather_blocks(DIVUP(total_select, THREADS_PER_BLOCK));
    dim3 gather_threads(THREADS_PER_BLOCK);
    gather_set_voxel_inds_kernel<<<gather_blocks, gather_threads>>>(
        occupancy_y,
        occupancy_x,
        select_idx,
        out_set_voxel_inds,
        total_select
    );

    cudaFree(occupancy_y);
    cudaFree(occupancy_x);
}


__global__ void build_packed_metadata_kernel(
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
) {
    int flat_row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = num_partitions * set_num;
    if (flat_row_idx >= total_rows) {
        return;
    }

    int partition_idx = flat_row_idx / set_num;
    int row_idx = flat_row_idx % set_num;
    int32_t write_offset = partition_offsets[partition_idx] + row_offsets[flat_row_idx];
    int64_t row_base = static_cast<int64_t>(flat_row_idx) * voxel_num_set;
    int64_t restore_base = static_cast<int64_t>(partition_idx) * total_voxel_num;
    for (int col_idx = 0; col_idx < voxel_num_set; ++col_idx) {
        int64_t flat_idx = row_base + col_idx;
        int64_t voxel_idx = set_voxel_inds[flat_idx];
        bool valid = (!set_voxel_mask[flat_idx]) && (voxel_idx >= 0);
        if (valid) {
            token_indices[write_offset] = voxel_idx;
            restore_perm_full[restore_base + voxel_idx] = write_offset;
            ++write_offset;
        }
    }
}


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
) {
    int total_rows = num_partitions * set_num;
    dim3 blocks(DIVUP(total_rows, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    build_packed_metadata_kernel<<<blocks, threads>>>(
        set_voxel_inds,
        set_voxel_mask,
        row_offsets,
        partition_offsets,
        token_indices,
        restore_perm_full,
        num_partitions,
        set_num,
        voxel_num_set,
        total_voxel_num
    );
}
