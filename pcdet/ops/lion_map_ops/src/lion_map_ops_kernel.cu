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

__global__ void fill_win2flat_kernel(
    const int64_t* batch_ids,
    const int64_t* batch_start_indices,
    const int64_t* batch_start_indices_p,
    int64_t* win2flat,
    int total_valid
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_valid) {
        return;
    }

    int64_t batch_id = batch_ids[idx];
    int64_t offset = batch_start_indices_p[batch_id] - batch_start_indices[batch_id];
    win2flat[idx] = idx + offset;
}


__global__ void fill_flat2win_kernel(
    const int64_t* num_per_batch,
    const int64_t* num_per_batch_p,
    const int64_t* batch_start_indices,
    const int64_t* batch_start_indices_p,
    int64_t* flat2win,
    int batch_size,
    int group_size
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }

    int64_t valid_count = num_per_batch[batch_idx];
    int64_t padded_count = num_per_batch_p[batch_idx];
    int64_t start_v = batch_start_indices[batch_idx];
    int64_t start_p = batch_start_indices_p[batch_idx];
    int64_t end_p = batch_start_indices_p[batch_idx + 1];
    int64_t tail_len = padded_count - valid_count;

    for (int64_t i = 0; i < padded_count; ++i) {
        flat2win[start_p + i] = start_v + i;
    }

    if (tail_len <= 0 || valid_count <= 0) {
        return;
    }

    if (padded_count > group_size) {
        int64_t src_start = end_p - group_size - tail_len;
        int64_t dst_start = end_p - tail_len;
        for (int64_t i = 0; i < tail_len; ++i) {
            flat2win[dst_start + i] = flat2win[src_start + i];
        }
    } else {
        int64_t dst_start = end_p - tail_len;
        for (int64_t i = 0; i < tail_len; ++i) {
            flat2win[dst_start + i] = start_v + (i % valid_count);
        }
    }
}


void lion_build_group_mappings_launcher(
    const int64_t* batch_ids,
    int64_t* num_per_batch,
    int64_t* num_per_batch_p,
    int64_t* batch_start_indices,
    int64_t* batch_start_indices_p,
    int64_t* flat2win,
    int64_t* win2flat,
    int batch_size,
    int group_size,
    int64_t total_valid,
    int64_t total_padded
) {
    if (total_valid > 0) {
        dim3 win2flat_blocks(DIVUP(total_valid, THREADS_PER_BLOCK));
        dim3 win2flat_threads(THREADS_PER_BLOCK);
        fill_win2flat_kernel<<<win2flat_blocks, win2flat_threads>>>(
            batch_ids,
            batch_start_indices,
            batch_start_indices_p,
            win2flat,
            total_valid
        );
    }

    if (total_padded > 0) {
        dim3 flat2win_blocks(DIVUP(batch_size, THREADS_PER_BLOCK));
        dim3 flat2win_threads(THREADS_PER_BLOCK);
        fill_flat2win_kernel<<<flat2win_blocks, flat2win_threads>>>(
            num_per_batch,
            num_per_batch_p,
            batch_start_indices,
            batch_start_indices_p,
            flat2win,
            batch_size,
            group_size
        );
    }
}
