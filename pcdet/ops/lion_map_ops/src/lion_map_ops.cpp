#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous")
#define CHECK_LONG(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Long, #x, " must be int64")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x); \
  CHECK_LONG(x)

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
);


std::vector<at::Tensor> lion_build_group_mappings_gpu(
    at::Tensor batch_ids,
    int batch_size,
    int group_size
) {
    CHECK_INPUT(batch_ids);
    TORCH_CHECK(batch_ids.dim() == 1, "batch_ids must have shape (N)");

    auto opts = torch::TensorOptions().dtype(batch_ids.dtype()).device(batch_ids.device());
    auto cpu_opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
    auto batch_ids_cpu = batch_ids.to(torch::kCPU);
    auto num_per_batch_cpu = torch::zeros({batch_size}, cpu_opts);
    auto num_per_batch_ptr = num_per_batch_cpu.data_ptr<int64_t>();
    auto batch_ids_ptr = batch_ids_cpu.data_ptr<int64_t>();
    int64_t n = batch_ids_cpu.size(0);
    for (int64_t i = 0; i < n; ++i) {
        int64_t batch_id = batch_ids_ptr[i];
        TORCH_CHECK(batch_id >= 0 && batch_id < batch_size, "batch id out of range");
        num_per_batch_ptr[batch_id] += 1;
    }

    auto num_per_batch_p_cpu = torch::zeros({batch_size}, cpu_opts);
    auto num_per_batch_p_ptr = num_per_batch_p_cpu.data_ptr<int64_t>();
    auto batch_start_indices_cpu = torch::zeros({batch_size + 1}, cpu_opts);
    auto batch_start_indices_p_cpu = torch::zeros({batch_size + 1}, cpu_opts);
    auto batch_start_ptr = batch_start_indices_cpu.data_ptr<int64_t>();
    auto batch_start_p_ptr = batch_start_indices_p_cpu.data_ptr<int64_t>();

    int64_t total_valid = 0;
    int64_t total_padded = 0;
    for (int i = 0; i < batch_size; ++i) {
        int64_t count = num_per_batch_ptr[i];
        int64_t padded = ((count + group_size - 1) / group_size) * group_size;
        num_per_batch_p_ptr[i] = padded;
        batch_start_ptr[i] = total_valid;
        batch_start_p_ptr[i] = total_padded;
        total_valid += count;
        total_padded += padded;
    }
    batch_start_ptr[batch_size] = total_valid;
    batch_start_p_ptr[batch_size] = total_padded;

    auto flat2win = torch::empty({total_padded}, opts);
    auto win2flat = torch::empty({total_valid}, opts);
    auto num_per_batch = num_per_batch_cpu.to(batch_ids.device());
    auto num_per_batch_p = num_per_batch_p_cpu.to(batch_ids.device());
    auto batch_start_indices = batch_start_indices_cpu.to(batch_ids.device());
    auto batch_start_indices_p = batch_start_indices_p_cpu.to(batch_ids.device());

    lion_build_group_mappings_launcher(
        batch_ids.data_ptr<int64_t>(),
        num_per_batch.data_ptr<int64_t>(),
        num_per_batch_p.data_ptr<int64_t>(),
        batch_start_indices.data_ptr<int64_t>(),
        batch_start_indices_p.data_ptr<int64_t>(),
        flat2win.data_ptr<int64_t>(),
        win2flat.data_ptr<int64_t>(),
        batch_size,
        group_size,
        total_valid,
        total_padded
    );

    return {flat2win, win2flat};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_group_mappings", &lion_build_group_mappings_gpu, "LION fused group mappings (CUDA)");
}
