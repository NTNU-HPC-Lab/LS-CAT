#include "includes.h"
__global__ void NormalizeOutput(const int num_elements, const int* original, int64_t* to_normalize, int64_t batch_index, int64_t class_index) {
for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements; idx += blockDim.x * gridDim.x) {
to_normalize[idx * 3] = batch_index;
to_normalize[idx * 3 + 1] = class_index;
to_normalize[idx * 3 + 2] = static_cast<int64_t>(original[idx]);
}
}