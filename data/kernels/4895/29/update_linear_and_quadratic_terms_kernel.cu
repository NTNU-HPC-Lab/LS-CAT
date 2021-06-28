#include "includes.h"
__global__ void update_linear_and_quadratic_terms_kernel( int32_t n, float prior_offset, float* cur_tot_weight, int32_t max_count, float* quadratic, float* linear) {
float val = 1.0f;
float cur_weight = *cur_tot_weight;

if (max_count > 0.0f) {
float new_scale = max((float)cur_weight, (float)max_count) / max_count;

float prior_scale_change = new_scale - 1.0f;
val += prior_scale_change;
}

for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
i += blockDim.x * gridDim.x) {
int32_t diag_idx = ((i + 1) * (i + 2) / 2) - 1;
quadratic[diag_idx] += val;
}

if (threadIdx.x == 0) {
linear[0] += val * prior_offset;
}
}