#include "includes.h"
__global__ void OneHotKernel(const float* params, int64_t num_features, int embed_size, int batch_size, const int64_t* indices, float* ret) {
int tid = threadIdx.x, bid = blockIdx.x;

ret[bid * embed_size + tid] = params[(int64_t)indices[bid] * embed_size + tid];
}