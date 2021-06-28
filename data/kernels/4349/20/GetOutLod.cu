#include "includes.h"
__global__ void GetOutLod(const size_t* num_erased, const size_t* in_lod, const size_t lod_len, size_t* out_lod0) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < lod_len) {
out_lod0[index] = in_lod[index] - num_erased[in_lod[index]];
}
}