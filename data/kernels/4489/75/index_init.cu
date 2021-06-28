#include "includes.h"
__global__ void index_init(int* out_data, int h, int w) {
int idx = threadIdx.x + blockIdx.x * blockDim.x;
for (int i = idx; i < h * w; i += blockDim.x * gridDim.x) {
int w_id = i % w;
out_data[i] = w_id;
}
}