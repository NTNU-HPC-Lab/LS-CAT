#include "includes.h"
__global__ void column_sum(const float* data, float* sum, int nx, int ny, int num_threads, int offset ) {

float s = 0.0;
const uint idx = threadIdx.x + blockIdx.x*num_threads+offset;
for(int i =0; i < ny; i++) {
s += data[idx + i*nx];
}
sum[idx] = s;
}