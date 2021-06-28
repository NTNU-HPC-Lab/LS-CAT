#include "includes.h"



cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);





__global__ void shmem_scan(float* d_out, float* d_in) {
extern __shared__ float sdata[];
int idx = threadIdx.x;
float out = 0.00f;
sdata[idx] = d_in[idx];
__syncthreads();
for (int interpre = 1; interpre < sizeof(d_in); interpre *= 2) {
if (idx - interpre >= 0) {
out = sdata[idx] + sdata[idx - interpre];
}
__syncthreads();
if (idx - interpre >= 0) {
sdata[idx] = out;
out = 0.00f;
}
}
d_out[idx] = sdata[idx];
}