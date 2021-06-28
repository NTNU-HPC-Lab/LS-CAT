#include "includes.h"
__global__ void trapz_kernel(float* y, float* x, float* auc, int num_selected) {
__shared__ float s_auc;
s_auc = 0.0f;
__syncthreads();
int gid_base = blockIdx.x * blockDim.x + threadIdx.x;
for (int gid = gid_base; gid < num_selected - 1; gid += blockDim.x * gridDim.x) {
float a = x[gid];
float b = x[gid + 1];
float fa = y[gid];
float fb = y[gid + 1];
float area = (b - a) * (fa + fb) / 2.0f;
if (gid == 0) {
area += (a * fa / 2.0f);
}
atomicAdd(&s_auc, area);
}
__syncthreads();
if (threadIdx.x == 0) {
atomicAdd(auc, s_auc);
}
}