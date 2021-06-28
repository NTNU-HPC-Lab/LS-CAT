#include "includes.h"
__global__ void group_point_grad_gpu(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points) {
int batch_index = blockIdx.x;
idx += m*nsample*batch_index;
grad_out += m*nsample*c*batch_index;
grad_points += n*c*batch_index;

int index = threadIdx.x;
int stride = blockDim.x;

for (int j=index;j<m;j+=stride) {
for (int k=0;k<nsample;++k) {
int ii = idx[j*nsample+k];
for (int l=0;l<c;++l) {
// Use atomic add to avoid race condition
atomicAdd(&grad_points[ii*c+l], grad_out[j*nsample*c+k*c+l]);
}
}
}
}