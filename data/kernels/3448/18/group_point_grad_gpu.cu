#include "includes.h"
__global__ void group_point_grad_gpu(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points) {
int index = threadIdx.x;
idx += m*nsample*index;
grad_out += m*nsample*c*index;
grad_points += n*c*index;

for (int j=0;j<m;++j) {
for (int k=0;k<nsample;++k) {
int ii = idx[j*nsample+k];
for (int l=0;l<c;++l) {
grad_points[ii*c+l] += grad_out[j*nsample*c+k*c+l];
}
}
}
}