#include "includes.h"
__global__ void _adam32(int n, int t, double eps, double b1, double b2, float *fstm, float *scndm, float *dw) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
while (i < n) {
fstm[i] = b1*fstm[i] + (1-b1)*dw[i];
scndm[i] = b2*scndm[i] + (1-b2)*(dw[i] *dw[i]);
dw[i] = (fstm[i] / (1 - pow(b1,(double)t))) / (sqrt(scndm[i] / (1 - pow(b2,(double)t))) + eps);

i += blockDim.x * gridDim.x;
}
}