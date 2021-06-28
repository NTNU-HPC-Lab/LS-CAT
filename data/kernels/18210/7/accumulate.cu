#include "includes.h"
__global__ void accumulate(float *da, float* ans_device, int N){
int bx = blockIdx.x;
int tx = threadIdx.x;
int idx = bx * blockDim.x + tx;
//printf("%d\n", idx);
for(int stride = N / 2; stride > 0; stride >>= 1){
if(idx < stride){
da[idx] = da[idx] + da[idx + stride];
}
__syncthreads();
}
if(idx == 0){
ans_device[0] = da[idx];
//printf("ans 0: %f\n", ans_device[0]);
}
}