#include "includes.h"
__global__ void l1_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(i < n){
float diff = truth[i] - pred[i];
error[i] = abs(diff);
delta[i] = (diff > 0) ? 1 : -1;
}
}