#include "includes.h"

extern "C" {
}


__global__ void l2_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(i < n){
float diff = truth[i] - pred[i];
error[i] = diff * diff; //I know this is technically wrong, deal with it.
delta[i] = diff;
}
}