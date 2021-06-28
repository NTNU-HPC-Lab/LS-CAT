#include "includes.h"
__global__ void wgan_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(i < n){
error[i] = truth[i] ? -pred[i] : pred[i];
delta[i] = (truth[i] > 0) ? 1 : -1;
}
}