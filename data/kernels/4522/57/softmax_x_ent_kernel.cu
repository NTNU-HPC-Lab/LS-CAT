#include "includes.h"
__global__ void softmax_x_ent_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(i < n){
float t = truth[i];
float p = pred[i];
error[i] = (t) ? -log(p) : 0;
delta[i] = t-p;
}
}