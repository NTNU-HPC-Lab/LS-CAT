#include "includes.h"
__global__ void kernelUpdateNablaW(float *nabla_w,float *delta_nabla_w,int tws) {
if ((blockIdx.x*blockDim.x+threadIdx.x)<tws) {
nabla_w[blockIdx.x*blockDim.x+threadIdx.x]+=delta_nabla_w[blockIdx.x*blockDim.x+threadIdx.x];
}
}