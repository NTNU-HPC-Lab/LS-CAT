#include "includes.h"
__global__ void kernelInitNablaW(float *nabla_w,int tws) {
if ((blockIdx.x*blockDim.x+threadIdx.x)<tws) {
nabla_w[blockIdx.x*blockDim.x+threadIdx.x]=0.0;
}
}