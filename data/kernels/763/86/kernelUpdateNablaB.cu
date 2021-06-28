#include "includes.h"
__global__ void kernelUpdateNablaB(float *nabla_b,float *delta_nabla_b) {
nabla_b[threadIdx.x]+=delta_nabla_b[threadIdx.x];
}