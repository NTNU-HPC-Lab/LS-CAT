#include "includes.h"
__global__ void kernelBackprop3b(float *delta_nabla_b,int b_off,float *zs) {
delta_nabla_b[b_off+threadIdx.x]*=(1.0/(1.0+expf(-zs[threadIdx.x])))*(1.0-(1.0/(1.0+expf(-zs[threadIdx.x]))));
}