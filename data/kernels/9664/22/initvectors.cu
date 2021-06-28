#include "includes.h"
__global__ void initvectors(double4 *acc3, float4 *apred){
int i = blockIdx.x*blockDim.x + threadIdx.x;
acc3[i].x = acc3[i].y = acc3[i].z = 0.0;
apred[i].x = apred[i].y = apred[i].z = 0.0f;
}