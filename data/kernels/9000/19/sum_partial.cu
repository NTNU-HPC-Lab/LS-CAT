#include "includes.h"
__global__ void sum_partial(double4 *a, double4 *b, unsigned int nextsize){

unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

if(i >= nextsize)
return;

extern __shared__ double4 shaccelerations[];
double4 *shacc = (double4*) shaccelerations;
double4 myacc;

myacc = b[i];
shacc[threadIdx.x] = a[i];

myacc.x += shacc[threadIdx.x].x;
myacc.y += shacc[threadIdx.x].y;
myacc.z += shacc[threadIdx.x].z;

b[i] = myacc;

}