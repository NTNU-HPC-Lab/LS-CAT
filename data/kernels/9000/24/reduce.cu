#include "includes.h"
__global__ void reduce(double4 *ac, double4 *ac1, double4 *ac2, unsigned int bf_real, unsigned int dimension){

unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int k = dimension*bf_real;
double4 myacc;

extern __shared__ double4 shaccelerations[];
double4 *shacc = (double4*) shaccelerations;

if(i < k){
myacc = ac[i];

shacc[threadIdx.x] = ac[i + k];

myacc.x += shacc[threadIdx.x].x;
myacc.y += shacc[threadIdx.x].y;
myacc.z += shacc[threadIdx.x].z;

ac[i] = myacc;
}
else if (i >= k && i < 2*k){
myacc = ac1[i - k];

shacc[threadIdx.x] = ac1[i];

myacc.x += shacc[threadIdx.x].x;
myacc.y += shacc[threadIdx.x].y;
myacc.z += shacc[threadIdx.x].z;

ac1[i - k] = myacc;
}
else {
myacc = ac2[i - 2*k];

shacc[threadIdx.x] = ac2[i - k];

myacc.x += shacc[threadIdx.x].x;
myacc.y += shacc[threadIdx.x].y;
myacc.z += shacc[threadIdx.x].z;

ac2[i - 2*k] = myacc;
}
}