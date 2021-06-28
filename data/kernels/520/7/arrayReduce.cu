#include "includes.h"
__global__ void arrayReduce(int *m, int *ms){
int id = threadIdx.x + blockIdx.x * blockDim.x;
if (m[id] > -1)
m[id] = m[id] - ms[blockIdx.x];
}