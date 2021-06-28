#include "includes.h"
__global__ void add(int *fData, int *sData, int *oData, int x, int y){

int index = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;

for(int i = index; i < x*y; i += stride)
{
oData[i] = fData[i] + sData[i];
}
}