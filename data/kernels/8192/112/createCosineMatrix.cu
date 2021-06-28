#include "includes.h"
__global__ void createCosineMatrix(float* matrix, int xsize){
int threadGlobalID = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
+ threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

int i;
for (i = 0; i < xsize; i++){
if (threadGlobalID == 0)
matrix[threadGlobalID + i * xsize] = 1 / sqrt((float)xsize);
else
matrix[threadGlobalID + i * xsize] = (sqrt((float)2 / xsize) * cos((PI * (2 * i + 1) * threadGlobalID) / (2 * xsize)));
}
}