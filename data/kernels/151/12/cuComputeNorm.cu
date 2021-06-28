#include "includes.h"
__global__ void cuComputeNorm(float *mat, int width, int pitch, int height, float *norm){
unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
if (xIndex<width){
float val, sum=0;
int i;
for (i=0;i<height;i++){
val  = mat[i*pitch+xIndex];
sum += val*val;
}
norm[xIndex] = sum;
}
}