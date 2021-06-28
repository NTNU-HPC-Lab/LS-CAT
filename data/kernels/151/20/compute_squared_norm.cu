#include "includes.h"
__global__ void compute_squared_norm(float * array, int width, int pitch, int height, float * norm){
unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
if (xIndex<width){
float sum = 0.f;
for (int i=0; i<height; i++){
float val = array[i*pitch+xIndex];
sum += val*val;
}
norm[xIndex] = sum;
}
}