#include "includes.h"
__global__ void MyKernel(float* devPtr, size_t pitch, int width, int height)
{
for(int r=0; r<height; ++r){
float* row = (float*)((char*)devPtr + r * pitch);
for (int c = 0; c < width; ++c){
row[c] = 17.3;
}
}
}