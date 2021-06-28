#include "includes.h"
/*
* cuArraysPadding.cu
* Padding Utitilies for oversampling
*/


//padding zeros in the middle, move quads to corners
//for raw chunk data oversampling
//tested
__global__ void cuArraysR2C_kernel(float *image1, float2 *image2, int size)
{
int idx =  threadIdx.x + blockDim.x*blockIdx.x;
if(idx < size)
{
image2[idx].x = image1[idx];
image2[idx].y =  0.0f;
}
}