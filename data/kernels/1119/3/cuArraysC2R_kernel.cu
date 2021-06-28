#include "includes.h"
/*
* cuArraysPadding.cu
* Padding Utitilies for oversampling
*/


//padding zeros in the middle, move quads to corners
//for raw chunk data oversampling
//tested
__global__ void cuArraysC2R_kernel(float2 *image1, float *image2, int size)
{
int idx =  threadIdx.x + blockDim.x*blockIdx.x;
if(idx < size)
{
image2[idx] = image1[idx].x;
}
}