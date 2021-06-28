#include "includes.h"
__global__ void combine_im_kernel(const float *A, const float *B, float *C, int numElements)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;

/*
combines images for a joint histogram computation with the formula:
comb_im = B1*(im1 + im2*(B2-1))/(B1*B2 - 1)

for a joint histogram of 256: B1*B2 must equal 256
--> choose B1=B2=16
*/

float B1 = 16.0;
float B2 = 16.0;

if (i < numElements)
{
C[i] = B1*(A[i] + B[i] * (B2 - 1)) / (B1*B2 - 1);
}
}