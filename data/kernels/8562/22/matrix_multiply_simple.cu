#include "includes.h"
__global__ void matrix_multiply_simple(float *a, float *b, float *ab, size_t width)
{
//TODO: write the kernel to perform matrix a times b, store results into ab.
// width is the size of the square matrix along one dimension.
int row = blockIdx.y*blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if(row < width && col < width)
{
float pvalue = 0;
for(int k = 0; k < width; k++)
{
pvalue += a[row * width + k] * b[k * width +col];
}
ab[row * width + col] = pvalue;
}

}