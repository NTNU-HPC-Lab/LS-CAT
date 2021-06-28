#include "includes.h"
__global__ void threshold_and_support(float *vec, int *support, const int n, const float T)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

if (xIndex < n) {
if (abs(vec[xIndex])<T) {
vec[xIndex] = 0.0f;
support[xIndex]=2;
}
}
}