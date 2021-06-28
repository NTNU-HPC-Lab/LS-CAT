#include "includes.h"
// filename: vmult!.cu
// a simple CUDA kernel to element multiply two vectors C=alpha*A.*B

extern "C"   // ensure function name to be exactly "vmult!"
{
}
__global__ void vabs(const int n, const double *a, double *b)
{
int i =	 threadIdx.x + blockIdx.x * blockDim.x;
if (i<n)
{b[i]=fabs(a[i]);}
}