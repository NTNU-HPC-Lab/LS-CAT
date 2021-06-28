#include "includes.h"
// filename: vmult!.cu
// a simple CUDA kernel to element multiply two vectors C=alpha*A.*B

extern "C"   // ensure function name to be exactly "vmultbang"
{
}
__global__ void binaryentropy(const int lengthX, const double *x,  const double *y, double *z)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
if (i<lengthX)
{
z[i] = x[i]*log(x[i]/y[i])+ (1.0-x[i])*log((1.0-x[i])/(1.0-y[i]));
}
}