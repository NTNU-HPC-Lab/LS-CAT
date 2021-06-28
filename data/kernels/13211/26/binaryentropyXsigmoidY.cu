#include "includes.h"
// filename: gaxpy2.cu
// a simple CUDA kernel to add two vectors

extern "C"   // ensure function name to be exactly "gaxpy2"
{
}
__global__ void binaryentropyXsigmoidY(const int lengthX, const double *x,  const double *y, double *z)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
if (i<lengthX)
{
z[i]=x[i]*log(x[i])+(1.0-x[i])*log(1.0-x[i])-x[i]*y[i]+log(1.0+exp(y[i]));
}
}