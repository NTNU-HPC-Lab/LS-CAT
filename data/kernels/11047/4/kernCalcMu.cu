#include "includes.h"
__global__ void kernCalcMu( const size_t numPoints, const size_t pointDim, const double* X, const double* loggamma, const double* GammaK, double* dest ) {
// Assumes a 2D grid of 1024x1 1D blocks
int b = blockIdx.y * gridDim.x + blockIdx.x;
int i = b * blockDim.x + threadIdx.x;
if(i >= numPoints) {
return;
}

const double a = exp(loggamma[i]) / exp(*GammaK);
const double* x = & X[i * pointDim];
double* y = & dest[i * pointDim];

for(size_t i = 0; i < pointDim; ++i) {
y[i] = a * x[i];
}
}