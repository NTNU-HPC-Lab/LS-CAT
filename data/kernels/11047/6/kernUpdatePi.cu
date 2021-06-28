#include "includes.h"
__global__ void kernUpdatePi( const size_t numPoints, const size_t numComponents, double* logpi, double* Gamma ) {
int b = blockIdx.y * gridDim.x + blockIdx.x;
int comp = b * blockDim.x + threadIdx.x;
if(comp > numComponents) {
return;
}

__shared__ double A[1024];
A[comp] = logpi[comp] + log(Gamma[comp * numPoints]);
__syncthreads();

double sum = 0;
for(size_t k = 0; k < numComponents; ++k) {
sum += exp(A[k]);
}

logpi[comp] = A[comp] - log(sum);
}