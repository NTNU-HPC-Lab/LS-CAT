#include "includes.h"
__global__ void setTensorCheckPatternKernel(unsigned int* data, unsigned int ndata) {
for (unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;i < ndata;i += blockDim.x*gridDim.x) {
data[i] = i;
}
}