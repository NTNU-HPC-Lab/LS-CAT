#include "includes.h"
__global__ void rdiv(float * res, const unsigned int * fsum, const float * csum) {

int idx = threadIdx.x + blockIdx.x*blockDim.x;
res[idx] = (float)fsum[idx] / csum[idx];
}