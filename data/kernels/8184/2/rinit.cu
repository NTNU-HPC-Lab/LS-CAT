#include "includes.h"
__global__ void rinit(float * init, const unsigned int * fsum, const float * ncrs) {

int idx = threadIdx.x + blockIdx.x*blockDim.x;
init[idx] = sqrtf((float)fsum[idx] / ncrs[idx]);
}