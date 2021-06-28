#include "includes.h"
__global__ void CrossVector(float *first , float *second) {

int idx = threadIdx.x + blockIdx.x * blockDim.x; // the element of the vector
first[idx] *= sqrtf(second[idx]);

}