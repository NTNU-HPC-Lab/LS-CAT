#include "includes.h"
__global__ void radd(float * resp, const float * res, float alpha) {

int idx = threadIdx.x + blockIdx.x*blockDim.x;

resp[idx] = (1 - alpha)*resp[idx] + alpha*res[idx];
}