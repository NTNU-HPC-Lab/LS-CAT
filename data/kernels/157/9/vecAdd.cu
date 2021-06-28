#include "includes.h"
__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
//@@ Insert code to implement vector addition here
int idx  = threadIdx.x + blockDim.x * blockIdx.x;
if (idx  < len) {
out[idx ] = in1[idx] + in2[idx];
}
}