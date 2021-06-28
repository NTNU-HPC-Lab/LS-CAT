#include "includes.h"
__global__ void cube(float * d_out, float * d_in){
// Todo: Fill in this function
int idx = threadIdx.x;
float f = d_in[idx];
d_out[idx] = f*f*f;
}