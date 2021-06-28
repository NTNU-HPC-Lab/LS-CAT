#include "includes.h"


__global__ void cube(float * d_out, float * d_in){
// Todo: Fill in this function
int index = threadIdx.x;
float f = d_in[index];
d_out[index] = f * f * f;
}