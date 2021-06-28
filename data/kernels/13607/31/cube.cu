#include "includes.h"
__global__ void cube(float * d_out, float * d_in){
int id = threadIdx.x;
float num = d_in[id];
d_out[id] = num * num * num;
}