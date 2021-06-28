#include "includes.h"


__global__ void cube(float * d_out, float * d_in){
// Todo: Fill in this function
int i = threadIdx.x;
if(i<96){
d_out[i]=d_in[i]*d_in[i]*d_in[i];
}
}