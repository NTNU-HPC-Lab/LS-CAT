#include "includes.h"
__global__ void init(double* out, int n){
int gid = threadIdx.x + blockIdx.x * blockDim.x;
if(gid >= n) return ;
out[gid] = 0.0;
}