#include "includes.h"
__global__ void addVector(int *d1_in, int *d2_in, int *d_out, int n){
int ind = blockDim.x*blockIdx.x + threadIdx.x;
if(ind<n){
d_out[ind] = d1_in[ind]+d2_in[ind];
}
}