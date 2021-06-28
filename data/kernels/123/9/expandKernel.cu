#include "includes.h"
__global__ void expandKernel(double* values, int n_original, int factor, double* expanded){
int tid0 = threadIdx.x + blockIdx.x*blockDim.x ;
int stride = blockDim.x*gridDim.x ;
for ( int tid = tid0 ; tid < n_original*factor ; tid += stride){
int idx = floor(double(tid)/factor) ;
expanded[tid] = values[idx] ;
}
}