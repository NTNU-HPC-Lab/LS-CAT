#include "includes.h"
extern "C"
__global__ void abs_double(int n,int idx,double *dy,int incy,double *result) {
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
if(i >= idx && i % incy == 0)
result[i] =  abs(dy[i]);
}

}