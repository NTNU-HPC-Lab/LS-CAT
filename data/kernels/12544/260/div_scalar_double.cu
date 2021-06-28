#include "includes.h"
__global__ void div_scalar_double(int n, int idx,double dx,double *dy,int incy,double * result) {
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
if(i >= idx && i % incy == 0)
result[i] = dy[i] / dx;
}

}