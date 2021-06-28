#include "includes.h"
extern "C"


__global__ void sigmoid_float(int n,int idx,float *dy,int incy,float *result) {
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
if(i >= idx && i % incy == 0)
result[i] =  1.0 / (1.0 + expf(-dy[i]));
}

}