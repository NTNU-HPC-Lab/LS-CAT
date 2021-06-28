#include "includes.h"
__global__ void gGetValueByKey(float* d_in, float* d_out, int* indeces, int n) {
int tid = threadIdx.x + blockDim.x * blockIdx.x;
if(tid < n) {
int index = indeces[tid];
d_out[tid] = d_in[index];
}
}