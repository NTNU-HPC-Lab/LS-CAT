#include "includes.h"
__global__ void kernel(int* d_vec, int n) {

int tid = threadIdx.x;

if(threadIdx.x < n) {
int i = d_vec[tid];
d_vec[tid] = i > 5 ? -i : i;
}
}