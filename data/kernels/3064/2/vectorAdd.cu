#include "includes.h"
__global__ void vectorAdd(int* a, int* b, int* c, int n){
int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
if (tid < n){
c[tid] = a[tid] + b[tid];
}

}