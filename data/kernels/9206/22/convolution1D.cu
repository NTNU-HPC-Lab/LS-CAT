#include "includes.h"
__global__ void convolution1D(const int *d_arr, const int *d_conv, int *d_result, int N, int M) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;
int globalId = i*N + j;
if(globalId < N) {
int convSum = 0, cnum = 0, k = M/2;
for(int i=-k; i<=k; i++) {
if(globalId + i >= 0 && globalId + i < N && cnum < M) {
convSum += d_arr[globalId + i]*d_conv[cnum];
}
cnum++;
}
d_result[globalId] = convSum;
}
}