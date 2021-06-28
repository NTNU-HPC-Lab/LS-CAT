#include "includes.h"
__global__ void convolution2D(const float *d_arr, const float *d_mask, float *d_result, int N) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;
int globalId = i*N + j;
if(i < N && j< N) {
float avgSum = 0;
int id, cnum = 0;
for(int p = i-1; p <= i+1; p++) {
for(int q = j-1; q<= j+1; q++) {
if(p >=0 && p < N && q>=0 && q < N) {
id = p*N + q;
avgSum += d_arr[id]*d_mask[cnum];
}
cnum++;
}
}
d_result[globalId] = avgSum;
}
}