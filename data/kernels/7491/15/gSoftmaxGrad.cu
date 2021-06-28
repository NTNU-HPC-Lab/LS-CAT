#include "includes.h"
__global__ void gSoftmaxGrad(float* grad, const float* adj, const float* val, const int rows, const int cols) {
for(int bid = 0; bid < rows; bid += gridDim.x) {
int j = bid + blockIdx.x;
if(j < rows) {
extern __shared__ float _share[];
float* _sum = _share + blockDim.x;

float* gradRow = grad + j * cols;
const float* adjRow = adj + j * cols;
const float* valRow = val + j * cols;
_sum[threadIdx.x] = 0.0;
for(int tid = 0; tid < cols; tid += blockDim.x) {
int id = tid + threadIdx.x;
if(id < cols) {
_sum[threadIdx.x] += valRow[id] * adjRow[id];
}
}
__syncthreads();
int len = blockDim.x;
while(len != 1) {
__syncthreads();
int skip = (len + 1) >> 1;
if(threadIdx.x < (len >> 1))
_sum[threadIdx.x] += _sum[threadIdx.x + skip];
len = (len + 1) >> 1;
}
__syncthreads();
for(int tid = 0; tid < cols; tid += blockDim.x) {
int id = tid + threadIdx.x;
if(id < cols) {
float val = valRow[id] * (adjRow[id] - _sum[0]);
if(val)
gradRow[id] += val;
}
}
}
}
}