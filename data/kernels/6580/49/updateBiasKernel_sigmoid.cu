#include "includes.h"
__device__ float sigmoid(float x) {
return 1.0f / (1 + __expf(-x));
}
__global__ void updateBiasKernel_sigmoid(float* dZ, float* b, int cols, int row, float learning_rate){
int bid = blockIdx.x;
extern __shared__ float _share[];
//float * _max = _share;
float * _sum = _share;
float* sp = dZ + cols * bid;
_sum[threadIdx.x] = 0.0;

for(int id = threadIdx.x ; id < cols; id += blockDim.x){
//	int id = tid + threadIdx.x;
//if(id < cols){
_sum[threadIdx.x] += sp[id];
//}
}
__syncthreads();
int len = blockDim.x;
while(len != 1)
{
__syncthreads();
int skip = (len + 1) >> 1;
if(threadIdx.x < (len >> 1))
{
_sum[threadIdx.x] += _sum[threadIdx.x + skip];
}
len = (len + 1) >> 1;
}
__syncthreads();
b[bid] -= learning_rate * (_sum[0]/cols);
}