#include "includes.h"
__global__ void g_getSoftMaxP(float* softMaxP, float* b, int cols, int row){
int bid = blockIdx.x;
extern __shared__ float _share[];
float * _max = _share;
float * _sum = _share + blockDim.x;
float* sp = softMaxP + bid;
_sum[threadIdx.x] = 0.0;
_max[threadIdx.x] = -100000000.0;
for(int tid = threadIdx.x * cols + blockIdx.x; tid < row * cols; tid += cols){
//int id = tid + threadIdx.x;
//if(id < cols){
sp[tid] += b[tid];
_max[threadIdx.x] = max(_max[threadIdx.x], sp[tid]);
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
if(_max[threadIdx.x] < _max[threadIdx.x + skip])
{
_max[threadIdx.x] = _max[threadIdx.x + skip];
}
}
len = (len + 1) >> 1;
}
__syncthreads();
for(int tid = threadIdx.x * cols + blockIdx.x; tid < row * cols; tid += cols){
//	int id = tid + threadIdx.x;
//if(id < cols){
sp[tid] -= _max[0];
sp[tid] = __expf(sp[tid]);
_sum[threadIdx.x] += sp[tid];
//}
}
__syncthreads();
len = blockDim.x;
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
for(int tid = threadIdx.x * cols + blockIdx.x; tid < row * cols; tid += cols){
//int id = tid + threadIdx.x;
//if(id < cols){
sp[tid] /= _sum[0];
//}
}
}