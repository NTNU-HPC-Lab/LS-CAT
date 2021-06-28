#include "includes.h"
__global__ void softmax_trivial(float* softmaxP, float* b, int rows, int cols){
int tid = threadIdx.x;
int bid = blockIdx.x;

float _max = -100000000.0;
float sum = 0.0;

if(tid * cols + bid < rows * cols){
for(int i = 0 ; i < rows ; i++)	_max = max(_max, b[i * cols + bid]);
for(int i = 0 ; i < rows ; i++)	softmaxP[i * cols + bid] = (b[i * cols + bid] - _max);
for(int i = 0 ; i < rows ; i++)	softmaxP[i * cols + bid] = __expf(softmaxP[i * cols + bid]);
for(int i = 0 ; i < rows ; i++)	sum += softmaxP[i * cols + bid];
for(int i = 0 ; i < rows ; i++)	softmaxP[i * cols + bid] /= sum;
}
}