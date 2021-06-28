#include "includes.h"
__device__ void swap(int &a, int &b){
int t = a;
a = b;
b = t;
}
__global__ void littleBinoticSort(int* arr,int num, int numMax){
unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

if(tid >= num) arr[tid] = INT_MAX;

__syncthreads();

for(unsigned int i=2; i<=numMax; i<<=1){
for(unsigned int j=i>>1; j>0; j>>=1){
unsigned int swapIdx = tid ^ j;

if(swapIdx > tid){
if((tid & i)==0){
if(arr[tid] > arr[swapIdx]){
swap(arr[tid], arr[swapIdx]);
}
}
else{
if(arr[tid] < arr[swapIdx]){
swap(arr[tid], arr[swapIdx]);
}
}
}

__syncthreads();
}
}
}