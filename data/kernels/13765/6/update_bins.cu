#include "includes.h"
__global__ void update_bins(unsigned int* bins, int* in_binID, int binNumber, const int size){
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x;
int nt = blockDim.x * blockDim.y;

__shared__ unsigned int temp[1024];
temp[tid] = 0;
__syncthreads();

for(int x=tid; x<size; x+=nt){
if(in_binID[x] == i){;
temp[tid]++;
}
if(in_binID[x] > i){
break;
}
}
__syncthreads();

if(tid == 0){
for(int x = 0; x<binNumber;x++){
bins[i] += temp[x];
}
}


}