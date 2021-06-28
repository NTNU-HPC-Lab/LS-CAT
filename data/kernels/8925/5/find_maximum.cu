#include "includes.h"
__global__ void find_maximum(double *array, double *max, int dSize, int *d_mutex){
int index = threadIdx.x + blockIdx.x*blockDim.x;
int stride = gridDim.x*blockDim.x;
int offset = 0;

__shared__ double cache[threadsPerBlock];

double temp = -999999999.0;
while(index + offset < dSize){
temp = fmaxf(temp, array[index + offset]);
offset += stride;
}

cache[threadIdx.x] = temp;

__syncthreads();


// reduction
unsigned int i = blockDim.x/2;
while(i != 0){
if(threadIdx.x < i){
cache[threadIdx.x] = fmax(cache[threadIdx.x], cache[threadIdx.x + i]);
}

__syncthreads();
i /= 2;
}

if(threadIdx.x == 0){
while(atomicCAS(d_mutex,0,1) != 0);  //lock
*max = fmax(*max, cache[0]);
atomicExch(d_mutex, 0);  //unlock
}
}