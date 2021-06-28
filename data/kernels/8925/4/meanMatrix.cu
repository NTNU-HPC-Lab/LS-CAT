#include "includes.h"
__global__ void meanMatrix(double *dMatrix, double *dMean, int dSize, int *d_mutex){
__shared__ double cache[threadsPerBlock];
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int cacheIndex = threadIdx.x;
double temp = 0;
while (tid < dSize) {
temp += dMatrix[tid];
tid  += blockDim.x * gridDim.x;
}
// set the cache values
cache[cacheIndex] = temp;
// synchronize threads in this block
__syncthreads();

int i = blockDim.x/2;
while (i != 0) {
if (cacheIndex < i)
cache[cacheIndex] += cache[cacheIndex + i];
__syncthreads();
i /= 2;
}

if(cacheIndex == 0){
while(atomicCAS(d_mutex,0,1) != 0);  //lock
*dMean += cache[0];
atomicExch(d_mutex, 0);  //unlock

*dMean = dMean[0]/dSize;
}
}