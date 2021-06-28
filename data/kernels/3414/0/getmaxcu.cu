#include "includes.h"

//function declaration
unsigned int getmax(unsigned int *, unsigned int);
//unsigned int getmaxSeq(unsigned int *, unsigned int);

__global__ void getmaxcu(unsigned int* num, int size, int threadCount)
{
__shared__ int localBiggest[32];
if (threadIdx.x==0) {
for (int i = 0; i < 32; i++) {
localBiggest[i] = 0;
}
}
__syncthreads();

int current =  blockIdx.x *blockDim.x + threadIdx.x;   //get current thread ID
int localBiggestCurrent = (current - blockIdx.x *blockDim.x)/32;   //get currentID's warp number
//if current number is bigger than the biggest number so far in the warp, replace it
if ((num[current] > localBiggest[localBiggestCurrent]) && (current < size)) {
localBiggest[localBiggestCurrent] = num[current];
}
__syncthreads();

//using only one thread, loop through all the biggest numbers in each warp
//and return the biggest number out of them all
if (threadIdx.x==0) {
int biggest = localBiggest[0];
for (int i = 1; i < 32; i++) {
if (biggest < localBiggest[i]) {
biggest = localBiggest[i];
}
}
//once found the biggest number in this block, put back into global array
//num with corresponding block number
num[blockIdx.x] = biggest;
}

}