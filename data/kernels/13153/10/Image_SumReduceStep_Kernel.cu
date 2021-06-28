#include "includes.h"
__global__ void  Image_SumReduceStep_Kernel( int* devBufIn, int* devBufOut, int  lastBlockSize)
{
// ONLY USE THIS FUNCTION WITH BLOCK SIZE = (256,1,1);
// NOTE: This method was originally written to use exactly the amt
//       of shared memory available for each block, but I believe
//       I later was told that cmd args use shared mem, which would
//       result in this method spilling over.  Need to check on that.
__shared__ char sharedMem[4096];
int* shmBuf1 = (int*)sharedMem;
int* shmBuf2 = (int*)&sharedMem[512];

int globalIdx = 512 * blockIdx.x + threadIdx.x;
int localIdx  = threadIdx.x;

shmBuf1[localIdx]     = 0;
shmBuf1[localIdx+256] = 0;
shmBuf2[localIdx]     = 0;
shmBuf2[localIdx+256] = 0;

if(blockIdx.x == gridDim.x-1)
{
if(localIdx+256 >= lastBlockSize) devBufIn[globalIdx+256] = 0;
if(localIdx     >= lastBlockSize) devBufIn[globalIdx]     = 0;
}

// Now we reduce each block of 512 values (256 threads) to a single number

shmBuf1[localIdx] = devBufIn[globalIdx] + devBufIn[globalIdx + 256]; __syncthreads();
if(localIdx < 128) shmBuf2[localIdx] = shmBuf1[localIdx]+shmBuf1[localIdx+128]; __syncthreads();
if(localIdx < 64)  shmBuf1[localIdx] = shmBuf2[localIdx]+shmBuf2[localIdx+64];  __syncthreads();
if(localIdx < 32)  shmBuf2[localIdx] = shmBuf1[localIdx]+shmBuf1[localIdx+32];  __syncthreads();
if(localIdx < 16)  shmBuf1[localIdx] = shmBuf2[localIdx]+shmBuf2[localIdx+16];  __syncthreads();
if(localIdx < 8)   shmBuf2[localIdx] = shmBuf1[localIdx]+shmBuf1[localIdx+8];   __syncthreads();
if(localIdx < 4)   shmBuf1[localIdx] = shmBuf2[localIdx]+shmBuf2[localIdx+4];   __syncthreads();
if(localIdx < 2)   shmBuf2[localIdx] = shmBuf1[localIdx]+shmBuf1[localIdx+2];   __syncthreads();

// 2 -> 1
if(localIdx < 1)
devBufOut[blockIdx.x] = shmBuf2[localIdx] + shmBuf2[localIdx + 1];
__syncthreads();

}