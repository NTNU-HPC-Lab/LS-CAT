#include "includes.h"
/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/



//All three kernels run 512 threads per workgroup
//Must be a power of two
#define THREADBLOCK_SIZE 1024

////////////////////////////////////////////////////////////////////////////////
// Basic scan codelets
////////////////////////////////////////////////////////////////////////////////
//Naive inclusive scan: O(N * log2(N)) operations
//Allocate 2 * 'size' local memory, initialize the first half
//with 'size' zeros avoiding if(pos >= offset) condition evaluation
//and saving instructions
__global__ void uniformUpdate( uint4 *d_Data, uint *d_Buffer )
{
__shared__ uint buf;
uint pos = blockIdx.x * blockDim.x + threadIdx.x;

if (threadIdx.x == 0)
{
buf = d_Buffer[blockIdx.x];
}

__syncthreads();

uint4 data4 = d_Data[pos];
data4.x += buf;
data4.y += buf;
data4.z += buf;
data4.w += buf;
d_Data[pos] = data4;
}