#include "includes.h"
/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and
* proprietary rights in and to this software and related documentation.
* Any use, reproduction, disclosure, or distribution of this software
* and related documentation without an express license agreement from
* NVIDIA Corporation is strictly prohibited.
*
* Please refer to the applicable NVIDIA end user license agreement (EULA)
* associated with this source code for terms and conditions that govern
* your use of this NVIDIA software.
*
*/



#define SIZE    (100*1024*1024)



__global__ void histo_kernel( unsigned char *buffer, long size, unsigned int *histo ) {

// clear out the accumulation buffer called temp
// since we are launched with 256 threads, it is easy
// to clear that memory with one write per thread
__shared__  unsigned int temp[256];
temp[threadIdx.x] = 0;
__syncthreads();

// calculate the starting index and the offset to the next
// block that each thread will be processing
int i = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while (i < size) {
atomicAdd( &temp[buffer[i]], 1 );
i += stride;
}
// sync the data from the above writes to shared memory
// then add the shared memory values to the values from
// the other thread blocks using global memory
// atomic adds
// same as before, since we have 256 threads, updating the
// global histogram is just one write per thread!
__syncthreads();
atomicAdd( &(histo[threadIdx.x]), temp[threadIdx.x] );
}