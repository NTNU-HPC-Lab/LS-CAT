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


////////////////////////////////////////////////////////////////////////////////
// Shortcut shared memory atomic addition functions
////////////////////////////////////////////////////////////////////////////////

#define TAG_MASK 0xFFFFFFFFU
__global__ void mergeHistogram256Kernel( uint *d_Histogram, uint *d_PartialHistograms, uint histogramCount )
{
uint sum = 0;

for (uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE)
{
sum += d_PartialHistograms[blockIdx.x + i * HISTOGRAM256_BIN_COUNT];
}

__shared__ uint data[MERGE_THREADBLOCK_SIZE];
data[threadIdx.x] = sum;

for (uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1)
{
__syncthreads();

if (threadIdx.x < stride)
{
data[threadIdx.x] += data[threadIdx.x + stride];
}
}

if (threadIdx.x == 0)
{
d_Histogram[blockIdx.x] = data[0];
}
}