#include "includes.h"
////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

//
// This sample illustrates the usage of CUDA events for both GPU timing and
// overlapping CPU and GPU execution.  Events are inserted into a stream
// of CUDA calls.  Since CUDA stream calls are asynchronous, the CPU can
// perform computations while GPU is executing (including DMA memcopies
// between the host and device).  CPU can query CUDA events to determine
// whether GPU has completed tasks.
//

// includes, system

// includes CUDA Runtime

// includes, project


__global__ void increment_kernel(int *g_data, int inc_value)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
g_data[idx] = g_data[idx] + inc_value;
}