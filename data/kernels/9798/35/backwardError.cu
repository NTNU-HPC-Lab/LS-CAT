#include "includes.h"
/*
* JCuda - Java bindings for NVIDIA CUDA driver and runtime API
* http://www.jcuda.org
*
*
* This code is based on the NVIDIA 'reduction' CUDA sample,
* Copyright 1993-2010 NVIDIA Corporation.
*/


extern "C"

extern "C"


extern "C"


extern "C"

extern "C"

extern "C"

extern "C"

__global__ void backwardError(int n, double *actual, double *target, double* out)
{

int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i<n)
{
out[i] += (actual[i] - target[i]);
}
}