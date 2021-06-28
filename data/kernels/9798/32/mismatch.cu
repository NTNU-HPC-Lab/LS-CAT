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

__global__ void mismatch(int n, double* actual, double *target, int *mis)
{

mis[0] = 0;

int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i<n)
{
if(target[i] >= 0.5 && actual[i] < 0.5) {mis[0] = 1;}
if(target[i] < 0.5 && actual[i] >= 0.5) {mis[0] = 1;}
}
}