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

__global__ void normalize(double *g_idata, double *g_odata, unsigned int n, int maxIndx)
{

double max = g_idata[maxIndx];
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i < n)
{
g_odata[i] = exp(g_idata[i] - max);
}

}