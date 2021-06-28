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

__global__ void getTargetIndex(int n, int *index, double *w)
{

int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i<n)
{
if(w[i] == 1.0) {index[0] = i;}
}
}