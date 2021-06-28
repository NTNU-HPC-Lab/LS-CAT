#include "includes.h"
/*
* JCuda - Java bindings for NVIDIA CUDA driver and runtime API
* http://www.jcuda.org
*
* Copyright 2010 Marco Hutter - http://www.jcuda.org
*/


/**
* Kernels for the JCudaDriverTextureTest class. These
* kernels will read data via the texture references at
* the given positions, and store the value that is
* read into the given output memory.
*/

texture<float,  1, cudaReadModeElementType> texture_float_1D;
texture<float,  2, cudaReadModeElementType> texture_float_2D;
texture<float,  3, cudaReadModeElementType> texture_float_3D;

texture<float4, 1, cudaReadModeElementType> texture_float4_1D;
texture<float4, 2, cudaReadModeElementType> texture_float4_2D;
texture<float4, 3, cudaReadModeElementType> texture_float4_3D;

extern "C"


extern "C"


extern "C"


extern "C"


extern "C"


extern "C"






__global__ void test_float4_1D(float4 *output, float posX)
{
float4 result = tex1D(texture_float4_1D, posX);
output[0] = result;
}