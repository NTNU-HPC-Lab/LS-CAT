#pragma once

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#define MAX_EPSILON_ERROR 5e-3f

////////////////////////////////////////////////////////////////////////////////
// Constants
const float angle = 0.5f;        // angle to rotate image by (in radians)

								 // Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;

////////////////////////////////////////////////////////////////////////////////
//! Transform an image using texture lookups
//! @param outputData  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void transformKernel(float *outputData,
	int width,
	int height,
	float theta);
extern "C"
void runTest(void* texData, int width, int height, int textureDataSize);
extern "C"
int RunKernel();