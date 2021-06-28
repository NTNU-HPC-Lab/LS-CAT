#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <time.h>

#define __syncthreads()
#ifdef __cplusplus 
extern "C" {//<-- extern ½ÃÀÛ
#endif

	class GPUACC
	{
	public:
		GPUACC(void);
		virtual ~GPUACC(void);
		double MatrixMultiplication(float* M, float* N, float* P, int Width);
	};
	
#ifdef __cplusplus 
}
#endif