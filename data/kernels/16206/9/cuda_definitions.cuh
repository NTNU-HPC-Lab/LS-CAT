#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// accuracy
#define EPS (0.0000001f)

// global memory
#define M (4 * (1 << 20))
union device_global_memory {
	int int_t[M];
	float float_t[M];
};
__device__ device_global_memory gl;
