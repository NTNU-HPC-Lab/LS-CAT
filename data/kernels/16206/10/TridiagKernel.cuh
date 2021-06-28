#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_definitions.cuh"

namespace TridiagSolverImpl {

	// Callers
	void thomasGPU(size_t size, size_t dim, float a, float b, float c, float* d_y);
	void thomasGPU(size_t size, size_t dim, float* d_a, float* d_b, float* d_c, float* d_y);
	void pcr(size_t size, size_t dim, float* d_a, float* d_b, float* d_c, float* d_y);
	void pcr(size_t size, size_t dim, float a, float b, float c, float* d_y);

	// Kernels
	__global__ void thom_k(float a, float b, float c, float* y, int n);
	__global__ void thom_k(float* a, float* b, float* c, float* y, int n);
	__global__ void pcr_k(float* a, float* b, float* c, float* y, int n);
	__global__ void pcr_k(float a, float b, float c, float* y, int n);

}