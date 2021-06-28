#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void dumpMatrix(size_t size1, size_t size2, float* matrix_d, char* path);
namespace BulletOptionKernel {

	void boundaryConditionGPU(size_t spotGridSize, size_t stateSize, float* payoff, float strike);
	__global__ void boundaryCondition_k(float * payoff, size_t spotSize, float strike);

	void initPayoffGPU(size_t spotSize, size_t stateSize, float* payoff, float dx, float Smin, float strike, size_t P1, size_t P2);
	__global__ void initPayoff_k(float* payoff, float dx, float Smin, float strike, size_t P1, size_t P2);

	void interStepGPU(size_t spotSize, size_t stateSize, float* payoff, size_t scheduleCounter, float dx, float Smin, size_t P1, size_t P2, float barrier);
	__global__ void interStep_k(float* payoff, size_t scheduleCounter, float dx, float Smin, size_t P1, size_t P2, float barrier);

}

