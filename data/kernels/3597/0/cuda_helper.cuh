#pragma once

#include <cstdint>
#include "cuda_runtime.h"
#include "cuda_except.cuh"

void checkCudaStatus(cudaError_t status);

void cudaSetDeviceExcept(uint32_t deviceID);

template<typename T>
void allocateCudaBuffer(T*& ptr, uint32_t size)
{
	// Allocate GPU buffers for three vectors (two input, one output).
	checkCudaStatus(cudaMalloc((void**)&ptr, size * sizeof(T)));
}

template<typename T>
void cudaMemcpyExcept(T* dest, const T* src, uint32_t size, cudaMemcpyKind kind)
{
	// Allocate GPU buffers for three vectors (two input, one output).
	checkCudaStatus(cudaMemcpy(dest, src, size * sizeof(T), kind));
}

template<typename T>
void hostToDeviceMemcpy(T* dest, const T* src, uint32_t size)
{
	// Allocate GPU buffers for three vectors (two input, one output).
	cudaMemcpyExcept(dest, src, size, cudaMemcpyHostToDevice);
}

template<typename T>
void deviceToHostMemcpy(T* dest, const T* src, uint32_t size)
{
	// Allocate GPU buffers for three vectors (two input, one output).
	cudaMemcpyExcept(dest, src, size, cudaMemcpyDeviceToHost);
}


void cudaLastErrorToException();

void cudaWaitForDevice();
