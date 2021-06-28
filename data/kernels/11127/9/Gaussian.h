#pragma once
#include "GaussianSingle.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#define DEBUG

// Threads per block
const int threads1D = 256;
const int threads2D = 16;

// Guard the __syncthreads() command as it is not recognized:
#ifdef __INTELLISENSE__
#define syncthreads()
#else
#define syncthreads() __syncthreads()
#endif

inline int div_ceil(int numerator, int denominator)
{
	std::div_t res = std::div(numerator, denominator);
	return res.quot + (res.rem != 0);
};

Vector gaussSolveCudaBlock(Matrix& mat, Vector& v);


/* Swap row k with row i, using a specific number of blocks.
*/
__global__ void swapRow(float* mat, float* b, int cols, int num_block, int k);
/* Swap row k with row i using multiple blocks. Outputs the k:th column as a separate vector
*/
__global__ void swapRow(float* mat, float* b, float* column_k, int rows, int cols, int k);
__global__ void greatestRowK(float* mat, int rows, int cols, int k);