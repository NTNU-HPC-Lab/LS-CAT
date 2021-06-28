#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace ConstVolKernel {

	// Multiply tridiagonal matrix P by the matrix U
	// P is a square tridiagonal matrix of dimension dim x dim
	// U is a matrix of dimension dim x size, and is represented as U[i][j] = u[i + j * dim]
	void tridiag_x_matrix_GPU(int dim, int size, float* p_d, float* p_m, float* p_u, float* u);
	void tridiag_x_matrix_GPU(int dim, int size, float p_d, float p_m, float p_u, float* u);

	// Kernel for the function above. (P is dim x dim and U is dim x size)
	// blockSize = k * dim ---> (each block multiplies P by k column vectors of U)
	// nbBlocks = size / k ---> k must divide size
	__global__ void tridiag_x_matrix_k(float p_d, float p_m, float p_u, float* u, int n);
	__global__ void tridiag_x_matrix_k(float* p_d, float* p_m, float* p_u, float* u, int n);

}

