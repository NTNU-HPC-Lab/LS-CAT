#include <omp.h>
#include <stdio.h>  // stdio functions are used since C++ streams aren't necessarily thread safe
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace testCuda {
	int testInCuda();
}

namespace CudaMatrixCal {

	template <typename matrixAT>
	int testCCCC(Matrix *A) {
		return 1;
	};

	//Matrix *matrixMulByCuda(Matrix *matrixA, Matrix *matrixB);
	
}