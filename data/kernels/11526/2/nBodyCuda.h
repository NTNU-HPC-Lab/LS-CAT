#ifndef NBODYCUDA_H
#define NBODYCUDA_H

#include <unistd.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include "vector_types.h"
#include <cuda.h>
#include <stdio.h>
#include <device_functions.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
// probably some unecessary includes here

const int MAX_GLOB_MEM = 2095251456; 	// total global memory
const int MAX_SHARED_MEM = 49152; 		// shared memory per block
const int MAX_GRID_SIZE = 65535;		// Max 1Dimensional grid size

inline 
cudaError_t checkCuda(cudaError_t result){
	if (result != cudaSuccess){
		fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
		assert(result==cudaSuccess);
	}
	return result;
}
#endif
