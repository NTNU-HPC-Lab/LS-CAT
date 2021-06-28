#pragma once

#include "driver_types.h"		// cudaStream_t -> for create stream

struct cpu_gpu_mem {

	void* gpu_p;			// gpu pointer
	void* cpu_p;			// cpu pointer
	void* cpu_p_tiny;
	void* gpu_p_tiny;
	int numberCount;		
	int numberCountTiny;
	cudaStream_t stream;	// created stream
};