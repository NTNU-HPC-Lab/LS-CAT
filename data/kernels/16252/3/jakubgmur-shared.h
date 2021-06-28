#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

//__FILE__ is a preprocessor macro that expands to full path to the current file.
//__LINE__ is a preprocessor macro that expands to current line number in the source

#define ErrorCheckCUDA(ans) { CheckErrorCUDA((ans), __FILE__, __LINE__); }
inline void CheckErrorCUDA(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, ":::::::::::::CheckErrorCUDA::::::::::::: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

