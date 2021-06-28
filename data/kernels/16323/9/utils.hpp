#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <cuda_runtime.h>

//Simply holds a cuda check function. Can wrap cuda calls with CHECK(...) and will
//output any errors from the call.

#define CUDACHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	cudaError_t prevCode = cudaGetLastError();

	if (prevCode != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(prevCode), file, line);
        if (abort) exit(code);
    }


    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#endif