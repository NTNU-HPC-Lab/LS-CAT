#pragma once
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#include <driver_types.h>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <cstdlib>

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		getchar();
		exit(EXIT_FAILURE);
	}
}
