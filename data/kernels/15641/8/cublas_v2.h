#ifndef _CUDA_CUBLAS_ERROR_H_
#define _CUDA_CUBLAS_ERROR_H_

#include <stdio.h>
#include <cublas_v2.h>
// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
// Handy Macros //
////////////////////////////////////////////////////////
#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      cudaDeviceReset();                                      \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)
////////////////////////////////////////////////////////
#define checkCublasErrors(call)                                 \
  do {                                                          \
    cublasStatus_t err = call;                                  \
    if (err != CUBLAS_STATUS_SUCCESS) {                         \
      printf("CUBLAS error at %s %d: %s\n", __FILE__, __LINE__, \
             _cudaGetErrorEnum(err));                           \
      cudaDeviceReset();                                        \
      exit(EXIT_FAILURE);                                       \
    }                                                           \
  } while (0)
////////////////////////////////////////////////////////
#endif /* _CUDA_CUBLAS_ERROR_H_ */