#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <functional>
#include <math.h>
#include <time.h>
#include <random>
#include <assert.h>
#include <device_launch_parameters.h>

typedef unsigned char byte;

static void HandleError(cudaError_t err,
                        const char *file,
                        int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#define CUDA_CALL(err) (HandleError(err, __FILE__, __LINE__))

inline size_t get_number_of_parts(size_t whole, size_t divider)
{
    return ((whole + divider - 1) / divider);
}

#define CUDA_TIMED_BLOCK_START(fn_name)      \
    const char *___tmdFnName = fn_name;      \
    cudaEvent_t startEvent, stopEvent;       \
    float elapsedTime;                       \
    CUDA_CALL(cudaEventCreate(&startEvent)); \
    CUDA_CALL(cudaEventCreate(&stopEvent));  \
    CUDA_CALL(cudaEventRecord(startEvent, 0));

#define CUDA_TIMED_BLOCK_END                                              \
    CUDA_CALL(cudaEventRecord(stopEvent, 0));                             \
    CUDA_CALL(cudaEventSynchronize(stopEvent));                           \
    CUDA_CALL(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent)); \
    printf("%s took: %f ms\n", ___tmdFnName, elapsedTime);                \
    CUDA_CALL(cudaEventDestroy(startEvent));                              \
    CUDA_CALL(cudaEventDestroy(stopEvent));

template <typename T>
bool all_not_eq(const std::vector<T> &data, const T &cmp)
{
    for (size_t i = 0; i < data.size(); i++)
    {
        if (data[i] == cmp)
            return false;
    }
    return true;
}

template <typename T>
void safe_cuda_free(T *ptr)
{
    if (ptr != nullptr)
    {
        cudaFree(ptr);
    }
}
