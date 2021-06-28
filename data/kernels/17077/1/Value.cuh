/* Value Class Header File
 *
 * 200516
 * author = @ugurc
 * ugurcan.cakal@gmail.com
 */

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

// CUDA KERNELS
__global__ void fitness_kernel(int* chromosome, int* collision);

#ifndef VALUE_H
#define VALUE_H

#include <string>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Value {
private:
    int row;
    int* h_collision;
    int* h_chromosome;

    int* d_collision;
    int* d_chromosome;

    cudaError_t initIntDevice(int n, int*& d_vec, int*& const h_vec, bool alloc = true);

public:
    int fitness(int* chromosome);
    cudaError_t errorCheckCUDA(bool synchronize = false);
    cudaError_t getDeviceToHostCh(const int n, int*& h_chromosome, int*& const d_chromosome);
    //int fitness(int* chromosome, dim3 gridSize = 0, dim3 blockSize = 0, size_t shared = 0, bool synchronize = true, bool memCopy=true);
    Value(int n = 8);
    ~Value();
    std::string toString();
    void update(int* chromosome);

    float activity(int n, int* chromosome);
    int maxCollision(int n);
    std::string getInfo();
    //CUDA_CALLABLE_MEMBER std::string toString();
};

#endif // VALUE_H