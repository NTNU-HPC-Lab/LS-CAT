#ifndef _MY_CUDA_CUH_
#define _MY_CUDA_CUH_

// CUDA headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include<curand.h>

// thrust headers
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/device_ptr.h>
#include<thrust/tabulate.h>
#include<thrust/sequence.h>
#include<thrust/sort.h>

// C headers
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// C++ headers
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept>

using namespace std;

namespace myCuda {
  void gpuInfo();
  
  namespace print {
    __global__ void print_float(float* x, int leng);
    __global__ void print_int(int* x, int leng);
    __global__ void print_double(double* x, int leng);
    __global__ void print_long(long* x, int leng);
    __global__ void print_char(char* x, int leng);
    __global__ void print_cstr(char** x, int leng);
  }

  namespace math {
    __device__ float logit1(const float x);
    __global__ void logit(float* y, const float* x, int leng);
  }

  namespace ran {
    namespace int_hash_fn {
      __host__ __device__ unsigned int twong7(unsigned int);
    }
    namespace gen {
      struct runif_gen;
    }
    __host__ __device__  void runif(thrust::device_vector<float>&);
  }
}
#endif