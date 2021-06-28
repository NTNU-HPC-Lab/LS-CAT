#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef BASIC_H
#define BASIC_H

__global__ void add(int*, int*, int*);
__global__ void sub(int*, int*, int*);
__global__ void mul(int*, int*, int*);
__global__ void div(float*, float*, float*);

#endif
