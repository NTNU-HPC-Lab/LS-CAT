/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   warpAcceleration.h
 * Author: ziqi
 *
 * Created on February 16, 2019, 8:31 AM
 */

#ifndef WARPACCELERATION_H
#define WARPACCELERATION_H

#ifndef BDIMX
#define BDIMX 32
#endif

#ifndef SMEMDIM
#define SMEMDIM 32
#endif

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>

__global__ void test_shfl_broadcast(float *d_out, float *d_in, const int srcLane);

__global__ void test_shfl_up(float *d_out, float *d_in, const int delta);

__global__ void test_shfl_down(float *d_out, float *d_in, const int delta);

__global__ void test_shfl_xor(float *d_out, float *d_in, const int mask);

__inline__ __device__ float warpReduce(float mySum);
#endif /* WARPACCELERATION_H */

