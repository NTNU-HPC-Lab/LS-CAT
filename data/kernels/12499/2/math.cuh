#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "Defines.hpp"


__device__ float fInvSqrt_D(const float& in);
__device__ void doParticle(p_type* pos, p_type* vel, p_type* acc, p_type* mass, int numParticles, int pIndex2, int index2, int thisIndex, float tstep);
__global__ void beginFrame(p_type* pos, p_type* vel, p_type* acc, p_type* mass, int numParticles, int numBlocks);
__global__ void ARR_ADD(p_type* getter, const p_type *giver, int N);
__global__ void ARR_ADDC(float* result, float* in1, float* in2, int N);
__global__ void ARR_SET(p_type* getter, const p_type value, int N);
__host__ void doFrame(p_type* d_pos, p_type* d_vel, p_type* d_acc, p_type* d_mass, int numParticles, int numBlocks, int numBlocks2);



