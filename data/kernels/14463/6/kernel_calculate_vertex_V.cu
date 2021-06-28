#include "includes.h"
/**
*  Quantum Lattice Boltzmann
*  (c) 2015 Fabian Thüring, ETH Zurich
*
*  This file contains all the CUDA kernels and function that make use of the
*  CUDA runtime API
*/

// Local includes

// ==== CONSTANTS ====

__constant__ unsigned int d_L;
__constant__ float d_dx;
__constant__ float d_dt;
__constant__ float d_mass;
__constant__ float d_g;
__constant__ unsigned int d_t;

__constant__ float d_scaling;
__constant__ int d_current_scene;

// ==== INITIALIZATION ====

__global__ void kernel_calculate_vertex_V(float3* vbo_ptr, float* d_ptr)
{
int i = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;

if(i < d_L && j < d_L)
vbo_ptr[d_L*i + j].y = d_scaling * fabsf( d_ptr[i*d_L +j] ) - 0.005f*d_L;
}