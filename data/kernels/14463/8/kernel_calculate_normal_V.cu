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

__global__ void kernel_calculate_normal_V(float3* vbo_ptr, float* d_ptr)
{
int i = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;

if(i < d_L && j < d_L)
{
int ik = (i + 1) % d_L;
int jk = (d_L - 1 + j) % d_L;

// x
float x2 =  d_scaling * fabsf( d_ptr[i*d_L +j] );

// a
float a1 =  d_dx;
float a2 =  d_scaling * fabsf( d_ptr[ik*d_L +j] ) - x2;

// b
float b2 =  d_scaling * fabsf( d_ptr[i*d_L +jk] ) - x2;
float b3 = -d_dx;

// n = a x b
float3 n;
n.x =  a2*b3;
n.y = -a1*b3;
n.z =  a1*b2;

// normalize
float norm = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);

vbo_ptr[d_L*i + j].x = n.x/norm;
vbo_ptr[d_L*i + j].y = n.y/norm;
vbo_ptr[d_L*i + j].z = n.z/norm;
}
}