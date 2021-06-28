#include "includes.h"
/*
Hello world of wave propagation in CUDA. FDTD acoustic wave propagation in homogeneous medium. Second order accurate in time and eigth in space.

Oleg Ovcharenko
Vladimir Kazei, 2019

oleg.ovcharenko@kaust.edu.sa
vladimir.kazei@kaust.edu.sa
*/

/*
Add this to c_cpp_properties.json if linting isn't working for CUDA libraries
"includePath": [
"/usr/local/cuda-10.0/targets/x86_64-linux/include",
"${workspaceFolder}/**"
],
*/

// Check error codes for CUDA functions
__global__ void kernel_add_wavelet(float *d_u, float *d_wavelet, int it)
{
/*
d_u             :pointer to an array on device where to add source term
d_wavelet       :pointer to an array on device with source signature
it              :time step id
*/
unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
unsigned int idx = gy * c_nx + gx;

if ((gx == c_isrc) && (gy == c_jsrc))
{
d_u[idx] += d_wavelet[it];
}
}