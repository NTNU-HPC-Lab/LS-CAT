#include "includes.h"






__global__ void scale_down_after_fft(float *d_Ex, float *d_Ey, float *d_Ez, int N_grid, int N_grid_all){
int i = blockIdx.x*blockDim.x + threadIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y;
int k = blockIdx.z*blockDim.z + threadIdx.z;
int index = k*N_grid*N_grid + j*N_grid + i;

if(i<N_grid && j<N_grid && k<N_grid){
d_Ex[index] /= float(N_grid_all);
d_Ey[index] /= float(N_grid_all);
d_Ez[index] /= float(N_grid_all);
}
}