#include "includes.h"
__global__ void gpu_transpo_kernel_naive(u_char *Source, u_char *Resultat, unsigned width, unsigned height){
int j = blockIdx.x*blockDim.x + threadIdx.x;
int i = blockIdx.y*blockDim.y + threadIdx.y;

if ((i<0)||(i>=height)||(j<0)||(j>=width)) {}
else {
Resultat[j*height + i]  = Source[i*width + j];
}
}