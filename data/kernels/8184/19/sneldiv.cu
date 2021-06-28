#include "includes.h"
__global__ void sneldiv(unsigned short *inA, float *inB, int   *sub, int Nprj, int snno)
{
int idz = threadIdx.x + blockDim.x*blockIdx.x;
if (blockIdx.y<Nprj && idz<snno) {
// inB > only active bins of the subset
// inA > all sinogram bins
float a = (float)inA[snno*sub[blockIdx.y] + idz];
a /= inB[snno*blockIdx.y + idz];//sub[blockIdx.y]
inB[snno*blockIdx.y + idz] = a; //sub[blockIdx.y]
}
}