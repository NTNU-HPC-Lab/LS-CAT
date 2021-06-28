#include "includes.h"
__device__ __forceinline__ size_t gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int d)
{
return (NX*(NY*(d-1)+y)+x);
}
__device__ __forceinline__ size_t gpu_field0_index(unsigned int x, unsigned int y)
{
return NX*y+x;
}
__global__ void gpu_bc_charge(double *h0, double *h1, double *h2)
{
unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
unsigned int y = blockIdx.y;

perturb = 0;

if (y == 0) {
double multi0c = 2.0*charge0*w0;
double multisc = 2.0*charge0*ws;
double multidc = 2.0*charge0*wd;
// lower plate for charge density

double ht1 = h2[gpu_fieldn_index(x, 0, 1)];
double ht2 = h2[gpu_fieldn_index(x, 0, 2)];
double ht3 = h2[gpu_fieldn_index(x, 0, 3)];
double ht4 = h2[gpu_fieldn_index(x, 0, 4)];
double ht5 = h2[gpu_fieldn_index(x, 0, 5)];
double ht6 = h2[gpu_fieldn_index(x, 0, 6)];
double ht7 = h2[gpu_fieldn_index(x, 0, 7)];
double ht8 = h2[gpu_fieldn_index(x, 0, 8)];
// lower plate for constant charge density

h0[gpu_field0_index(x, 0)] = -h0[gpu_field0_index(x, 0)] + multi0c;
h1[gpu_fieldn_index(x, 0, 3)] = -ht1 + multisc;
h1[gpu_fieldn_index(x, 0, 4)] = -ht2 + multisc;
h1[gpu_fieldn_index(x, 0, 1)] = -ht3 + multisc;
h1[gpu_fieldn_index(x, 0, 2)] = -ht4 + multisc;
h1[gpu_fieldn_index(x, 0, 7)] = -ht5 + multidc;
h1[gpu_fieldn_index(x, 0, 8)] = -ht6 + multidc;
h1[gpu_fieldn_index(x, 0, 5)] = -ht7 + multidc;
h1[gpu_fieldn_index(x, 0, 6)] = -ht8 + multidc;
}
}