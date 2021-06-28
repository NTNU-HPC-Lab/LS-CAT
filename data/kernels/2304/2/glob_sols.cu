#include "includes.h"


//////////// Calculates weighting for assembling single element solution ///////////
// One weight is evaluated for each node
// Added back to global memory
__global__ void glob_sols( float *Le, float *w, float *u_glob, float *ue, int *cells, int num_cells)
{
int idx = blockIdx.x*blockDim.x + threadIdx.x;
int idy = blockIdx.y*blockDim.y + threadIdx.y;
int v;
float Lii, weight;

if(idx < num_cells && idy < blockDim.y){
v = cells[(idx*3) + idy];               // getting global vertex number
Lii = Le[(idx*9) + (idy*3) + idy];

weight = Lii/w[v];

atomicAdd(&u_glob[v], weight * ue[(idx*3) + idy]);
}
}