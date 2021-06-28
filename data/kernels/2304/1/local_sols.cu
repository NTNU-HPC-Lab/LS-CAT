#include "includes.h"


//////////// Calculates weighting for assembling single element solution ///////////
// One weight is evaluated for each node
// Added back to global memory
__device__ void jacobi_iter( float *ue, float *up_glob, int *cells, float *temp1, int idx, int idy)
{
float ue_new;
int v;
int offset = 15*threadIdx.x;

/*
Le_shrd = &temp1[offset];
be_shrd = &temp1[offset + 9];
u_old  = &temp1[offset + 12];
*/

v = cells[(idx*3) + idy];

ue_new = temp1[(offset + 9) + idy];
temp1[(offset + 12) + idy] = up_glob[v];

__syncthreads();

ue_new -= temp1[offset + (idy*3) + ((idy+1)%3) ] * temp1[(offset + 12) + (idy+1) % 3];
ue_new -= temp1[offset + (idy*3) + ((idy+2)%3) ] * temp1[(offset + 12) +  (idy+2) % 3];

ue_new /= temp1[offset + (idy*3) + idy];

ue[(idx*3) + idy] = ue_new;
}
__device__ void elems_shared_cpy(float *Le, float *be, float *temp1, int idx, int idy){
int offset = 15*threadIdx.x;

// Le_shrd = &temp1[offset];
// be_shrd = &temp1[offset + 9];

temp1[(offset + 9) + idy] = be[(idx*3) + idy];
for(int i=0; i<3; i++){
temp1[offset + (idy*3) + i] = Le[(idx*9) + (idy*3) + i];
}
}
__global__ void local_sols( float *Le, float *be, float *ue, float *up_glob, int *cells, int num_cells)
{
int idx = blockIdx.x*blockDim.x + threadIdx.x;
int idy = blockIdx.y*blockDim.y + threadIdx.y;
extern __shared__ float temp1[];

if(idx < num_cells && idy < blockDim.y){
elems_shared_cpy(Le, be, temp1, idx, idy);
__syncthreads();
jacobi_iter(ue, up_glob, cells, temp1, idx, idy);
}
}