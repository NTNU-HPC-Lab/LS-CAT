#include "includes.h"
__global__ void HydroUpdatePrim_CUDA3_kernel(float *Rho, float *Vx, float *Vy, float *Vz, float *Etot, float *dUD, float *dUS1, float *dUS2, float *dUS3, float *dUTau, float dt, int size)
{
// get thread and block index
const long tx = threadIdx.x;
const long bx = blockIdx.x;
const long by = blockIdx.y;

int igrid = tx + bx*CUDA_BLOCK_SIZE + by*CUDA_BLOCK_SIZE*CUDA_GRID_SIZE;

if (igrid < 2 || igrid > size - 3)
return;

float D, S1, S2, S3, Tau;
D   = Rho[igrid];
S1  = D*Vx[igrid];
S2  = D*Vy[igrid];
S3  = D*Vz[igrid];
Tau = D*Etot[igrid];

D   += dUD[igrid];
S1  += dUS1[igrid];
S2  += dUS2[igrid];
S3  += dUS3[igrid];
Tau += dUTau[igrid];

Rho[igrid] = D;
Vx[igrid] = S1/D;
Vy[igrid] = S2/D;
Vz[igrid] = S3/D;
Etot[igrid] = Tau/D;

}