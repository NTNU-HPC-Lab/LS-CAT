#include "includes.h"
__global__ void InitComputeAccelKernel (double *CellAbscissa, double *CellOrdinate, double *Rmed, int nsec, int nrad)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;


if (i<nrad && j<nsec){
CellAbscissa[i*nsec+j] = Rmed[i] * cos((2.0*PI*(double)j)/(double)nsec);
CellOrdinate[i*nsec+j] = Rmed[i] * sin((2.0*PI*(double)j)/(double)nsec);
}
}