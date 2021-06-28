#include "includes.h"
__global__ void CircumPlanetaryMassKernel (double *Dens, double *Surf, double *CellAbscissa, double *CellOrdinate, double xpl, double ypl, int nrad, int nsec, double HillRadius, double *mdcp0) /* LISTA */
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

double dist;

if (i<nrad && j<nsec){
dist = sqrt((CellAbscissa[i*nsec + j]-xpl)*(CellAbscissa[i*nsec + j]-xpl) + (CellOrdinate[i*nsec + j]-ypl) * \
(CellOrdinate[i*nsec + j]-ypl));
if (dist < HillRadius) mdcp0[i*nsec + j] = Surf[i]* Dens[i*nsec + j];
else mdcp0[i*nsec + j] = 0.0;
}
}