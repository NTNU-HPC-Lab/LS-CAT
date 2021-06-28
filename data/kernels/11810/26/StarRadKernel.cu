#include "includes.h"
__global__ void StarRadKernel (double *Qbase2, double *Vrad, double *QStar, double dt, int nrad, int nsec, double *invdiffRmed, double *Rmed, double *dq)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

double dqm, dqp;

if (i<nrad && j<nsec){
if ((i == 0 || i == nrad-1)) dq[i + j*nrad] = 0.0;
else {
dqm = (Qbase2[i*nsec + j] - Qbase2[(i-1)*nsec + j])*invdiffRmed[i];
dqp = (Qbase2[(i+1)*nsec + j] - Qbase2[i*nsec + j])*invdiffRmed[i+1];

if (dqp * dqm > 0.0)
dq[i+j*nrad] = 2.0*dqp*dqm/(dqp+dqm);
else
dq[i+j*nrad] = 0.0;
}
}
}