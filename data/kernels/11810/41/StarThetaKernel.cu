#include "includes.h"
__global__ void StarThetaKernel (double *Qbase, double *Rmed, int nrad, int nsec, double *dq, double dt)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

double dxtheta, invdxtheta, dqp, dqm;
if (i<nrad && j<nsec){
if (i<nrad){
dxtheta = 2.0*PI/(double)nsec*Rmed[i];
invdxtheta = 1.0/dxtheta;
}
dqm = (Qbase[i*nsec + j] - Qbase[i*nsec + ((j-1)+nsec)%nsec]);
dqp = (Qbase[i*nsec + (j+1)%nsec] - Qbase[i*nsec + j]);

if (dqp * dqm > 0.0)
dq[i*nsec + j] = dqp*dqm/(dqp+dqm)*invdxtheta;
else
dq[i*nsec + j] = 0.0;
}
}