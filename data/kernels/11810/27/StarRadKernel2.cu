#include "includes.h"
__global__ void StarRadKernel2 (double *Qbase2, double *Vrad, double *QStar, double dt, int nrad, int nsec, double *invdiffRmed, double *Rmed, double *dq)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

if (i<nrad && j<nsec){
if (Vrad[i*nsec + j] > 0.0 && i > 0)
QStar[i*nsec + j] = Qbase2[(i-1)*nsec + j] + (Rmed[i]-Rmed[i-1]-Vrad[i*nsec + j]*dt)*0.5*dq[i-1+j*nrad];
else
QStar[i*nsec + j] = Qbase2[i*nsec + j]-(Rmed[i+1]-Rmed[i]+Vrad[i*nsec + j]*dt)*0.5*dq[i+j*nrad];

}

if (i == 0 && j<nsec)
QStar[j] = QStar[j+nsec*nrad] = 0.0;
}