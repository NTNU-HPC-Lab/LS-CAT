#include "includes.h"
__global__ void VanLeerThetaKernel (double *Rsup, double *Rinf, double *Surf, double dt, int nrad, int nsec, int UniformTransport, int *NoSplitAdvection, double *QRStar, double *DensStar, double *Vazimutal_d, double *Qbase)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

double dxrad, invsurf, varq;

if (i<nrad && j<nsec){
if ((UniformTransport == NO) || (NoSplitAdvection[i] == NO)){
dxrad = (Rsup[i]-Rinf[i])*dt;
invsurf = 1.0/Surf[i];
varq = dxrad*QRStar[i*nsec + j]*DensStar[i*nsec + j]*Vazimutal_d[i*nsec + j];
varq -= dxrad*QRStar[i*nsec + (j+1)%nsec]*DensStar[i*nsec + (j+1)%nsec]*Vazimutal_d[i*nsec + (j+1)%nsec];
Qbase[i*nsec + j] += varq*invsurf;
}
}
}