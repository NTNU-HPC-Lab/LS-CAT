#include "includes.h"
__global__ void VanLeerRadialKernel (double *Rinf, double *Rsup, double *QRStar, double *DensStar, double *Vrad, double *LostByDisk, int nsec, int nrad, double dt, int OpenInner, double *Qbase, double *invSurf)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

double varq, dtheta;

if (i<nrad && j<nsec){
dtheta = 2.0*PI/(double)nsec;
varq = dt*dtheta*Rinf[i]*QRStar[i*nsec + j]* DensStar[i*nsec + j]*Vrad[i*nsec + j];
varq -= dt*dtheta*Rsup[i]*QRStar[(i+1)*nsec + j]* DensStar[(i+1)*nsec + j]*Vrad[(i+1)*nsec + j];
Qbase[i*nsec + j] += varq*invSurf[i];

if (i==0 && OpenInner)
LostByDisk[j] = varq;

}
}