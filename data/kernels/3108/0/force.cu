#include "includes.h"

//Device Code....

__global__ void force(double *deviceq,double *devicex,double *devicey,double *devicez,double *deviceFx,double *deviceFy,double *deviceFz,double *deviceU,int N)
{
double foxij,foyij,fozij,xij,yij,zij,rij;
int i = blockDim.x * blockIdx.x + threadIdx.x;
int j;
if(i < N-1)
{       //Anurag Dogra
for(j=i;j<N;j++)
{
if(i!=j)
{
xij = devicex[i] - devicex[j];
yij = devicey[i] - devicey[j];
zij = devicez[i] - devicez[j];

//Distance calculation
rij = sqrt((xij*xij)+(yij*yij)+(zij*zij));

foxij = foxij + ((deviceq[i]*deviceq[j]*xij)/(rij*rij*rij));
foyij = foyij + ((deviceq[i]*deviceq[j]*yij)/(rij*rij*rij));
fozij = fozij + ((deviceq[i]*deviceq[j]*zij)/(rij*rij*rij));

deviceFx[i] = deviceFx[i] + foxij;
deviceFy[i] = deviceFy[i] + foyij;
deviceFz[i] = deviceFz[i] + fozij;
deviceU[i] = deviceU[i] + 2*(deviceq[j]/rij);

}
}
}

}