#include "includes.h"
__device__ double min2(double a, double b)
{
if (b < a) return b;
return a;
}
__device__ double max2(double a, double b)
{
if (b > a) return b;
return a;
}
__global__ void ConditionCFLKernel2D1 (double *Rsup, double *Rinf, double *Rmed, int nsec, int nrad, double *Vresidual, double *Vtheta, double *Vmoy, int FastTransport, double *SoundSpeed, double *Vrad, double *DT2D)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

double dxrad, dxtheta, invdt1, invdt2, invdt3, invdt4, dvr, dvt, dt;

if (i > 0 && i<nrad && j<nsec){
dxrad = Rsup[i]-Rinf[i];
dxtheta = Rmed[i]*2.0*PI/(double)nsec;
if (FastTransport) Vresidual[i*nsec + j] = Vtheta[i*nsec + j]-Vmoy[i]; /* Fargo algorithm */
else Vresidual[i*nsec + j] = Vtheta[i*nsec + j];                       /* Standard algorithm */
//Vresidual[i*nsec + nsec] = Vresidual[i*nsec];
invdt1 = SoundSpeed[i*nsec + j]/(min2(dxrad,dxtheta));
invdt2 = fabs(Vrad[i*nsec + j])/dxrad;
invdt3 = fabs(Vresidual[i*nsec + j])/dxtheta;
dvr = Vrad[(i+1)*nsec + j]-Vrad[i*nsec + j];
dvt = Vtheta[i*nsec + (j+1)%nsec]-Vtheta[i*nsec + j];
if (dvr >= 0.0) dvr = 1e-10;
else dvr = -dvr;
if (dvt >= 0.0) dvt = 1e-10;
else dvt = -dvt;
invdt4 = max2(dvr/dxrad, dvt/dxtheta);
invdt4*= 4.0*CVNR*CVNR;
dt = CFLSECURITY/sqrt(invdt1*invdt1+invdt2*invdt2+invdt3*invdt3+invdt4*invdt4);
DT2D[i*nsec + j] = dt; // array nrad*nsec size dt
}
}