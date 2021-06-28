#include "includes.h"
__global__ void Substep1Kernel (double *Pressure, double *Dens, double *VradInt, double *invdiffRmed, double *Potential, double *Rinf, double *invRinf, double *Vrad, double *VthetaInt, double *Vtheta, double *Rmed, double dt, int nrad, int nsec, double OmegaFrame, int ZMPlus, double IMPOSEDDISKDRIFT, double SIGMASLOPE)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;
double gradp, gradphi, vradint, vradint2, supp_torque, dxtheta, invdxtheta;
double vt2;

// i=1->nrad , j=0->nsec
if (i > 0 && i<nrad && j<nsec){
gradp = (Pressure[i*nsec + j] - Pressure[(i-1)*nsec + j])*2.0/(Dens[i*nsec + j] + Dens[(i-1)*nsec + j])*invdiffRmed[i];
gradphi = (Potential[i*nsec + j] - Potential[(i-1)*nsec + j])*invdiffRmed[i];
vt2 = Vtheta[i*nsec + j] + Vtheta[i*nsec + (j+1)%nsec] + Vtheta[(i-1)*nsec + j] + Vtheta[(i-1)*nsec + (j+1)%nsec];
vt2 = vt2/4.0  +OmegaFrame*Rinf[i];
vt2 = vt2*vt2;

vradint = -gradp - gradphi;
vradint2 = vradint + vt2*invRinf[i];
VradInt[i*nsec + j] = Vrad[i*nsec+j] + dt*vradint2;


}

// i=0->nrad ,   j=0->nsec
if (i<nrad && j<nsec){

supp_torque = IMPOSEDDISKDRIFT*0.5*pow(Rmed[i], -2.5+SIGMASLOPE);
dxtheta = 2.0*PI/(double)nsec*Rmed[i];
invdxtheta = 1.0/dxtheta;

gradp = (Pressure[i*nsec + j] - Pressure[i*nsec + ((j-1)+nsec)%nsec])*2.0/(Dens[i*nsec +j] +Dens[i*nsec + ((j-1)+nsec)%nsec]) \
*invdxtheta;

//if (ZMPlus) gradp *= 1; //gradp *= SG_aniso_coeff;  Definir mas adelante SG_aniso_coeff

gradphi = (Potential[i*nsec+ j] - Potential[i*nsec + ((j-1)+nsec)%nsec])*invdxtheta;
VthetaInt[i*nsec + j] =  Vtheta[i*nsec+j] - dt*(gradp+gradphi);
VthetaInt[i*nsec + j] += dt*supp_torque;
}
}