#include "includes.h"
__global__ void UpdateVelocitiesKernel (double *VthetaInt, double *VradInt, double *invRmed, double *Rmed, double *Rsup, double *Rinf, double *invdiffRmed, double *invdiffRsup, double *Dens, double *invRinf, double *TAURR, double *TAURP, double *TAUPP, double DeltaT, int nrad, int nsec)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;

double dphi, invdphi;

/* Now we can update velocities
with the viscous source term
of Navier-Stokes equation */

/* vtheta first */
if (i > 0 && i<nrad-1 && j<nsec){
dphi = 2.0*M_PI/(double)nsec;
invdphi = 1.0/dphi;

VthetaInt[i*nsec +j] += DeltaT*invRmed[i]*((Rsup[i]*TAURP[(i+1)*nsec+ j] - Rinf[i]*TAURP[i*nsec +j])*invdiffRsup[i] + \
(TAUPP[i*nsec +j] - TAUPP[i*nsec + ((j-1)+nsec)%nsec])*invdphi + 0.5*(TAURP[i*nsec + j] + TAURP[(i+1)*nsec +j]))/ \
(0.5*(Dens[i*nsec +j]+Dens[i*nsec + ((j-1)+nsec)%nsec]));
}

/* now vrad */
if (i > 0 && i<nrad && j<nsec){
dphi = 2.0*M_PI/(double)nsec;
invdphi = 1.0/dphi;

VradInt[i*nsec +j] += DeltaT*invRinf[i]*((Rmed[i]*TAURR[i*nsec +j] - Rmed[i-1]*TAURR[(i-1)*nsec + j])*invdiffRmed[i] + \
(TAURP[i*nsec + (j+1)%nsec] - TAURP[i*nsec + j])*invdphi - 0.5*(TAUPP[i*nsec +j] + TAUPP[(i-1)*nsec + j]))/ \
(0.5*(Dens[i*nsec +j] + Dens[(i-1)*nsec + j]));
}
}