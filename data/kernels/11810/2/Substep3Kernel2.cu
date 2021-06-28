#include "includes.h"
__global__ void Substep3Kernel2 (double *Dens, double *Qplus, double *viscosity_array, double *TAURR, double *TAURP,double *TAUPP, double *DivergenceVelocity, int nrad, int nsec, double *Rmed, int Cooling, double *EnergyNew, double dt, double *EnergyMed, double *SigmaMed, double *CoolingTimeMed, double *EnergyInt, double ADIABATICINDEX, double *QplusMed)
{
int j = threadIdx.x + blockDim.x*blockIdx.x;
int i = threadIdx.y + blockDim.y*blockIdx.y;
double den, num;

if (i==0 && j<nsec){
/* We calculate the heating source term Qplus for i=0 */

if (viscosity_array[nrad-1] != 0.0) {
/* power-law extrapolation */
Qplus[i*nsec + j] = Qplus[(i+1)*nsec + j]*exp(log(Qplus[(i+1)*nsec + j]/Qplus[(i+2)*nsec + j]) * \
log(Rmed[i]/Rmed[i+1]) / log(Rmed[i+1]/Rmed[i+2]));
}
else
Qplus[i*nsec + j] = 0.0;
}

/* Now we can update energy with source terms from i=0 */
if (i<nrad && j<nsec){
if (!Cooling){
num = dt*Qplus[i*nsec + j] + EnergyInt[i*nsec + j];
den = 1.0+(ADIABATICINDEX-1.0)*dt*DivergenceVelocity[i*nsec + j];
EnergyNew[i*nsec + j] = num/den;
}
else{
num = EnergyMed[i]*dt*Dens[i*nsec + j]/SigmaMed[i] + CoolingTimeMed[i]*EnergyInt[i*nsec + j] + \
dt*CoolingTimeMed[i]*(Qplus[i*nsec + j]-QplusMed[i]*Dens[i*nsec + j]/SigmaMed[i]);

den = dt + CoolingTimeMed[i] + (ADIABATICINDEX-1.0)*dt*CoolingTimeMed[i]*DivergenceVelocity[i*nsec + j];
EnergyNew[i*nsec + j] = num/den;
}
}
}