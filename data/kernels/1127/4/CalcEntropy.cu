#include "includes.h"
__global__ void CalcEntropy(double *Entropy_d, double *pressure_d, double *temperature_d, double  Cp, double  Rd, double  A, double  P_Ref, double *Altitude_d, double *Altitudeh_d, double *lonlat_d, double *areasT, double *func_r_d, int     num, bool    DeepModel) {

int id  = blockIdx.x * blockDim.x + threadIdx.x;
int nv  = gridDim.y;
int lev = blockIdx.y;

if (id < num) {
double kappa = Rd / Cp;
double potT  = temperature_d[id * nv + lev] * pow(P_Ref / pressure_d[id * nv + lev], kappa);
double Sdens = Cp * log(potT);

//calculate control volume
double zup, zlow, Vol;
zup  = Altitudeh_d[lev + 1] + A;
zlow = Altitudeh_d[lev] + A;
if (DeepModel) {
Vol = areasT[id] / pow(A, 2) * (pow(zup, 3) - pow(zlow, 3)) / 3;
}
else {
Vol = areasT[id] * (zup - zlow);
}

//total energy in the control volume
Entropy_d[id * nv + lev] = Sdens * Vol;
}
}