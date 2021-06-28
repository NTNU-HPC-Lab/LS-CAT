#include "includes.h"
__global__ void CalcMass(double *Mass_d, double *GlobalMass_d, double *Rho_d, double  A, double *Altitudeh_d, double *lonlat_d, double *areasT, int     num, bool    DeepModel) {

int id  = blockIdx.x * blockDim.x + threadIdx.x;
int nv  = gridDim.y;
int lev = blockIdx.y;

if (id < num) {
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

//mass in control volume = density*volume
Mass_d[id * nv + lev] = Rho_d[id * nv + lev] * Vol;
}
}