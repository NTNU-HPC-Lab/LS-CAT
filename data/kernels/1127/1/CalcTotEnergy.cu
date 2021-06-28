#include "includes.h"
__global__ void CalcTotEnergy(double *Etotal_d, double *GlobalE_d, double *Mh_d, double *W_d, double *Rho_d, double *temperature_d, double  Gravit, double  Cp, double  Rd, double  A, double *Altitude_d, double *Altitudeh_d, double *lonlat_d, double *areasT, double *func_r_d, int     num, bool    DeepModel) {

int id  = blockIdx.x * blockDim.x + threadIdx.x;
int nv  = gridDim.y;
int lev = blockIdx.y;

if (id < num) {
double Ek, Eint, Eg;
double wx, wy, wz;
double Cv = Cp - Rd;

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

//calc cartesian values of vertical wind
wx = W_d[id * nv + lev] * cos(lonlat_d[id * 2 + 1]) * cos(lonlat_d[id * 2]);
wy = W_d[id * nv + lev] * cos(lonlat_d[id * 2 + 1]) * sin(lonlat_d[id * 2]);
wz = W_d[id * nv + lev] * sin(lonlat_d[id * 2 + 1]);

//kinetic energy density 0.5*rho*v^2
Ek = 0.5
* ((Mh_d[id * 3 * nv + lev * 3 + 0] + wx) * (Mh_d[id * 3 * nv + lev * 3 + 0] + wx)
+ (Mh_d[id * 3 * nv + lev * 3 + 1] + wy) * (Mh_d[id * 3 * nv + lev * 3 + 1] + wy)
+ (Mh_d[id * 3 * nv + lev * 3 + 2] + wz) * (Mh_d[id * 3 * nv + lev * 3 + 2] + wz))
/ Rho_d[id * nv + lev];

//internal energy rho*Cv*T
Eint = Cv * temperature_d[id * nv + lev] * Rho_d[id * nv + lev];

//gravitation potential energy rho*g*altitude (assuming g = constant)
Eg = Rho_d[id * nv + lev] * Gravit * Altitude_d[lev];

//total energy in the control volume
Etotal_d[id * nv + lev] = (Ek + Eint + Eg) * Vol;

// printfn("E = %e\n",Etotal_d[id*nv+lev]);
}
}