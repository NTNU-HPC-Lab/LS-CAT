#include "includes.h"
__global__ void CalcAngMom(double *AngMomx_d, double *AngMomy_d, double *AngMomz_d, double *GlobalAMx_d, double *GlobalAMy_d, double *GlobalAMz_d, double *Mh_d, double *Rho_d, double  A, double  Omega, double *Altitude_d, double *Altitudeh_d, double *lonlat_d, double *areasT, double *func_r_d, int     num, bool    DeepModel) {

int id  = blockIdx.x * blockDim.x + threadIdx.x;
int nv  = gridDim.y;
int lev = blockIdx.y;

if (id < num) {
double AMx, AMy, AMz;
double rx, ry, rz, r;

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

//radius vector
r  = (A + Altitude_d[lev]);
rx = r * func_r_d[id * 3 + 0];
ry = r * func_r_d[id * 3 + 1];
rz = r * func_r_d[id * 3 + 2];

//angular momentum r x p (total x and y over globe should ~ 0, z ~ const)
AMx = ry * Mh_d[id * 3 * nv + lev * 3 + 2] - rz * Mh_d[id * 3 * nv + lev * 3 + 1]
- Rho_d[id * nv + lev] * Omega * r * rz * cos(lonlat_d[id * 2 + 1])
* cos(lonlat_d[id * 2]);
AMy = -rx * Mh_d[id * 3 * nv + lev * 3 + 2] + rz * Mh_d[id * 3 * nv + lev * 3 + 0]
- Rho_d[id * nv + lev] * Omega * r * rz * cos(lonlat_d[id * 2 + 1])
* sin(lonlat_d[id * 2]);
AMz = rx * Mh_d[id * 3 * nv + lev * 3 + 1] - ry * Mh_d[id * 3 * nv + lev * 3 + 0]
+ Rho_d[id * nv + lev] * Omega * r * r * cos(lonlat_d[id * 2 + 1])
* cos(lonlat_d[id * 2 + 1]);
//AMx, AMy should go to zero when integrated over globe
// (but in practice, are just much smaller than AMz)

//total in control volume
AngMomx_d[id * nv + lev] = AMx * Vol;
AngMomy_d[id * nv + lev] = AMy * Vol;
AngMomz_d[id * nv + lev] = AMz * Vol;
}
}