#include "includes.h"
__global__ void rayleighHS(double *Mh_d, double *pressure_d, double *Rho_d, double *Altitude_d, double  surf_drag, double  bl_sigma, double  Gravit, double  time_step, int     num) {

int id  = blockIdx.x * blockDim.x + threadIdx.x;
int nv  = gridDim.y;
int lev = blockIdx.y;

if (id < num) {
double sigma;
double sigmab = bl_sigma;
double kf     = surf_drag;
double kv_hs;
double ps, pre;
double psm1;

//      Calculates surface pressure
psm1 = pressure_d[id * nv + 1]
- Rho_d[id * nv + 0] * Gravit * (-Altitude_d[0] - Altitude_d[1]);
ps = 0.5 * (pressure_d[id * nv + 0] + psm1);

pre   = pressure_d[id * nv + lev];
sigma = (pre / ps);

//      Momentum dissipation constant.
kv_hs = kf * max(0.0, (sigma - sigmab) / (1.0 - sigmab));

//      Update momenta
for (int k = 0; k < 3; k++)
Mh_d[id * 3 * nv + lev * 3 + k] =
Mh_d[id * 3 * nv + lev * 3 + k] / (1.0 + kv_hs * time_step);

// Wh_d[id * (nv + 1) + lev + k] = Wh_d[id * (nv + 1) + lev + k] / (1.0 + kv_hs * time_step);
}
}