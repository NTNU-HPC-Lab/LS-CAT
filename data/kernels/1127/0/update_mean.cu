#include "includes.h"
__global__ void update_mean(double* pressure_mean_d, double* pressure_d, double* Rho_mean_d, double* Rho_d, double* Mh_mean_d, double* Mh_d, double* Wh_mean_d, double* Wh_d, int     n_since_out, int     num) {

int id  = blockIdx.x * blockDim.x + threadIdx.x;
int nv  = gridDim.y;
int lev = blockIdx.y;

if (id < num) {
pressure_mean_d[id * nv + lev] =
1.0 / n_since_out
* (pressure_mean_d[id * nv + lev] * (n_since_out - 1) + pressure_d[id * nv + lev]);
Rho_mean_d[id * nv + lev] =
1.0 / n_since_out
* (Rho_mean_d[id * nv + lev] * (n_since_out - 1) + Rho_d[id * nv + lev]);
Mh_mean_d[3 * id * nv + 3 * lev + 0] =
1.0 / n_since_out
* (Mh_mean_d[3 * id * nv + 3 * lev + 0] * (n_since_out - 1)
+ Mh_d[3 * id * nv + 3 * lev] + 0);
Mh_mean_d[3 * id * nv + 3 * lev + 1] =
1.0 / n_since_out
* (Mh_mean_d[3 * id * nv + 3 * lev + 1] * (n_since_out - 1)
+ Mh_d[3 * id * nv + 3 * lev + 1]);
Mh_mean_d[3 * id * nv + 3 * lev + 2] =
1.0 / n_since_out
* (Mh_mean_d[3 * id * nv + 3 * lev + 2] * (n_since_out - 1)
+ Mh_d[3 * id * nv + 3 * lev + 2]);
Wh_mean_d[id * (nv + 1) + lev] =
1.0 / n_since_out
* (Wh_mean_d[id * (nv + 1) + lev] * (n_since_out - 1) + Wh_d[id * (nv + 1) + lev]);
if (lev == nv - 1) {
Wh_mean_d[id * (nv + 1) + lev + 1] =
1.0 / n_since_out
* (Wh_mean_d[id * (nv + 1) + lev + 1] * (n_since_out - 1)
+ Wh_d[id * (nv + 1) + lev + 1]);
}
}
}