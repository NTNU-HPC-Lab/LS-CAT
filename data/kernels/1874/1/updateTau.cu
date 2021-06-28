#include "includes.h"
__device__ void updateCMax(const int nbrOfGrids, const double *d_u1, const double *d_u2, const double *d_u3, const double *d_gama, double *d_cMax)
{
*d_cMax = 0; int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
double ro, p, u;
__shared__ double c;
for (int i = index; i < nbrOfGrids; i += stride){
if (d_u1[i] == 0)
continue;
ro = d_u1[i];
u = d_u2[i] / ro;
p = (d_u3[i] - ro * u * u / 2) * (*d_gama - 1);
c = sqrt(*d_gama * abs(p) / ro);
if (*d_cMax < c + abs(u))
*d_cMax = c + abs(u);
}
}
__global__ void updateTau(const int nbrOfGrids, const double *d_u1, const double *d_u2, const double *d_u3, const double *d_gama, double *d_cMax, const double *d_h, const double *d_cfl, double *d_tau) {
updateCMax(nbrOfGrids, d_u1, d_u2, d_u3, d_gama, d_cMax);
*d_tau = *d_cfl * *d_h / *d_cMax;
}