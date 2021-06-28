#include "includes.h"
__device__ void updateU(const int nbrOfGrids, double *d_u1, double *d_u2, double *d_u3, const double *d_u1Temp, const double *d_u2Temp, const double *d_u3Temp) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x;
for (int i = index; i < nbrOfGrids; i += stride) {
if ((i > 0) && (i < nbrOfGrids - 1)) {
d_u1[i] = d_u1Temp[i];
d_u2[i] = d_u2Temp[i];
d_u3[i] = d_u3Temp[i];
}
}
}
__device__ void step(const int nbrOfGrids, const double *d_u1, const double *d_u2, const double *d_u3, double *d_u1Temp, double *d_u2Temp, double *d_u3Temp, const double *d_f1, const double *d_f2, const double *d_f3, const double *d_tau, const double *d_h) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x;
for (int i = index; i < nbrOfGrids; i += stride) {
if ((i > 0) && (i < nbrOfGrids - 1)) {
d_u1Temp[i] = d_u1[i] - *d_tau / *d_h * (d_f1[i] - d_f1[i - 1]);
d_u2Temp[i] = d_u2[i] - *d_tau / *d_h * (d_f2[i] - d_f2[i - 1]);
d_u3Temp[i] = d_u3[i] - *d_tau / *d_h * (d_f3[i] - d_f3[i - 1]);
}
}
}
__device__ void halfStep(const int nbrOfGrids, const double *d_u1, const double *d_u2, const double *d_u3, double *d_u1Temp, double *d_u2Temp, double *d_u3Temp, const double *d_f1, const double *d_f2, const double *d_f3, const double *d_tau, const double *d_h) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x;
for (int i = index; i < nbrOfGrids; i += stride) {
if ((i > 0) && (i < nbrOfGrids - 1)) {
d_u1Temp[i] = (d_u1[i + 1] + d_u1[i]) / 2 - *d_tau / 2 / *d_h * (d_f1[i + 1] - d_f1[i]);
d_u2Temp[i] = (d_u2[i + 1] + d_u2[i]) / 2 - *d_tau / 2 / *d_h * (d_f2[i + 1] - d_f2[i]);
d_u3Temp[i] = (d_u3[i + 1] + d_u3[i]) / 2 - *d_tau / 2 / *d_h * (d_f3[i + 1] - d_f3[i]);
}
}
}
__device__ void updateFlux(const int nbrOfGrids, const double *d_u1, const double *d_u2, const double *d_u3, double *d_f1, double *d_f2, double *d_f3, const double *d_gama) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x;
double rho, m, e, p;
for (int i = index; i < nbrOfGrids; i += stride) {
rho = d_u1[i];
m = d_u2[i];
e = d_u3[i];
p = (*d_gama - 1) * (e - m * m / rho / 2);
d_f1[i] = m;
d_f2[i] = m * m / rho + p;
d_f3[i] = m / rho * (e + p);
}
}
__device__ void d_boundaryCondition(const int nbrOfGrids, double *d_u1, double *d_u2, double *d_u3) {
d_u1[0] = d_u1[1];
d_u2[0] = -d_u2[1];
d_u3[0] = d_u3[1];
d_u1[nbrOfGrids - 1] = d_u1[nbrOfGrids - 2];
d_u2[nbrOfGrids - 1] = -d_u2[nbrOfGrids - 2];
d_u3[nbrOfGrids - 1] = d_u3[nbrOfGrids - 2];
}
__global__	void laxWendroffStep(const int nbrOfGrids, double *d_u1, double *d_u2, double *d_u3, double *d_u1Temp, double *d_u2Temp, double *d_u3Temp, double *d_f1, double *d_f2, double *d_f3, const double *d_tau, const double *d_h, const double *d_gama) {
updateFlux(nbrOfGrids, d_u1, d_u2, d_u3, d_f1, d_f2, d_f3, d_gama);
halfStep(nbrOfGrids, d_u1, d_u2, d_u3, d_u1Temp, d_u2Temp, d_u3Temp, d_f1, d_f2, d_f3, d_tau, d_h);
d_boundaryCondition(nbrOfGrids, d_u1Temp, d_u2Temp, d_u3Temp);
updateFlux(nbrOfGrids, d_u1Temp, d_u2Temp, d_u3Temp, d_f1, d_f2, d_f3, d_gama);
step(nbrOfGrids, d_u1, d_u2, d_u3, d_u1Temp, d_u2Temp, d_u3Temp, d_f1, d_f2, d_f3, d_tau, d_h);
updateU(nbrOfGrids, d_u1, d_u2, d_u3, d_u1Temp, d_u2Temp, d_u3Temp);
d_boundaryCondition(nbrOfGrids, d_u1, d_u2, d_u3);
}