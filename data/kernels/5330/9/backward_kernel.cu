#include "includes.h"
__global__ void backward_kernel(const float *dz, const float *z, const float *var, const float *weight, const float *bias, const float *edz, const float *eydz, float *dx, float *dweight, float *dbias, float eps, int N, int C, int S) {
int plane = blockIdx.x;
float _edz = edz[plane];
float _eydz = eydz[plane];

float gamma = weight != 0 ? abs(weight[plane]) + eps : 1.f;
float beta = bias != 0 ? bias[plane] : 0.f;

if (dx != 0) {
float _var = var[plane];
float invStd = 0;
if (_var != 0.f || eps != 0.f) {
invStd = 1 / sqrt(_var + eps);
}

float mul = gamma * invStd;

for (int batch = 0; batch < N; ++batch) {
for (int n = threadIdx.x; n < S; n += blockDim.x) {
float _dz = dz[(batch * C + plane) * S + n];
float _y = (z[(batch * C + plane) * S + n] - beta) / gamma;
dx[(batch * C + plane) * S + n] = (_dz - _edz - _y * _eydz) * mul;
}
}
}

if (dweight != 0 || dbias != 0) {
float norm = N * S;

if (dweight != 0) {
if (threadIdx.x == 0) {
if (weight[plane] > 0)
dweight[plane] += _eydz * norm;
else if (weight[plane] < 0)
dweight[plane] -= _eydz * norm;
}
}

if (dbias != 0) {
if (threadIdx.x == 0) {
dbias[plane] += _edz * norm;
}
}
}
}