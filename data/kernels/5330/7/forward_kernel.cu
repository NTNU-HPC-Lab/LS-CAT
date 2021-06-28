#include "includes.h"
__global__ void forward_kernel(const float *x, const float *mean, const float *var, const float *weight, const float *bias, float *y, float *z, float eps, int N, int C, int S) {
int plane = blockIdx.x;

float _mean = mean[plane];
float _var = var[plane];
float invStd = 0;
if (_var != 0.f || eps != 0.f) {
invStd = 1 / sqrt(_var + eps);
}

float gamma = weight != 0 ? abs(weight[plane]) + eps : 1.f;
float beta = bias != 0 ? bias[plane] : 0.f;
for (int batch = 0; batch < N; ++batch) {
for (int n = threadIdx.x; n < S; n += blockDim.x) {
float _x = x[(batch * C + plane) * S + n];
float _y = (_x - _mean) * invStd;
float _z = _y * gamma + beta;

y[(batch * C + plane) * S + n] = _y;
z[(batch * C + plane) * S + n] = _z;
}
}
}