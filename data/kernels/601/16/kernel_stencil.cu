#include "includes.h"
__global__ void kernel_stencil(float *new_data, float *data, float *param_a, float *param_b, float *param_c, float *param_wrk, float *param_bnd) {

int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

if (_tid_ >= 129 * 65 * 65) return;

int idx_0 =_tid_ / 65 / 65;
int idx_1 = (_tid_ / 65) % 65;
int idx_2 = (_tid_ / 1) % 65;

if (idx_0 - 1 < 0 || idx_0 + 1 >= 129) { new_data[_tid_] = 0.0; return; }
if (idx_1 - 1 < 0 || idx_2 + 1 >= 65) { new_data[_tid_] = 0.0; return; }
if (idx_1 - 1 < 0 || idx_2 + 1 >= 65) { new_data[_tid_] = 0.0; return; }

float v000 = data[(idx_0) * 65 * 65 + (idx_1) * 65 + (idx_2)];
float v100 = data[(idx_0 + 1) * 65 * 65 + (idx_1) * 65 + (idx_2)];
float v010 = data[(idx_0) * 65 * 65 + (idx_1 + 1) * 65 + (idx_2)];
float v001 = data[(idx_0) * 65 * 65 + (idx_1) * 65 + (idx_2 + 1)];
float v110 = data[(idx_0 + 1) * 65 * 65 + (idx_1 + 1) * 65 + (idx_2)];
float v120 = data[(idx_0 + 1) * 65 * 65 + (idx_1 - 1) * 65 + (idx_2)];
float v210 = data[(idx_0 - 1) * 65 * 65 + (idx_1 + 1) * 65 + (idx_2)];
float v220 = data[(idx_0 - 1) * 65 * 65 + (idx_1 - 1) * 65 + (idx_2)];
float v011 = data[(idx_0) * 65 * 65 + (idx_1 + 1) * 65 + (idx_2 + 1)];
float v021 = data[(idx_0) * 65 * 65 + (idx_1 - 1) * 65 + (idx_2 + 1)];
float v012 = data[(idx_0) * 65 * 65 + (idx_1 + 1) * 65 + (idx_2 - 1)];
float v022 = data[(idx_0) * 65 * 65 + (idx_1 - 1) * 65 + (idx_2 - 1)];
float v101 = data[(idx_0 + 1) * 65 * 65 + (idx_1) * 65 + (idx_2 + 1)];
float v201 = data[(idx_0 - 1) * 65 * 65 + (idx_1) * 65 + (idx_2 + 1)];
float v102 = data[(idx_0 + 1) * 65 * 65 + (idx_1) * 65 + (idx_2 - 1)];
float v202 = data[(idx_0 - 1) * 65 * 65 + (idx_1) * 65 + (idx_2 - 1)];
float v200 = data[(idx_0 - 1) * 65 * 65 + (idx_1) * 65 + (idx_2)];
float v020 = data[(idx_0) * 65 * 65 + (idx_1 - 1) * 65 + (idx_2)];
float v002 = data[(idx_0) * 65 * 65 + (idx_1) * 65 + (idx_2 - 1)];

new_data[_tid_] =
v000 + 0.8 * (((
param_a[65 * 65 * 4 * idx_0 + 65 * 4 * idx_1 + 4 * idx_2 + 0] * v100 +
param_a[65 * 65 * 4 * idx_0 + 65 * 4 * idx_1 + 4 * idx_2 + 1] * v010 +
param_a[65 * 65 * 4 * idx_0 + 65 * 4 * idx_1 + 4 * idx_2 + 2] * v001 +
param_b[65 * 65 * 3 * idx_0 + 65 * 3 * idx_1 + 3 * idx_2 + 0] *
(v110 - v120 - v210 + v220) +
param_b[65 * 65 * 3 * idx_0 + 65 * 3 * idx_1 + 3 * idx_2 + 1] *
(v011 - v021 - v012 + v022) +
param_b[65 * 65 * 3 * idx_0 + 65 * 3 * idx_1 + 3 * idx_2 + 2] *
(v101 - v201 - v102 + v202) +
param_c[65 * 65 * 3 * idx_0 + 65 * 3 * idx_1 + 3 * idx_2 + 0] * v200 +
param_c[65 * 65 * 3 * idx_0 + 65 * 3 * idx_1 + 3 * idx_2 + 1] * v020 +
param_c[65 * 65 * 3 * idx_0 + 65 * 3 * idx_1 + 3 * idx_2 + 2] * v002 +
param_wrk[65 * 65 * idx_0 + 65 * idx_1 + idx_2]) *
param_a[65 * 65 * 4 * idx_0 + 65 * 4 * idx_1 + 4 * idx_2 + 3] -
v000) * param_bnd[65 * 65 * idx_0 + 65 * idx_1 + idx_2]);
}