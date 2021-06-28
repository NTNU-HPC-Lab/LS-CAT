#include "includes.h"
__global__ void simple_sinf(float* out, const size_t _data_size, int fnCode, const float _dx, const float _frange_start) {
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < _data_size) {
float x  = _frange_start + i * _dx;
int idx  = 2 * i;
out[idx] = x;

switch (fnCode) {
case 0: out[idx + 1] = sinf(x); break;
case 1: out[idx + 1] = cosf(x); break;
case 2: out[idx + 1] = tanf(x); break;
case 3: out[idx + 1] = log10f(x); break;
}
}
}