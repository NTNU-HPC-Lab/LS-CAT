#include "includes.h"
__device__ __forceinline__ void copy_c(float const *in, float *out, int slicesizein, int slicesizeout, int C) {
// *out = *in;
for (size_t c(0); c < C; ++c)
out[c * slicesizeout] = in[c * slicesizein];
}
__device__ __forceinline__ void add_c(float const *in, float *out, int slicesizein, int slicesizeout, int C) {
// *out = *in + *out;
for (size_t c(0); c < C; ++c)
out[c * slicesizeout] += in[c * slicesizein];
}
__device__ __forceinline__ int get_index(int X, int Y, int Z, int C, int x, int y, int z) {
return z * (C * X * Y) + y * X + x;
}
__global__ void shift_kernel(float const *in, float *out, int X, int Y, int C, int dx, int dy, float const beta) {
int x(threadIdx.x + blockDim.x * blockIdx.x);
int y(x / X);
x = x % X;

int x_to(x + dx);
int y_to(y + dy);
// int x_to(x);
// int y_to(y);
// y = 0;
// y_to = 0;


if (x >= X || y >= Y || x_to >= X || y_to >= Y || x_to < 0 || y_to < 0)
return;
if (beta>0)
add_c(in + get_index(X, Y, 1, C, x, y, 0), out + get_index(X, Y, 1, C, x_to, y_to, 0), X * Y, X * Y, C);
else
copy_c(in + get_index(X, Y, 1, C, x, y, 0), out + get_index(X, Y, 1, C, x_to, y_to, 0), X * Y, X * Y, C);

}