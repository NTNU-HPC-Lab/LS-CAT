#include "includes.h"
__global__ static void update_inverse_cuda (float *Ainv, float *u, int N, int rowstride, int k)
{
__shared__ float A_k[NMAX], u_shared[NMAX], Ainv_u[NMAX], Ainv_shared[NMAX];
A_k[threadIdx.x] = Ainv[k*rowstride+threadIdx.x];
u_shared[threadIdx.x] = u[threadIdx.x];

// First, compute k'th element of Ainv_u
Ainv_u[threadIdx.x] = u_shared[threadIdx.x] * A_k[threadIdx.x];
__syncthreads();
for (int n=N>>1; n>0; n = n>>1) {
float a;
if (threadIdx.x < n)
a = Ainv_u[2*threadIdx.x] + Ainv_u[2*threadIdx.x+1];
__syncthreads();
Ainv_u[threadIdx.x] = a;
__syncthreads();
}
float prefact = -1.0f/(1.0f + Ainv_u[0]);

for (int row=0; row<N; row++) {
Ainv_shared[threadIdx.x] = Ainv[row*rowstride+threadIdx.x];
__syncthreads();
Ainv_u[threadIdx.x] = u_shared[threadIdx.x] * Ainv_shared[threadIdx.x];
for (int n=N>>1; n>0; n = n>>1) {
float a;
if (threadIdx.x < n)
a = Ainv_u[2*threadIdx.x] + Ainv_u[2*threadIdx.x+1];
__syncthreads();
Ainv_u[threadIdx.x] = a;
__syncthreads();
}
__syncthreads();
// Now Ainv_u[0] has the row'th element of Ainv_u.
Ainv[row*rowstride + threadIdx.x] =
Ainv_shared[threadIdx.x] + prefact*Ainv_u[0]*A_k[threadIdx.x];
}

}