#include "includes.h"
__device__ float digamma_fl(float x) {
float result = 0.0f, xx, xx2, xx4;
for ( ; x < 7.0f; ++x) { /* reduce x till x<7 */
result -= 1.0f/x;
}
x -= 1.0f/2.0f;
xx = 1.0f/x;
xx2 = xx*xx;
xx4 = xx2*xx2;
result += logf(x)+(1.0f/24.0f)*xx2-(7.0f/960.0f)*xx4+(31.0f/8064.0f)*xx4*xx2-(127.0f/30720.0f)*xx4*xx4;
return result;
}
__device__ double digamma(double x) {
double result = 0.0, xx, xx2, xx4;
for ( ; x < 7.0; ++x) { /* reduce x till x<7 */
result -= 1.0/x;
}
x -= 1.0/2.0;
xx = 1.0/x;
xx2 = xx*xx;
xx4 = xx2*xx2;
result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
return result;
}
__global__ void kernel_evaluatenu_fl_eight(int Nd, float qsum, float *q, float deltanu,float nulow, float nu0) {
unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
/* each block calculte  psi((nu+8)/2)-log((nu+8)/2) */
/* actually p=2, so psi((nu+2)/2)-log((nu+2)/2) */
float dgm0;
if (threadIdx.x==0) {
dgm0=digamma_fl(nu0*0.5f+1.0f);
dgm0=dgm0-logf((nu0+2.0f)*0.5f); /* psi((nu0+8)/2)-log((nu0+8)/2) */
}
__syncthreads();
if (tid<Nd) {
float thisnu=(nulow+((float)tid)*deltanu);
q[tid]=dgm0; /* psi((nu0+8)/2)-log((nu0+8)/2) */
float dgm=digamma_fl(thisnu*0.5f);
q[tid]+=-dgm+logf((thisnu)*0.5f); /* -psi((nu)/2)+log((nu)/2) */
q[tid]+=-qsum+1.0f; /* -(-sum(ln(w_i))/N+sum(w_i)/N)+1 */
}
}