#include "includes.h"
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
__global__ void kernel_evaluatenu(int Nd, double qsum, double *q, double deltanu,double nulow) {
unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
if (tid<Nd) {
double thisnu=(nulow+((double)tid)*deltanu);
double dgm=digamma(thisnu*0.5+0.5);
q[tid]=dgm-log((thisnu+1.0)*0.5); /* psi((nu+1)/2)-log((nu+1)/2) */
dgm=digamma(thisnu*0.5);
q[tid]+=-dgm+log((thisnu)*0.5); /* -psi((nu)/2)+log((nu)/2) */
q[tid]+=-qsum+1.0; /* -(-sum(ln(w_i))/N+sum(w_i)/N)+1 */
}
}