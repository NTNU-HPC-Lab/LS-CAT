#include "includes.h"
__device__ float machine_eps_flt() {
typedef union {
int i32;
float f32;
} flt_32;

flt_32 s;

s.f32 = 1.;
s.i32++;
return (s.f32 - 1.);
}
__device__ double machine_eps_dbl() {
typedef union {
long long i64;
double d64;
} dbl_64;

dbl_64 s;

s.d64 = 1.;
s.i64++;
return (s.d64 - 1.);
}
__global__ void calc_consts(float *fvals, double *dvals) {

int i = threadIdx.x + blockIdx.x*blockDim.x;
if (i==0) {
fvals[EPS] = machine_eps_flt();
dvals[EPS]= machine_eps_dbl();

float xf, oldxf;
double xd, oldxd;

xf = 2.; oldxf = 1.;
xd = 2.; oldxd = 1.;

/* double until overflow */
/* Note that real fmax is somewhere between xf and oldxf */
while (!isinf(xf))  {
oldxf *= 2.;
xf *= 2.;
}

while (!isinf(xd))  {
oldxd *= 2.;
xd *= 2.;
}

dvals[MAX] = oldxd;
fvals[MAX] = oldxf;

/* half until overflow */
/* Note that real fmin is somewhere between xf and oldxf */
xf = 1.; oldxf = 2.;
xd = 1.; oldxd = 2.;

while (xf != 0.)  {
oldxf /= 2.;
xf /= 2.;
}

while (xd != 0.)  {
oldxd /= 2.;
xd /= 2.;
}

dvals[MIN] = oldxd;
fvals[MIN] = oldxf;

}
return;
}