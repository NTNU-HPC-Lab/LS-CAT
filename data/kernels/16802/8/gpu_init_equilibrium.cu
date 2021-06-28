#include "includes.h"
__device__ __forceinline__ size_t gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int z, unsigned int d)
{
return (NX*(NY*(NZ*(d-1)+z)+y)+x);
}
__device__ __forceinline__ size_t gpu_scalar_index(unsigned int x, unsigned int y, unsigned int z)
{
return NX*(NY*z + y)+x;
}
__device__ __forceinline__ size_t gpu_field0_index(unsigned int x, unsigned int y, unsigned int z)
{
return NX*(NY*z + y)+x;
}
__global__ void gpu_init_equilibrium(double *f0, double *f1, double *h0, double *h1, double *temp0, double *temp1, double *r, double *c, double *u, double *v, double *w, double *ex, double *ey, double *ez, double*temp)
{
unsigned int y = blockIdx.y;
unsigned int z = blockIdx.z;
unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;

double rho    = r[gpu_scalar_index(x,y,z)];
double ux     = u[gpu_scalar_index(x,y,z)];
double uy     = v[gpu_scalar_index(x,y,z)];
double uz     = w[gpu_scalar_index(x,y,z)];
double charge = c[gpu_scalar_index(x,y,z)];
double Ex     = ex[gpu_scalar_index(x,y,z)];
double Ey     = ey[gpu_scalar_index(x,y,z)];
double Ez     = ez[gpu_scalar_index(x,y,z)];
double Temp   = temp[gpu_scalar_index(x,y,z)];

// load equilibrium
// feq_i  = w_i rho [1 + 3(ci . u) + (9/2) (ci . u)^2 - (3/2) (u.u)]
// feq_i  = w_i rho [1 - 3/2 (u.u) + (ci . 3u) + (1/2) (ci . 3u)^2]
// feq_i  = w_i rho [1 - 3/2 (u.u) + (ci . 3u){ 1 + (1/2) (ci . 3u) }]

// temporary variables
double w0r = w0*rho;
double wsr = ws*rho;
double war = wa*rho;
double wdr = wd*rho;

double w0c = w0*charge;
double wsc = ws*charge;
double wac = wa*charge;
double wdc = wd*charge;

double w0t = w0*Temp;
double wst = ws*Temp;
double wat = wa*Temp;
double wdt = wd*Temp;

double omusq   = 1.0 - 0.5*(ux*ux+uy*uy+uz*uz)/cs_square;
double omusq_c = 1.0 - 0.5*((ux + K*Ex)*(ux + K*Ex) + (uy + K*Ey)*(uy + K*Ey) + (uz + K*Ez)*(uz + K*Ez)) / cs_square;

double tux   = ux / cs_square / CFL;
double tuy   = uy / cs_square / CFL;
double tuz   = uz / cs_square / CFL;
double tux_c = (ux + K*Ex) / cs_square / CFL;
double tuy_c = (uy + K*Ey) / cs_square / CFL;
double tuz_c = (uz + K*Ez) / cs_square / CFL;

// zero weight
f0[gpu_field0_index(x,y,z)]      = w0r*(omusq);
h0[gpu_field0_index(x,y,z)]      = w0c*(omusq_c);
temp0[gpu_field0_index(x, y, z)] = w0t*(omusq);

// adjacent weight
// flow
double cidot3u = tux;
f1[gpu_fieldn_index(x,y,z,1)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
cidot3u = -tux;
f1[gpu_fieldn_index(x,y,z,2)]  = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuy;
f1[gpu_fieldn_index(x,y,z,3)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
cidot3u = -tuy;
f1[gpu_fieldn_index(x,y,z,4)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
cidot3u = tuz;
f1[gpu_fieldn_index(x,y,z,5)] = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = -tuz;
f1[gpu_fieldn_index(x,y,z,6)] = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));

// charge
cidot3u = tux_c;
h1[gpu_fieldn_index(x,y,z,1)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = -tux_c;
h1[gpu_fieldn_index(x,y,z,2)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuy_c;
h1[gpu_fieldn_index(x,y,z,3)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = -tuy_c;
h1[gpu_fieldn_index(x,y,z,4)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuz_c;
h1[gpu_fieldn_index(x,y,z,5)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = -tuz_c;
h1[gpu_fieldn_index(x,y,z,6)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));

// temperature
cidot3u = tux;
temp1[gpu_fieldn_index(x, y, z, 1)] = wst*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = -tux;
temp1[gpu_fieldn_index(x, y, z, 2)] = wst*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuy;
temp1[gpu_fieldn_index(x, y, z, 3)] = wst*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = -tuy;
temp1[gpu_fieldn_index(x, y, z, 4)] = wst*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuz;
temp1[gpu_fieldn_index(x, y, z, 5)] = wst*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = -tuz;
temp1[gpu_fieldn_index(x, y, z, 6)] = wst*(omusq + cidot3u*(1.0 + 0.5*cidot3u));

// diagonal weight
// flow
cidot3u = tux+tuy;
f1[gpu_fieldn_index(x,y,z,7)]  = war*(omusq + cidot3u*(1.0+0.5*cidot3u));
cidot3u = -tuy-tux;
f1[gpu_fieldn_index(x,y,z,8)]  = war*(omusq + cidot3u*(1.0+0.5*cidot3u));
cidot3u = tux+tuz;
f1[gpu_fieldn_index(x,y,z,9)]  = war*(omusq + cidot3u*(1.0+0.5*cidot3u));
cidot3u = -tux-tuz;
f1[gpu_fieldn_index(x,y,z,10)] = war*(omusq + cidot3u*(1.0+0.5*cidot3u));
cidot3u = tuz + tuy;
f1[gpu_fieldn_index(x,y,z,11)] = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = -tuy - tuz;
f1[gpu_fieldn_index(x,y,z,12)] = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tux - tuy;
f1[gpu_fieldn_index(x,y,z,13)] = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuy - tux;
f1[gpu_fieldn_index(x,y,z,14)] = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tux - tuz;
f1[gpu_fieldn_index(x,y,z,15)] = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuz - tux;
f1[gpu_fieldn_index(x,y,z,16)] = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuy - tuz;
f1[gpu_fieldn_index(x,y,z,17)] = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuz - tuy;
f1[gpu_fieldn_index(x,y,z,18)] = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));

// charge
cidot3u = tux_c + tuy_c;
h1[gpu_fieldn_index(x, y, z, 7)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = -tuy_c - tux_c;
h1[gpu_fieldn_index(x, y, z, 8)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tux_c + tuz_c;
h1[gpu_fieldn_index(x, y, z, 9)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = -tux_c - tuz_c;
h1[gpu_fieldn_index(x, y, z, 10)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuy_c + tuz_c;
h1[gpu_fieldn_index(x, y, z, 11)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = -tuy_c - tuz_c;
h1[gpu_fieldn_index(x, y, z, 12)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tux_c - tuy_c;
h1[gpu_fieldn_index(x, y, z, 13)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuy_c - tux_c;
h1[gpu_fieldn_index(x, y, z, 14)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tux_c - tuz_c;
h1[gpu_fieldn_index(x, y, z, 15)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuz_c - tux_c;
h1[gpu_fieldn_index(x, y, z, 16)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuy_c - tuz_c;
h1[gpu_fieldn_index(x, y, z, 17)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuz_c - tuy_c;
h1[gpu_fieldn_index(x, y, z, 18)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));

// temperature
cidot3u = tux + tuy;
temp1[gpu_fieldn_index(x, y, z, 7)] = wat*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = -tuy - tux;
temp1[gpu_fieldn_index(x, y, z, 8)] = wat*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tux + tuz;
temp1[gpu_fieldn_index(x, y, z, 9)] = wat*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = -tux - tuz;
temp1[gpu_fieldn_index(x, y, z, 10)] = wat*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuy + tuz;
temp1[gpu_fieldn_index(x, y, z, 11)] = wat*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = -tuy - tuz;
temp1[gpu_fieldn_index(x, y, z, 12)] = wat*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tux - tuy;
temp1[gpu_fieldn_index(x, y, z, 13)] = wat*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuy - tux;
temp1[gpu_fieldn_index(x, y, z, 14)] = wat*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tux - tuz;
temp1[gpu_fieldn_index(x, y, z, 15)] = wat*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuz - tux;
temp1[gpu_fieldn_index(x, y, z, 16)] = wat*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuy - tuz;
temp1[gpu_fieldn_index(x, y, z, 17)] = wat*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuz - tuy;
temp1[gpu_fieldn_index(x, y, z, 18)] = wat*(omusq + cidot3u*(1.0 + 0.5*cidot3u));


// 3d diagonal
//flow
cidot3u = tux + tuy + tuz;
f1[gpu_fieldn_index(x, y, z, 19)] = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = -tuy - tux - tuz;
f1[gpu_fieldn_index(x, y, z, 20)] = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tux + tuy - tuz;
f1[gpu_fieldn_index(x, y, z, 21)] = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuz - tux - tuy;
f1[gpu_fieldn_index(x, y, z, 22)] = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tux + tuz - tuy;
f1[gpu_fieldn_index(x, y, z, 23)] = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuy - tux - tuz;
f1[gpu_fieldn_index(x, y, z, 24)] = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuy + tuz - tux;
f1[gpu_fieldn_index(x, y, z, 25)] = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tux - tuy - tuz;
f1[gpu_fieldn_index(x, y, z, 26)] = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));

//charge
cidot3u = tux_c + tuy_c + tuz_c;
h1[gpu_fieldn_index(x, y, z, 19)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = -tux_c -tuy_c - tuz_c;
h1[gpu_fieldn_index(x, y, z, 20)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tux_c + tuy_c - tuz_c;
h1[gpu_fieldn_index(x, y, z, 21)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuz_c - tux_c - tuy_c;
h1[gpu_fieldn_index(x, y, z, 22)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tux_c + tuz_c - tuy_c;
h1[gpu_fieldn_index(x, y, z, 23)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuy_c - tux_c - tuz_c;
h1[gpu_fieldn_index(x, y, z, 24)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuy_c + tuz_c - tux_c;
h1[gpu_fieldn_index(x, y, z, 25)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tux_c - tuy_c - tuz_c;
h1[gpu_fieldn_index(x, y, z, 26)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));

//temperature
cidot3u = tux + tuy + tuz;
temp1[gpu_fieldn_index(x, y, z, 19)] = wdt*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = -tux - tuy - tuz;
temp1[gpu_fieldn_index(x, y, z, 20)] = wdt*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tux + tuy - tuz;
temp1[gpu_fieldn_index(x, y, z, 21)] = wdt*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuz - tux - tuy;
temp1[gpu_fieldn_index(x, y, z, 22)] = wdt*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tux + tuz - tuy;
temp1[gpu_fieldn_index(x, y, z, 23)] = wdt*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuy - tux - tuz;
temp1[gpu_fieldn_index(x, y, z, 24)] = wdt*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tuy + tuz - tux;
temp1[gpu_fieldn_index(x, y, z, 25)] = wdt*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
cidot3u = tux - tuy - tuz;
temp1[gpu_fieldn_index(x, y, z, 26)] = wdt*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
}