#include "includes.h"
__global__ void fill_bspline_4(const float4 *xyzq, const int ncoord, const float *recip, const int nfftx, const int nffty, const int nfftz, int *gix, int *giy, int *giz, float *charge, float *thetax, float *thetay, float *thetaz, float *dthetax, float *dthetay, float *dthetaz) {

// Position to xyzq and atomgrid
unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;

while (pos < ncoord) {
float4 xyzqi = xyzq[pos];
float x = xyzqi.x;
float y = xyzqi.y;
float z = xyzqi.z;
float q = xyzqi.w;

float w;
// NOTE: I don't think we need the +2.0f here..
w = x*recip[0] + y*recip[1] + z*recip[2] + 2.0f;
float frx = (float)(nfftx*(w - (floorf(w + 0.5f) - 0.5f)));

w = x*recip[3] + y*recip[4] + z*recip[5] + 2.0f;
float fry = (float)(nffty*(w - (floorf(w + 0.5f) - 0.5f)));

w = x*recip[6] + y*recip[7] + z*recip[8] + 2.0f;
float frz = (float)(nfftz*(w - (floorf(w + 0.5f) - 0.5f)));

int frxi = (int)(frx);
int fryi = (int)(fry);
int frzi = (int)(frz);

float wx = frx - (float)frxi;
float wy = fry - (float)fryi;
float wz = frz - (float)frzi;

gix[pos] = frxi;
giy[pos] = fryi;
giz[pos] = frzi;
charge[pos] = q;

float3 theta_tmp[4];
float3 dtheta_tmp[4];

theta_tmp[3].x = 0.0f;
theta_tmp[3].y = 0.0f;
theta_tmp[3].z = 0.0f;
theta_tmp[1].x = wx;
theta_tmp[1].y = wy;
theta_tmp[1].z = wz;
theta_tmp[0].x = 1.0f - wx;
theta_tmp[0].y = 1.0f - wy;
theta_tmp[0].z = 1.0f - wz;

// compute standard b-spline recursion
theta_tmp[2].x = 0.5f*wx*theta_tmp[1].x;
theta_tmp[2].y = 0.5f*wy*theta_tmp[1].y;
theta_tmp[2].z = 0.5f*wz*theta_tmp[1].z;

theta_tmp[1].x = 0.5f*((wx+1.0f)*theta_tmp[0].x + (2.0f-wx)*theta_tmp[1].x);
theta_tmp[1].y = 0.5f*((wy+1.0f)*theta_tmp[0].y + (2.0f-wy)*theta_tmp[1].y);
theta_tmp[1].z = 0.5f*((wz+1.0f)*theta_tmp[0].z + (2.0f-wz)*theta_tmp[1].z);

theta_tmp[0].x = 0.5f*(1.0f-wx)*theta_tmp[0].x;
theta_tmp[0].y = 0.5f*(1.0f-wy)*theta_tmp[0].y;
theta_tmp[0].z = 0.5f*(1.0f-wz)*theta_tmp[0].z;

// perform standard b-spline differentiationa
dtheta_tmp[0].x = -theta_tmp[0].x;
dtheta_tmp[0].y = -theta_tmp[0].y;
dtheta_tmp[0].z = -theta_tmp[0].z;

dtheta_tmp[1].x = theta_tmp[0].x - theta_tmp[1].x;
dtheta_tmp[1].y = theta_tmp[0].y - theta_tmp[1].y;
dtheta_tmp[1].z = theta_tmp[0].z - theta_tmp[1].z;

dtheta_tmp[2].x = theta_tmp[1].x - theta_tmp[2].x;
dtheta_tmp[2].y = theta_tmp[1].y - theta_tmp[2].y;
dtheta_tmp[2].z = theta_tmp[1].z - theta_tmp[2].z;

dtheta_tmp[3].x = theta_tmp[2].x - theta_tmp[3].x;
dtheta_tmp[3].y = theta_tmp[2].y - theta_tmp[3].y;
dtheta_tmp[3].z = theta_tmp[2].z - theta_tmp[3].z;

// one more recursion
theta_tmp[3].x = (1.0f/3.0f)*wx*theta_tmp[2].x;
theta_tmp[3].y = (1.0f/3.0f)*wy*theta_tmp[2].y;
theta_tmp[3].z = (1.0f/3.0f)*wz*theta_tmp[2].z;

theta_tmp[2].x = (1.0f/3.0f)*((wx+1.0f)*theta_tmp[1].x + (3.0f-wx)*theta_tmp[2].x);
theta_tmp[2].y = (1.0f/3.0f)*((wy+1.0f)*theta_tmp[1].y + (3.0f-wy)*theta_tmp[2].y);
theta_tmp[2].z = (1.0f/3.0f)*((wz+1.0f)*theta_tmp[1].z + (3.0f-wz)*theta_tmp[2].z);

theta_tmp[1].x = (1.0f/3.0f)*((wx+2.0f)*theta_tmp[0].x + (2.0f-wx)*theta_tmp[1].x);
theta_tmp[1].y = (1.0f/3.0f)*((wy+2.0f)*theta_tmp[0].y + (2.0f-wy)*theta_tmp[1].y);
theta_tmp[1].z = (1.0f/3.0f)*((wz+2.0f)*theta_tmp[0].z + (2.0f-wz)*theta_tmp[1].z);

theta_tmp[0].x = (1.0f/3.0f)*(1.0f-wx)*theta_tmp[0].x;
theta_tmp[0].y = (1.0f/3.0f)*(1.0f-wy)*theta_tmp[0].y;
theta_tmp[0].z = (1.0f/3.0f)*(1.0f-wz)*theta_tmp[0].z;

// Store theta_tmp and dtheta_tmp into global memory
int pos4 = pos*4;
thetax[pos4]   = theta_tmp[0].x;
thetax[pos4+1] = theta_tmp[1].x;
thetax[pos4+2] = theta_tmp[2].x;
thetax[pos4+3] = theta_tmp[3].x;

thetay[pos4]   = theta_tmp[0].y;
thetay[pos4+1] = theta_tmp[1].y;
thetay[pos4+2] = theta_tmp[2].y;
thetay[pos4+3] = theta_tmp[3].y;

thetaz[pos4]   = theta_tmp[0].z;
thetaz[pos4+1] = theta_tmp[1].z;
thetaz[pos4+2] = theta_tmp[2].z;
thetaz[pos4+3] = theta_tmp[3].z;

dthetax[pos4]   = dtheta_tmp[0].x;
dthetax[pos4+1] = dtheta_tmp[1].x;
dthetax[pos4+2] = dtheta_tmp[2].x;
dthetax[pos4+3] = dtheta_tmp[3].x;

dthetay[pos4]   = dtheta_tmp[0].y;
dthetay[pos4+1] = dtheta_tmp[1].y;
dthetay[pos4+2] = dtheta_tmp[2].y;
dthetay[pos4+3] = dtheta_tmp[3].y;

dthetaz[pos4]   = dtheta_tmp[0].z;
dthetaz[pos4+1] = dtheta_tmp[1].z;
dthetaz[pos4+2] = dtheta_tmp[2].z;
dthetaz[pos4+3] = dtheta_tmp[3].z;

pos += blockDim.x*gridDim.x;
}

}