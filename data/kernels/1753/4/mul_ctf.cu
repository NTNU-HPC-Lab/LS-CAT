#include "includes.h"
__global__ void mul_ctf(float *image, int nx, int ny, float defocus, float cs, float voltage, float apix, float bfactor, float ampcont) {

// Block index
int bx = blockIdx.x;

// Thread index
int tx = threadIdx.x;

float x, y;

x = float(bx);
if (tx >= ny>>1) y = float(tx-ny);
else y = float(tx);
int index = bx*2+tx*(nx+2);

float ak = sqrt(x*x+y*y)/nx/apix;
float cst = cs*1.0e7f;
float wgh = ampcont/100.0f;
float phase = atan(wgh/sqrt(1.0f-wgh*wgh));
float lambda = 12.398f/sqrt(voltage*(1022.f+voltage));
float ak2 = ak*ak;
float g1 = defocus*1.0e4f*lambda*ak2;
float g2 = cst*lambda*lambda*lambda*ak2*ak2/2.0f;
float ctfv = sin(PI*(g1-g2)+phase);
if (bfactor != 0.0f)  ctfv *= exp(-bfactor*ak2/4.0f);

image[index] *= ctfv;
image[index+1] *= ctfv;
}