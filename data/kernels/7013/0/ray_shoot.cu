#include "includes.h"
/* Vector addition deom on GPU

To compile: nvcc -o testprog1 testprog1.cu

*/

using namespace std;


#define FIRST_RUN 0
// Boundaries in physical units on the lens plane
const float WL  = 10.0;
const float XL1 = -WL;
const float XL2 =  WL;
const float YL1 = -WL;
const float YL2 =  WL;

// Source star parameters. You can adjust these if you like - it is
// interesting to look at the different lens images that result
const float rsrc = 0.1;      // radius
const float ldc  = 0.5;      // limb darkening coefficient
const float xsrc = 0.0;      // x and y centre on the map
const float ysrc = 0.0;

// Used to time code. OK for single threaded programs but not for
// multithreaded programs. See other demos for hints at timing CUDA
// code.
__global__ void ray_shoot(int *maxX, int *maxY, float *lens_scale, float *xlens, float *ylens, float*eps, int *num_lenses, float *dev_arr)
{
int threadBlockPos = (blockIdx.x * blockDim.x) + threadIdx.x;

int y = threadBlockPos / (*maxY);
int x = threadBlockPos - ((*maxX) * y);

const float rsrc2 = rsrc * rsrc;

float xl = XL1 + x * (*lens_scale);
float yl = YL1 + y * (*lens_scale);
float xs;
float ys;

float dx, dy, dr;

xs = xl;
ys = yl;

for(int p = 0; p < (*num_lenses); ++p){
dx = xl - xlens[p];
dy = yl - ylens[p];
dr = dx * dx + dy * dy;
xs -= eps[p] * dx / dr;
ys -= eps[p] * dy / dr;
}

float xd = xs - xsrc;
float yd = ys - ysrc;
float sep2 = (xd * xd) + (yd * yd);

if(sep2 < rsrc2){
float mu = sqrtf(1.0f-sep2/rsrc2);
dev_arr[threadBlockPos] = 1.0 - ldc * (1-mu);
}
}