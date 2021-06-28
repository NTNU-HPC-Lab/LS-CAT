#include "includes.h"
__device__ float computeDeterminant (float e00, float e01, float e02, float e10, float e11, float e12, float e20, float e21, float e22)
{
return e00*e11*e22-e00*e12*e21+e10*e21*e02-e10*e01*e22+e20*e01*e12-e20*e11*e02;
}
__global__ void hessianKernel ( float *d_output, const float *d_gxx, const float *d_gxy, const float *d_gxz, const float *d_gyy, const float *d_gyz, const float *d_gzz, float sigma, int imageW, int imageH, int imageD )
{
int n_blocks_per_width = imageW/blockDim.x;
int z = (int)ceilf(blockIdx.x/n_blocks_per_width);
int y = blockIdx.y*blockDim.y + threadIdx.y;
int x = (blockIdx.x - z*n_blocks_per_width)*blockDim.x + threadIdx.x;
int i = z*imageW*imageH + y*imageW + x;

// // //Brute force eigen-values computation
float a0, b0, c0, e0, f0, k0;
a0 = -d_gxx[i]; b0 = -d_gxy[i]; c0 = -d_gxz[i];
e0 = -d_gyy[i]; f0 = -d_gyz[i]; k0 = -d_gzz[i];


// http://en.wikipedia.org/wiki/Eigenvalue_algorithm
//Oliver K. Smith: Eigenvalues of a symmetric 3 × 3 matrix. Commun. ACM 4(4): 168 (1961)
float m = (a0+e0+k0)/3;
float q = computeDeterminant
(a0-m, b0, c0, b0, e0-m, f0, c0, f0, k0-m)/2;
float p = (a0-m)*(a0-m) + b0*b0 + c0*c0 + b0*b0 + (e0-m)*(e0-m) +
f0*f0 + c0*c0 + f0*f0 + (k0-m)*(k0-m);
p = p / 6;
float phi = 1.f/3.f*atan(sqrt(p*p*p-q*q)/q);
if(phi<0)
phi=phi+3.14159f/3;

float eig1 = m + 2*sqrt(p)*cos(phi);
float eig2 = m - sqrt(p)*(cos(phi) + sqrt(3.0f)*sin(phi));
float eig3 = m - sqrt(p)*(cos(phi) - sqrt(3.0f)*sin(phi));

if( (eig1 > eig2) & (eig1 > eig3))
d_output[i] = eig1*sigma*sigma;
if( (eig2 > eig1) & (eig2 > eig3))
d_output[i] = eig2*sigma*sigma;
if( (eig3 > eig2) & (eig3 > eig1))
d_output[i] = eig3*sigma*sigma;
}