#include "includes.h"
__global__ void d_sct_axinterp(float *sct3d, const float *scts1, const int4 *sctaxR, const float4 *sctaxW, const short *sn1_sn11, int NBIN, int NSN1, int SPN, int offtof)
{
//scatter crystal index
char ics = threadIdx.x;

//unscattered crystal index
char icu = 2 * threadIdx.y;

//span-1 sino index
short sni = blockIdx.x;

float tmp1, tmp2;

tmp1 = sctaxW[sni].x * scts1[NBIN*sctaxR[sni].x + icu*blockDim.x + ics] +
sctaxW[sni].y * scts1[NBIN*sctaxR[sni].y + icu*blockDim.x + ics] +
sctaxW[sni].z * scts1[NBIN*sctaxR[sni].z + icu*blockDim.x + ics] +
sctaxW[sni].w * scts1[NBIN*sctaxR[sni].w + icu*blockDim.x + ics];

//for the rest of the unscattered crystals (due to limited indexing of 1024 in a block)
icu += 1;
tmp2 = sctaxW[sni].x * scts1[NBIN*sctaxR[sni].x + icu*blockDim.x + ics] +
sctaxW[sni].y * scts1[NBIN*sctaxR[sni].y + icu*blockDim.x + ics] +
sctaxW[sni].z * scts1[NBIN*sctaxR[sni].z + icu*blockDim.x + ics] +
sctaxW[sni].w * scts1[NBIN*sctaxR[sni].w + icu*blockDim.x + ics];


//span-1 or span-11 scatter pre-sinogram interpolation
if (SPN == 1) {
sct3d[offtof + sni*NBIN + (icu - 1)*blockDim.x + ics] = tmp1;
sct3d[offtof + sni*NBIN + icu*blockDim.x + ics] = tmp2;
}
else if (SPN == 11) {
//only converting to span-11 when MRD<=60
if (sni<NSN1) {
short sni11 = sn1_sn11[sni];
atomicAdd(sct3d + offtof + sni11*NBIN + (icu - 1)*blockDim.x + ics, tmp1);
atomicAdd(sct3d + offtof + sni11*NBIN + icu*blockDim.x + ics, tmp2);
}
}

}