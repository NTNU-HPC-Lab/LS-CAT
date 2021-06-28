#include "includes.h"
__global__ static void gaussdensity_direct_alt(int natoms, const float4 *xyzr, float gridspacing, unsigned int z, float *densitygrid) {
unsigned int xindex  = (blockIdx.x * blockDim.x) * DUNROLLX + threadIdx.x;
unsigned int yindex  = (blockIdx.y * blockDim.y) + threadIdx.y;
unsigned int zindex  = (blockIdx.z * blockDim.z) + threadIdx.z;
unsigned int outaddr =
((gridDim.x * blockDim.x) * DUNROLLX) * (gridDim.y * blockDim.y) * zindex +
((gridDim.x * blockDim.x) * DUNROLLX) * yindex + xindex;
zindex += z;

float coorx = gridspacing * xindex;
float coory = gridspacing * yindex;
float coorz = gridspacing * zindex;

float densityvalx1=0.0f;
float densityvalx2=0.0f;
#if DUNROLLX >= 4
float densityvalx3=0.0f;
float densityvalx4=0.0f;
#endif
#if DUNROLLX >= 8
float densityvalx5=0.0f;
float densityvalx6=0.0f;
float densityvalx7=0.0f;
float densityvalx8=0.0f;
#endif

float gridspacing_coalesce = gridspacing * DBLOCKSZX;

int atomid;
for (atomid=0; atomid<natoms; atomid++) {
float4 atom = xyzr[atomid];
float dy = coory - atom.y;
float dz = coorz - atom.z;
float dyz2 = dy*dy + dz*dz;

float dx1 = coorx - atom.x;
float r21 = (dx1*dx1 + dyz2) * atom.w;
densityvalx1 += exp2f(-r21);

float dx2 = dx1 + gridspacing_coalesce;
float r22 = (dx2*dx2 + dyz2) * atom.w;
densityvalx2 += exp2f(-r22);

#if DUNROLLX >= 4
float dx3 = dx2 + gridspacing_coalesce;
float r23 = (dx3*dx3 + dyz2) * atom.w;
densityvalx3 += exp2f(-r23);

float dx4 = dx3 + gridspacing_coalesce;
float r24 = (dx4*dx4 + dyz2) * atom.w;
densityvalx4 += exp2f(-r24);
#endif
#if DUNROLLX >= 8
float dx5 = dx4 + gridspacing_coalesce;
float r25 = (dx5*dx5 + dyz2) * atom.w;
densityvalx5 += exp2f(-r25);

float dx6 = dx5 + gridspacing_coalesce;
float r26 = (dx6*dx6 + dyz2) * atom.w;
densityvalx6 += exp2f(-r26);

float dx7 = dx6 + gridspacing_coalesce;
float r27 = (dx7*dx7 + dyz2) * atom.w;
densityvalx7 += exp2f(-r27);

float dx8 = dx7 + gridspacing_coalesce;
float r28 = (dx8*dx8 + dyz2) * atom.w;
densityvalx8 += exp2f(-r28);
#endif
}

densitygrid[outaddr             ] += densityvalx1;
densitygrid[outaddr+1*DBLOCKSZX] += densityvalx2;
#if DUNROLLX >= 4
densitygrid[outaddr+2*DBLOCKSZX] += densityvalx3;
densitygrid[outaddr+3*DBLOCKSZX] += densityvalx4;
#endif
#if DUNROLLX >= 8
densitygrid[outaddr+4*DBLOCKSZX] += densityvalx5;
densitygrid[outaddr+5*DBLOCKSZX] += densityvalx6;
densitygrid[outaddr+6*DBLOCKSZX] += densityvalx7;
densitygrid[outaddr+7*DBLOCKSZX] += densityvalx8;
#endif
}