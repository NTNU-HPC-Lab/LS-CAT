#include "includes.h"
__global__ static void gaussdensity_direct_tex(int natoms, const float4 *xyzr, const float4 *colors, float gridspacing, unsigned int z, float *densitygrid, float3 *voltexmap, float invisovalue) {
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
float3 densitycolx1;
densitycolx1=make_float3(0.0f, 0.0f, 0.0f);
float3 densitycolx2=densitycolx1;

#if DUNROLLX >= 4
float densityvalx3=0.0f;
float densityvalx4=0.0f;
float3 densitycolx3=densitycolx1;
float3 densitycolx4=densitycolx1;
#endif
#if DUNROLLX >= 8
float densityvalx5=0.0f;
float densityvalx6=0.0f;
float densityvalx7=0.0f;
float densityvalx8=0.0f;

float3 densitycolx5=densitycolx1;
float3 densitycolx6=densitycolx1;
float3 densitycolx7=densitycolx1;
float3 densitycolx8=densitycolx1;
#endif

float gridspacing_coalesce = gridspacing * DBLOCKSZX;

int atomid;
for (atomid=0; atomid<natoms; atomid++) {
float4 atom = xyzr[atomid];
float4 color = colors[atomid];

float dy = coory - atom.y;
float dz = coorz - atom.z;
float dyz2 = dy*dy + dz*dz;

float dx1 = coorx - atom.x;
float r21 = (dx1*dx1 + dyz2) * atom.w;
float tmp1 = exp2f(-r21);
densityvalx1 += tmp1;
tmp1 *= invisovalue;
densitycolx1.x += tmp1 * color.x;
densitycolx1.y += tmp1 * color.y;
densitycolx1.z += tmp1 * color.z;

float dx2 = dx1 + gridspacing_coalesce;
float r22 = (dx2*dx2 + dyz2) * atom.w;
float tmp2 = exp2f(-r22);
densityvalx2 += tmp2;
tmp2 *= invisovalue;
densitycolx2.x += tmp2 * color.x;
densitycolx2.y += tmp2 * color.y;
densitycolx2.z += tmp2 * color.z;

#if DUNROLLX >= 4
float dx3 = dx2 + gridspacing_coalesce;
float r23 = (dx3*dx3 + dyz2) * atom.w;
float tmp3 = exp2f(-r23);
densityvalx3 += tmp3;
tmp3 *= invisovalue;
densitycolx3.x += tmp3 * color.x;
densitycolx3.y += tmp3 * color.y;
densitycolx3.z += tmp3 * color.z;

float dx4 = dx3 + gridspacing_coalesce;
float r24 = (dx4*dx4 + dyz2) * atom.w;
float tmp4 = exp2f(-r24);
densityvalx4 += tmp4;
tmp4 *= invisovalue;
densitycolx4.x += tmp4 * color.x;
densitycolx4.y += tmp4 * color.y;
densitycolx4.z += tmp4 * color.z;
#endif
#if DUNROLLX >= 8
float dx5 = dx4 + gridspacing_coalesce;
float r25 = (dx5*dx5 + dyz2) * atom.w;
float tmp5 = exp2f(-r25);
densityvalx5 += tmp5;
tmp5 *= invisovalue;
densitycolx5.x += tmp5 * color.x;
densitycolx5.y += tmp5 * color.y;
densitycolx5.z += tmp5 * color.z;

float dx6 = dx5 + gridspacing_coalesce;
float r26 = (dx6*dx6 + dyz2) * atom.w;
float tmp6 = exp2f(-r26);
densityvalx6 += tmp6;
tmp6 *= invisovalue;
densitycolx6.x += tmp6 * color.x;
densitycolx6.y += tmp6 * color.y;
densitycolx6.z += tmp6 * color.z;

float dx7 = dx6 + gridspacing_coalesce;
float r27 = (dx7*dx7 + dyz2) * atom.w;
float tmp7 = exp2f(-r27);
densityvalx7 += tmp7;
tmp7 *= invisovalue;
densitycolx7.x += tmp7 * color.x;
densitycolx7.y += tmp7 * color.y;
densitycolx7.z += tmp7 * color.z;

float dx8 = dx7 + gridspacing_coalesce;
float r28 = (dx8*dx8 + dyz2) * atom.w;
float tmp8 = exp2f(-r28);
densityvalx8 += tmp8;
tmp8 *= invisovalue;
densitycolx8.x += tmp8 * color.x;
densitycolx8.y += tmp8 * color.y;
densitycolx8.z += tmp8 * color.z;
#endif
}

densitygrid[outaddr             ] += densityvalx1;
voltexmap[outaddr             ].x += densitycolx1.x;
voltexmap[outaddr             ].y += densitycolx1.y;
voltexmap[outaddr             ].z += densitycolx1.z;

densitygrid[outaddr+1*DBLOCKSZX] += densityvalx2;
voltexmap[outaddr+1*DBLOCKSZX].x += densitycolx2.x;
voltexmap[outaddr+1*DBLOCKSZX].y += densitycolx2.y;
voltexmap[outaddr+1*DBLOCKSZX].z += densitycolx2.z;

#if DUNROLLX >= 4
densitygrid[outaddr+2*DBLOCKSZX] += densityvalx3;
voltexmap[outaddr+2*DBLOCKSZX].x += densitycolx3.x;
voltexmap[outaddr+2*DBLOCKSZX].y += densitycolx3.y;
voltexmap[outaddr+2*DBLOCKSZX].z += densitycolx3.z;

densitygrid[outaddr+3*DBLOCKSZX] += densityvalx4;
voltexmap[outaddr+3*DBLOCKSZX].x += densitycolx4.x;
voltexmap[outaddr+3*DBLOCKSZX].y += densitycolx4.y;
voltexmap[outaddr+3*DBLOCKSZX].z += densitycolx4.z;
#endif
#if DUNROLLX >= 8
densitygrid[outaddr+4*DBLOCKSZX] += densityvalx5;
voltexmap[outaddr+4*DBLOCKSZX].x += densitycolx5.x;
voltexmap[outaddr+4*DBLOCKSZX].y += densitycolx5.y;
voltexmap[outaddr+4*DBLOCKSZX].z += densitycolx5.z;

densitygrid[outaddr+5*DBLOCKSZX] += densityvalx6;
voltexmap[outaddr+5*DBLOCKSZX].x += densitycolx6.x;
voltexmap[outaddr+5*DBLOCKSZX].y += densitycolx6.y;
voltexmap[outaddr+5*DBLOCKSZX].z += densitycolx6.z;

densitygrid[outaddr+6*DBLOCKSZX] += densityvalx7;
voltexmap[outaddr+6*DBLOCKSZX].x += densitycolx7.x;
voltexmap[outaddr+6*DBLOCKSZX].y += densitycolx7.y;
voltexmap[outaddr+6*DBLOCKSZX].z += densitycolx7.z;

densitygrid[outaddr+7*DBLOCKSZX] += densityvalx8;
voltexmap[outaddr+7*DBLOCKSZX].x += densitycolx8.x;
voltexmap[outaddr+7*DBLOCKSZX].y += densitycolx8.y;
voltexmap[outaddr+7*DBLOCKSZX].z += densitycolx8.z;
#endif
}