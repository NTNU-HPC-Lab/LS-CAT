#include "includes.h"
// Optimized using shared memory and on chip memory
// Compile source: $- nvcc src/TokamakSimulation.cu -o nBody -lglut -lm -lGLU -lGL
// Run Executable: $- ./nBody
//To stop hit "control c" in the window you launched it from.
//Make movies https://gist.github.com/JPEGtheDev/db078e1b066543ce40580060eee9c1bf

#define NR_NEUTRONS 8
#define NR_ELECTRONS 8
#define NR_PROTONS 8

//atomic mass (u)
#define MASS_PROTON 1.007276
#define MASS_NEUTRON 1.008664
#define MASS_ELECTRON 5.485799e-4

#define BLOCK 256

#define XWindowSize 2500
#define YWindowSize 2500

#define DRAW 10
#define DAMP 1.0

#define DT 0.001
#define STOP_TIME 10.0

#define G 6.67408E-11
#define H 1.0

#define EYE 8.5
#define FAR 80.0

#define SHAPE_CT 24
#define SHAPE_SIZE 256
#define PATH "./objects/Tokamak_256.obj" //256 vertices-shape (for array simplicity)
#define N 16*16*16

//***********************
// TODO:
//		Check units velocity calculation mag
//		ಠ_ಠ
//***********************

// Globals
float4 *p;
float3 *v, *f, *reactor,*r_GPU0, *r_GPU1;
float4 *p_GPU0, *p_GPU1;

__device__ float3 getMagForce(float4 p0, float3 v0, float3 dl_tail, float3 dl_head, float I){
//dl is the section of wire
float3 dB, dl;
dl.x = dl_head.x-dl_tail.x;
dl.y = dl_head.y-dl_tail.y;
dl.z = dl_head.z-dl_tail.z;

float rx = p0.x-dl_tail.x;
float ry = p0.y-dl_tail.y;
float rz = p0.z-dl_tail.z;

float r2 = rx*rx+ry*ry+rz*rz;
float inv_r2 = 1/r2;
float inv_r = 1/sqrtf(r2);
float3 rhat = {rx*inv_r, ry*inv_r, rz*inv_r};

//(dl cross rhat)/r2 = force
//gamma is mu0*I/4Pi which simplifies to Ie-7
float gamma = I;
dB.x = gamma*(dl.y*rhat.z-dl.z*rhat.y)*inv_r2;
dB.y = gamma*(dl.z*rhat.x-dl.x*rhat.z)*inv_r2;
dB.z = gamma*(dl.x*rhat.y-dl.y*rhat.x)*inv_r2;

return (dB);
}
__global__ void getForcesMag(float4 *g_pos, float3 *vel, float3 *force, int offset, float3 *g_reactor){

int id = threadIdx.x + blockDim.x*blockIdx.x;
float3 total_force, B, dB, dl_tail, dl_head, velMe;
float4 posMe;
__shared__ float3 shared_r[BLOCK];

total_force.x = B.x = 0.0;
total_force.y = B.y = 0.0;
total_force.z = B.z = 0.0;

posMe.x = g_pos[id+offset].x;
posMe.y = g_pos[id+offset].y;
posMe.z = g_pos[id+offset].z;
posMe.w = g_pos[id+offset].w;

velMe.x = vel[id].x;
velMe.y = vel[id].y;
velMe.z = vel[id].z;

for(int k=0;k<SHAPE_CT;k++){
shared_r[threadIdx.x] = g_reactor[threadIdx.x + blockDim.x*k];
__syncthreads();

for(int j = 1; j<=SHAPE_SIZE; j++){
dl_tail = shared_r[(j-1)];
dl_head = shared_r[(j%SHAPE_SIZE)];
dB = getMagForce(posMe, velMe, dl_tail, dl_head, 1.0); //current[i] =1

B.x += dB.x;
B.y += dB.y;
B.z += dB.z;
}
}

total_force.x = (velMe.y*B.z-velMe.z*B.y);
total_force.y = (velMe.z*B.x-velMe.x*B.z);
total_force.z = (velMe.x*B.y-velMe.y*B.x);

if(id<N){
force[id].x += total_force.x;
force[id].y += total_force.y;
force[id].z += total_force.z;
}
}