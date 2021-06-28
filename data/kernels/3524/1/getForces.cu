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

__device__ float3 getBodyBodyForce(float4 p0, float4 p1){
float3 f;
float dx = p1.x - p0.x;
float dy = p1.y - p0.y;
float dz = p1.z - p0.z;
float r2 = dx*dx + dy*dy + dz*dz;
float inv_r = 1/sqrt(r2);

float force  = (G*p0.w*p1.w)/(r2);// - (H*p0.w*p1.w)/(r2*r2);

f.x = force*dx*inv_r;
f.y = force*dy*inv_r;
f.z = force*dz*inv_r;

return(f);
}
__global__ void getForces(float4 *g_pos, float3 *force, int offset, int device_ct){
int ii;
float3 force_b2b, forceSum;
float4 posMe;
__shared__ float4 shPos[BLOCK];
int id = threadIdx.x + blockDim.x*blockIdx.x;

forceSum.x = 0.0;
forceSum.y = 0.0;
forceSum.z = 0.0;

posMe.x = g_pos[id+offset].x;
posMe.y = g_pos[id+offset].y;
posMe.z = g_pos[id+offset].z;
posMe.w = g_pos[id+offset].w;

for(int j=0; j < gridDim.x*device_ct; j++)
{
shPos[threadIdx.x] = g_pos[threadIdx.x + blockDim.x*j];
__syncthreads();

#pragma unroll 32
for(int i=0; i < blockDim.x; i++)
{
ii = i + blockDim.x*j;
if(ii != id+offset && ii < N)
{
force_b2b = getBodyBodyForce(posMe, shPos[i]);
forceSum.x += force_b2b.x;
forceSum.y += force_b2b.y;
forceSum.z += force_b2b.z;
}
}
}

if(id <N){
force[id].x = forceSum.x;
force[id].y = forceSum.y;
force[id].z = forceSum.z;
}
}