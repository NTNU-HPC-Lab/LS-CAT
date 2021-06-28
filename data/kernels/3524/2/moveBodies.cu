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

__global__ void moveBodies(float4 *g_pos, float4 *d_pos, float3 *vel, float3 * force, int offset){
int id = threadIdx.x + blockDim.x*blockIdx.x;
if(id < N){
vel[id].x += ((force[id].x-DAMP*vel[id].x)/d_pos[id].w)*DT;
vel[id].y += ((force[id].y-DAMP*vel[id].y)/d_pos[id].w)*DT;
vel[id].z += ((force[id].z-DAMP*vel[id].z)/d_pos[id].w)*DT;

d_pos[id].x += vel[id].x*DT;
d_pos[id].y += vel[id].y*DT;
d_pos[id].z += vel[id].z*DT;

g_pos[id+offset].x = d_pos[id].x;
g_pos[id+offset].y = d_pos[id].y;
g_pos[id+offset].z = d_pos[id].z;
}
}