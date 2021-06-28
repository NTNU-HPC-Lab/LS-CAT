#include "includes.h"
// Include files


// Parameters

#define N_ATOMS 343
#define MASS_ATOM 1.0f
#define time_step 0.01f
#define L 10.5f
#define T 0.728f
#define NUM_STEPS 10000

const int BLOCK_SIZE = 1024;
//const int L = ;
const int scheme = 1; // 0 for explicit, 1 for implicit

/*************************************************************************************************************/
/*************								INITIALIZATION CODE										**********/
/*************************************************************************************************************/

__device__ float PutInBox(float r){
if (fabs(r) > L / 2.0)
r += (2 * (r < 0) - 1)*ceil((fabs(r) - L / 2.0f) / L)*L;
return r;
}
__global__ void kinematics(float* positions, float* force, float* vel, int len){
int tx = threadIdx.x;
int bx = blockIdx.x;
int index = bx*blockDim.x + tx;
float tempr;
//if (index == 0){ printf("You have been trolled! \n"); }
if (index < len){
tempr = positions[index] + 0.5f * force[index] / MASS_ATOM * time_step*time_step + vel[index] * time_step;
positions[index] = PutInBox(tempr);
vel[index] += force[index] / MASS_ATOM * time_step;
}
}