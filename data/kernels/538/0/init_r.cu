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

__global__ void init_r(float* r, int N_cube){
int ix = threadIdx.x + blockDim.x* blockIdx.x;
int iy = threadIdx.y + blockDim.y* blockIdx.y;
int iz = threadIdx.z + blockDim.z* blockIdx.z;
int index = ix + iy*N_cube + iz * N_cube * N_cube;
if (ix < N_cube && iy < N_cube && iz<N_cube && index < N_ATOMS){
r[index] = L / 2.0 * (1.0 - float(2 * ix + 1) / N_cube);
r[index + N_ATOMS] = L / 2.0 * (1.0 - float(2 * iy + 1) / N_cube);
r[index + 2 * N_ATOMS] = L / 2.0 * (1.0 - float(2 * iz + 1) / N_cube);
}
}