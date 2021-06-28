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

__global__ void forcered_simple(float * force, float * forcered){
int index = threadIdx.x + blockDim.x*blockIdx.x;
int i = 0;
int findex;
__shared__ float forcered_sh[3 * N_ATOMS];
//if (index == 0){ printf("In force reduction kernel! \n"); }
if (index < 3 * N_ATOMS){
forcered_sh[index] = 0.0f;
}
__syncthreads();
if (index < 3 * N_ATOMS){
findex = int(index / N_ATOMS)*N_ATOMS*N_ATOMS + index % N_ATOMS;
for (i = 0; i < N_ATOMS; i++){
forcered_sh[index] += force[findex + i*N_ATOMS];
}
}
__syncthreads();
if (index < 3 * N_ATOMS){
forcered[index] = forcered_sh[index];
}
/*if (index == 0){
printf("forcered [0]= %f \n", forcered[0]);
printf("forcered [2]= %f \n", forcered[2]);
printf("forcered [4]= %f \n \n", forcered[4]);
}*/
}