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

__global__ void potForce(float * PairWise, int N, float * PotOut, float * ForceOut)
{
/*
PairWise - PairWise distances between atoms passed from global
N - # atoms
RowSize - # PairWise distances per block
RowCumSize - # nonzero RowSize array elements = # blocks launched in parallel
PotOut - Store the output Potential in global memory
ForceOut - Store the output Force in global memory along x, 1D array size N*N
*/
int bx = blockIdx.x;
int tx = threadIdx.x;
//Register variables to store pairwise separation
float delx;
float dely;
float delz;
float delr2, delrm6;
float Potential;
float Forcex;
float Forcey;
float Forcez;
int row = tx + bx*BLOCK_SIZE;
//if (row == 0) printf("I'm in 1! \n");
if (row < N*N)
{
delx = PairWise[row];
dely = PairWise[row + N*N];
delz = PairWise[row + N*N * 2];
delr2 = delx*delx + dely*dely + delz*delz;
delrm6 = __powf(delr2, (float)-3);
if (delr2 == 0.0) {
Potential = 0;
Forcex = 0;
Forcey = 0;
Forcez = 0;
}
else{
Potential = 4 * __fadd_rn(delrm6*delrm6, -1 * delrm6);
Forcex = -(delx / delr2) * 24 * __fadd_rn(2 * delrm6*delrm6, -1 * delrm6);
Forcey = -(dely / delr2) * 24 * __fadd_rn(2 * delrm6*delrm6, -1 * delrm6);
Forcez = -(delz / delr2) * 24 * __fadd_rn(2 * delrm6*delrm6, -1 * delrm6);
}
PotOut[row] = Potential;
ForceOut[row] = Forcex;
ForceOut[row + N*N] = Forcey;
ForceOut[row + N*N * 2] = Forcez;
}

}