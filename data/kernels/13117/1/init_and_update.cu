#include "includes.h"
/**********************************************************************
* DESCRIPTION:
*   Serial Concurrent Wave Equation - C Version
*   This program implements the concurrent wave equation
*********************************************************************/

#define MAXPOINTS 1000000
#define MAXSTEPS 1000000
#define MINPOINTS 20
#define PI 3.14159265

#define BLOCK_SIZE 512

void check_param(void);
void printfinal (void);


/**********************************************************************
*     Initialize points on line
*********************************************************************/

/**********************************************************************
*     Update all values along line a specified number of times
*********************************************************************/
__global__ void init_and_update (float *values_d, int tpoints, int nsteps){
int idx = threadIdx.x + blockIdx.x * BLOCK_SIZE;

if(idx <= 1 || idx >= tpoints)
return;

float old_v, v, new_v;

float x, tmp;
tmp = tpoints - 1;
x = idx / tmp;

v = sin(2.0 * PI * x);
old_v = v;

for (int i = 1; i <= nsteps; i++){
new_v = (2.0 * v) - old_v + (0.09 * (-2.0 * v));
old_v = v;
v = new_v;
}

values_d[idx] = v;

}