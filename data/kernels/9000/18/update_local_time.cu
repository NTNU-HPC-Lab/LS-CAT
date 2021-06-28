#include "includes.h"
__global__ void update_local_time(int *next, double *local_time, double GTIME){

unsigned int gtid = blockIdx.x*blockDim.x + threadIdx.x;
int who = next[gtid];

if(who < 0)
return;

local_time[who] = GTIME;

}