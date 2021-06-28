#include "includes.h"
__global__ void CalcInput(float* screen, float* weight, float* d_Votes, int stride){

//Current implementation, idk if it works. Probably doesn't, but it is worth a try, I think.
int id = threadIdx.x + blockDim.x * blockIdx.x;

d_Votes[id] = 0;

d_Votes[id] += screen[id] * weight[id];
d_Votes[id] += screen[id + 1] * weight[id + 1];
d_Votes[id] += screen[stride] * weight[stride];
d_Votes[id] += screen[stride + 1] * weight[stride + 1];

d_Votes[id] /= 4;
}