#include "includes.h"
__global__ void OutputLayer(float* hiddenVotes, float* weight, int d_numHiddenNodes, float* d_votes){
int id = threadIdx.x + blockDim.x * blockIdx.x;

float total = 0.0f;

for (int i = 0; i < d_numHiddenNodes; ++i){
//printf("Hidden Votes: %i\n", hiddenVotes[i]);
//printf("Hidden Votes: %f, Weight: %f\n", hiddenVotes[i], weight[id * d_numHiddenNodes + i]);
total += hiddenVotes[i] * weight[id * d_numHiddenNodes + i];
//printf("Weight: %f", weight[id * d_numHiddenNodes + i]);
//printf("\n");
}

d_votes[id] = total;
//printf("Votes: %f\n", d_votes[id]);
}