#include "includes.h"
__global__ void FirstHidden(float* input, float* weight, float* bias, int d_numVotes, float* d_votes){
int id = threadIdx.x + blockDim.x * blockIdx.x;

float total = 0.0f;

//printf("Num Votes: %i", d_numVotes);

for (int i = 0; i < d_numVotes; ++i){
//if (weight[id*d_numVotes + i] > 0) printf("Weight higher than 0: %f", weight[id*d_numVotes + i]);
//if (input[i] > 0) printf("Input: %f ", input[i]);
//printf("Weight: %f\n", weight[id * d_numVotes + i]);
//printf("Input: %f, Weight: %f\n", input[i], weight[id * d_numVotes + i]);
float sig = input[i] * weight[id * d_numVotes + i];
total += sig;// (1 / (1 + exp(-sig)));
}

//total /= d_numVotes;

printf("total: %f, Bias: %f\n", total, bias[id]);
total += bias[id];
//printf("Total: %f\n", total);
//printf("Bias: %f\n", bias[id]);
total = (1 / (1 + exp(-total)));
//total = ((int)(total)) % 3;
//if (total < 0.1) printf("Total %i: %f\n", id, total);

//printf("Total: %f\n", total);
d_votes[id] = total;
}