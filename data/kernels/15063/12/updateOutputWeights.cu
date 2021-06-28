#include "includes.h"
__global__ void updateOutputWeights(float* d_weights, float error, float lr, int keypress, int numHiddenNeurons, float* outputTotals, int numInput){
int id = threadIdx.x + blockDim.x * blockIdx.x;

int index = numHiddenNeurons * keypress + id;

float certainty = 0.0f;
for (int i = 0; i < numInput; ++i){
certainty += outputTotals[i];
}
certainty = outputTotals[keypress] / certainty;
//printf("Certainty: %f\n", certainty);

//int isPositive = 1;// d_weights[index] * 105 - 52.5;
//isPositive = min(isPositive, 1);
//isPositive = max(-1, isPositive);
//if (isPositive == 0){
//	isPositive = -1;
//}
//if(isPositive == 0)	printf("IsPositive: %i", isPositive);

//TODO test removing weight
float change = error * lr * d_weights[index] * certainty;

//printf("Error: %f, LR: %f, Weight: %f Change: %f\n", error, lr, d_weights[index], change);
d_weights[index] = d_weights[index] + change;

//Clamp
d_weights[index] = min(1.0f, d_weights[index]);
d_weights[index] = max(0.0f, d_weights[index]);
}