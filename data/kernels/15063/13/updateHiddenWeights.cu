#include "includes.h"
__global__ void updateHiddenWeights(float* d_weights, float error, float lr, int keyPress, float* d_outputweights, int screenSize, int numHiddenNeurons, float* d_bias, float* firstFire){
int id = threadIdx.x + blockDim.x * blockIdx.x;

float totalChange = 0.0f;
for (int i = 0; i < screenSize; ++i){
//Output weights stride is numNeurons, keypress is index into that section
float change = error * lr *d_outputweights[id * numHiddenNeurons + keyPress] * (firstFire[id] * 2 - 1);
totalChange += change;

d_weights[id * screenSize + i] = d_weights[id * screenSize + i] + change;

d_weights[id * screenSize + i] = min(1.0f, d_weights[id * screenSize + i]);
d_weights[id * screenSize + i] = max(0.0f, d_weights[id * screenSize + i]);
}
float biasChange = totalChange * -0.5f;
//printf("TotalChange: %f", biasChange);
d_bias[id] = d_bias[id] + biasChange;
}