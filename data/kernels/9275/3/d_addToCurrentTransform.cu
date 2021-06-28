#include "includes.h"
__global__ void d_addToCurrentTransform(float* d_currentTransform, float* d_invViewMatrix) {
float result[12] = {0.f};
for (int i = 0; i < 3; ++i) {
for (int j = 0; j < 4; ++j) {
for (int k = 0; k < 4; ++k) {
result[i * 4 + j] += d_invViewMatrix[i * 4 + k] * d_currentTransform[k * 4 + j];
}
}
}
for (int i = 0; i < 12; ++i) {	// The last row of currentTransform remains (0,0,0,1)
d_currentTransform[i] = result[i];
}
}