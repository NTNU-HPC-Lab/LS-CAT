#include "includes.h"
__global__ void gArgmax(float* out, const float* data, size_t rows, size_t cols) {
size_t row = blockIdx.x;
size_t startInd = row * cols;
float maxScore = -99999;
size_t maxInd;
for(size_t col = 0; col < cols; ++col) {
size_t ind = startInd + col;
float score = data[ind];
if(score > maxScore) {
maxScore = score;
maxInd = col;
}
}
out[row] = maxInd;
}