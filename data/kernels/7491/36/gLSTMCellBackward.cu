#include "includes.h"
__device__ inline float stableSigmoid(float x) {
if(x >= 0) {
float z = expf(-x);
return 1.0 / (1.0 + z);
} else {
float z = expf(x);
return z / (1.0 + z);
}
}
__global__ void gLSTMCellBackward(float* outCell, float* outXW, float* outSU, float* outB, const float* cell, const float* xW, const float* sU, const float* b, const float* mask, const float* adj, size_t rows, size_t cols) {
for(int bid = 0; bid < rows; bid += gridDim.x) {
int j = bid + blockIdx.x;
if(j < rows) {
float m = !mask || mask[j];

float* rowOutCell = outCell + j * cols;
float* rowOutXW = outXW + j * cols * 4;
float* rowOutSU = outSU + j * cols * 4;

const float* rowCell = cell + j * cols;
const float* xWrow = xW + j * cols * 4;
const float* sUrow = sU + j * cols * 4;

const float* rowAdj = adj + j * cols;

for(int tid = 0; tid < cols; tid += blockDim.x) {
int i = tid + threadIdx.x;
if(i < cols) {
float gf = stableSigmoid(xWrow[i] + sUrow[i] + b[i]);

int k = i + cols;
float gi = stableSigmoid(xWrow[k] + sUrow[k] + b[k]);

int l = i + 2 * cols;
float gc = tanhf(xWrow[l] + sUrow[l] + b[l]);

float adj = rowAdj[i];

// dc/dc_{t-1}
if(outCell)
rowOutCell[i] += (m * gf - m + 1) * adj;

// dc/d(b_f) = dc/d(xW_f) ...
float dcdxf = m * rowCell[i] * gf * (1 - gf) * adj;
if(outXW)
rowOutXW[i] += dcdxf;
if(outSU)
rowOutSU[i] += dcdxf;
if(outB)
atomicAdd(outB + i, dcdxf);

// dc/d(b_i) ...
float dcdb_i = m * gc * gi * (1 - gi) * adj;
if(outXW)
rowOutXW[k] += dcdb_i;
if(outSU)
rowOutSU[k] += dcdb_i;
if(outB)
atomicAdd(outB + k, dcdb_i);

// dc/d(b_c) ...
float dcdxc = m * gi * (1 - gc * gc) * adj;
if(outXW)
rowOutXW[l] += dcdxc;
if(outSU)
rowOutSU[l] += dcdxc;
if(outB)
atomicAdd(outB + l, dcdxc);
}
}
}
}
}