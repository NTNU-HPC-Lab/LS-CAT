#include "includes.h"
__global__ void Prepare_1_MeansForJoin(float* input, int c_src1, int c_src2, int c_n, float* delta, int imageWidth, int imageHeight)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;

int size = imageWidth * imageHeight;

if (id < size)
{
int px = id % imageWidth;
int py = id / imageWidth;

bool insideSrc1 = delta[c_src1 * NUM_SUMS * size + 4 * size + id] != 0;
bool insideSrc2 = delta[c_src2 * NUM_SUMS * size + 4 * size + id] != 0;

if (input[id] > 0 && (insideSrc1 || insideSrc2)) {

float2 pixPos = {  2.0f * px / imageWidth - 1,  2.0f * py / imageHeight - 1};

//w * pos
delta[c_n * NUM_SUMS * size + 0 * size + id] = input[id] * pixPos.x;
delta[c_n * NUM_SUMS * size + 1 * size + id] = input[id] * pixPos.y;

//w * pos^2
delta[c_n * NUM_SUMS * size + 2 * size + id] = input[id] * pixPos.x * pixPos.x;
delta[c_n * NUM_SUMS * size + 3 * size + id] = input[id] * pixPos.y * pixPos.y;

//w
delta[c_n * NUM_SUMS * size + 4 * size + id] = input[id];
}
else
{
delta[c_n * NUM_SUMS * size + 0 * size + id] = 0;
delta[c_n * NUM_SUMS * size + 1 * size + id] = 0;

//w * pos^2
delta[c_n * NUM_SUMS * size + 2 * size + id] = 0;
delta[c_n * NUM_SUMS * size + 3 * size + id] = 0;

//w
delta[c_n * NUM_SUMS * size + 4 * size + id] = 0;
}
}
}