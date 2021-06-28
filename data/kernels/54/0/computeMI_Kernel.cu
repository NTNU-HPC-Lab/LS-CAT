#include "includes.h"

cudaError_t calcCuda(float *output, const uint8_t *input, const size_t *size);

//********************************************************************************************\\
static cudaDeviceProp deviceProperties_;
__global__ void computeMI_Kernel(float *MIs, uint8_t *input, int rowCount, int colCount, int *countNZ, int offset)
{
int i = threadIdx.x + blockIdx.x * blockDim.x + offset;
if (i > rowCount*(rowCount-1)/2) return;
int joints[2][2] = { 0 };
int countNZA , countNZB, a, b, j, k;
float joint;
uint8_t *inputA = 0,
*inputB = 0;

MIs += i;
a = 1;
b = 0;
for (j = rowCount - 1; j > 1; j--)
{
if (i < j) break;

a++;
b++;
i -= j;
}
j = b;
i += a;

//  MIs += i + j*rowCount;
*MIs = 0;
// *MIs = i * 1000 + j;

//  for (j = 0; j < i; j++, MIs += colCount)
{

inputA = input + i;
inputB = input + j;
countNZA = countNZ[i];
countNZB = countNZ[j];
for (k = 0; k < colCount; k++, inputA += rowCount, inputB += rowCount)
{
joints[*inputA][*inputB]++;
}

for (k = 0; k < 4; k++)
{
a = k % 2;
b = k / 2;

joint = joints[a][b];
if (joint == 0)
continue;
joint /= colCount;
if (a) a = countNZA;
else a = colCount - countNZA;
if (b) b = countNZB;
else b = colCount - countNZB;

*MIs += joint * log2f(joint / ((float)a / colCount) / ((float)b / colCount));
}
}
/*  size_t i, t_count, b_count;
t_count = rowCount > deviceProperties_.maxThreadsPerBlock ? deviceProperties_.maxThreadsPerBlock : rowCount;
b_count = rowCount / deviceProperties_.maxThreadsPerBlock + 1;
computeMI_Kernel << <b_count, t_count >> > (MIs, input, i, rowCount, colCount, countNZ)
*/
}//********************************************************************************************\\