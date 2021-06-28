#include "includes.h"
using namespace std;
/*
const int sizePoint = 5;
const int sizeIndividum = 5;
const int mathValueMutation = 5;
const float dispersionMutation = 5.0f;
const int powCount = 3;
const float randMaxCount = 20.0f;
*/

const int sizePoint = 500;
const int sizeIndividum = 1000;
const int mathValueMutation = 5;
const float dispersionMutation = 5.0f;
const int powCount = 3;
const float randMaxCount = 20.0f;
const int maxPokoleney = 30;



__global__ void errorsKernel(float *points, float *individs, float *errors, int powCount, int sizePoint)
{

int id = threadIdx.x;
float ans = 0;
int x = 1;
for (int i = 0; i < sizePoint; i++)
{
for (int j = 0; j < powCount; j++)
{
for (int k = 0; k < j; k++)
{
x *= i;
}
x *= individs[id*powCount + j];
ans += x;
x = 1;
}

ans = points[i] - ans;
errors[id] += sqrt(ans * ans);
ans = 0;
}
}