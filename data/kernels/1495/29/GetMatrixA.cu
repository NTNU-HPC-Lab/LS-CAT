#include "includes.h"
__global__ static void GetMatrixA(int* OCTData, float* MatrixA, int NumPolynomial, int OneDataSize)
{
// 這個 Function 是去取得 MatrixA 的值
int id = blockIdx.x * blockDim.x + threadIdx.x;

// 例外判斷 (理論上應該也是不會超過)
if (id >= (NumPolynomial + 1) * (NumPolynomial + 1))
{
printf("多項式 Fitting 有問題!\n");
return;
}

// 算 Index
int rowIndex = id % (NumPolynomial + 1);
int colsIndex = id / (NumPolynomial + 1);

// 做相加
float value = 0;
for (int i = 0; i < OneDataSize; i++)
{
// 抓出兩項的值
float FirstValue = (float)i / OneDataSize;
float SecondValue = (float)i / OneDataSize;
value += pow(FirstValue, NumPolynomial - rowIndex) * pow(SecondValue, NumPolynomial - colsIndex);
}
MatrixA[id] = value;
}