#include "includes.h"
__global__ static void ConnectPointsStatus(int* PointType_BestN, int* ConnectStatus, int size, int rows, int ChooseBestN, int ConnectRadius)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id >= size * rows * ChooseBestN)						// 判斷是否超出大小
return;

// 算 Index
int sizeIndex = id / (rows * ChooseBestN);
int tempID = id % (rows * ChooseBestN);
int rowIndex = tempID / ChooseBestN;
int chooseIndex = tempID % ChooseBestN;

// 代表這個點沒有有效的點
if (PointType_BestN[sizeIndex * rows * ChooseBestN + rowIndex * ChooseBestN + chooseIndex] == -1)
return;

// 如果是有效的點，就繼續往下追
int finalPos = min(rowIndex + ConnectRadius, rows);		// 截止條件
for (int i = rowIndex + 1; i < finalPos; i++)
{
for (int j = 0; j < ChooseBestN; j++)
{
// 下一個點的位置 (第 i 個 row 的點)
// 然後的第 1 個點
if (PointType_BestN[sizeIndex * rows * ChooseBestN + i * ChooseBestN + j] != -1)
{
// 前面項為現在這個點
// 後面項為往下的點
int diffX = PointType_BestN[sizeIndex * rows * ChooseBestN + rowIndex * ChooseBestN + chooseIndex] -
PointType_BestN[sizeIndex * rows * ChooseBestN + i * ChooseBestN + j];
int diffY = i - rowIndex;
int Radius = diffX * diffX + diffY * diffY;

// 0 沒有用到喔
if (Radius < ConnectRadius * ConnectRadius)
{
// 張數的位移 + Row 的位移 + 現在在 Top N 的點 + 半徑的位移 + 往下 Top N 的結果
int index = sizeIndex * rows * ChooseBestN * ConnectRadius * ChooseBestN +			// 張數
rowIndex * ChooseBestN * ConnectRadius * ChooseBestN +					// Row
chooseIndex * ConnectRadius * ChooseBestN +								// 現在在 Top N
(i - rowIndex) * ChooseBestN +											// 半徑
j;
ConnectStatus[index] = Radius;
}
}
}
}
}