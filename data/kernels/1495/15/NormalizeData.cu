#include "includes.h"
__global__ static void NormalizeData(float* ShiftData, float MaxValue, float MinValue, int FinalDataSize)
{
// 這邊是根據資料的最大最小值，去做 Normalize 資料
int id = blockIdx.y * gridDim.x * gridDim.z * blockDim.x +			// Y	=> Y * 250 * (2 * 1024)
blockIdx.x * gridDim.z * blockDim.x +							// X	=> X * (2 * 1024)
blockIdx.z * blockDim.x +										// Z	=> (Z1 * 1024 + Z2)
threadIdx.x;

// 例外判斷
if (id >= FinalDataSize)
{
printf("Normaliza Data 超出範圍\n");
return;
}

if (ShiftData[id] < MinValue)
ShiftData[id] = 0;
else if (ShiftData[id] > MaxValue)
ShiftData[id] = 1;
else
ShiftData[id] = (ShiftData[id] - MinValue) / (MaxValue - MinValue);

}