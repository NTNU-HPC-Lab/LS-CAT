#include "includes.h"
__global__ static void squaresSum(int *data, int *sum, clock_t *time)
{
int sum_t = 0;
clock_t start = clock();
for (int i = 0; i < DATA_SIZE; ++i) {
sum_t += data[i] * data[i];
}
*sum = sum_t;
*time = clock() - start;
}