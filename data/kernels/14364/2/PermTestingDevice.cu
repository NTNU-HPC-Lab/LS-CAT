#include "includes.h"
__global__ void PermTestingDevice(int numPermutations, int N, int V, int N_gp1, double *dataDevice, int *permutationsDevice, double *MaxTDevice)
{
int threadId = threadIdx.x + (blockIdx.x * blockDim.x); // Current Permutation
printf("Starting thread: %d \n", threadId);
int permutationsStart = threadId * N;
int i,j;
int N_gp2 = N - N_gp1; // Size of group 2
double group1Sum = 0;
double group2Sum = 0;
double group1SumSquared = 0; // Sum of all terms squared of group1 : x1^2 + x2^2 ...
double group2SumSquared = 0; // Sum of all terms squared of group2
double group1Mean = 0;
double group2Mean = 0;
double group1Var = 0;
double group2Var = 0;
double meanDifference = 0; // t-statistics numerator
double denominator = 0; // t-statistic denominator
double MaxT = 0;
double tStat = 0;

double voxelStatistic = 0;
int currSubject = 0;

/* For each voxel calculate a t-statistic*/
for(i = 0; i < V;i++)
{
group1Sum = 0;
group2Sum = 0;
group1SumSquared = 0;
group2SumSquared = 0;

/* Add statistics of the first group */
for(j = 0;j < N_gp1;j++)
{
currSubject = permutationsDevice[permutationsStart + j] - 1;
voxelStatistic = dataDevice[currSubject*V];
group1Sum = group1Sum + voxelStatistic;
group1SumSquared = group1SumSquared + voxelStatistic*voxelStatistic;
}

/* Add statistics of second group */
for(j = N_gp1; j < N; j++)
{
currSubject = permutationsDevice[permutationsStart + j] - 1;
voxelStatistic = dataDevice[currSubject*V];
group2Sum = group2Sum + voxelStatistic;
group2SumSquared = group2SumSquared + voxelStatistic*voxelStatistic;
}

group1Mean = group1Sum/N_gp1;
group2Mean = group2Sum/N_gp2;

group1Var = (group1SumSquared/N_gp1) - (group1Mean*group1Mean);
group2Var = (group2SumSquared/N_gp2) - (group2Mean*group2Mean);

meanDifference = group1Mean - group2Mean;
denominator = sqrt((group1Var / N_gp1) + (group2Var / N_gp2));

tStat = meanDifference/denominator;
if(tStat > MaxT)
{
MaxT = tStat;
}
}

MaxTDevice[threadId] = MaxT;
}