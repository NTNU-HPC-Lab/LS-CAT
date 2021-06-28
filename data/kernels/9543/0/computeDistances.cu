#include "includes.h"
using namespace std;
__global__ void computeDistances(int numInstances, int numAttributes, float* dataset, float* distances)
{
int tid = blockDim.x * blockIdx.x + threadIdx.x;
int row = tid / numInstances; // instance1Index
int column = tid - ((tid / numInstances) * numInstances); //instance2Index
if ((tid < numInstances * numInstances))
{
float sum = 0;
int instance1 = row * numAttributes;
int instance2 = column * numAttributes;
for (int atIdx = 1; atIdx < numAttributes; atIdx++) // start at 1 so we don't compare the id of each city
{
sum += ((dataset[instance1 + atIdx] - dataset[instance2 + atIdx]) * (dataset[instance1 + atIdx] - dataset[instance2 + atIdx]));
}
distances[row * numInstances + column] = (float) sqrt(sum);
distances[column * numInstances + row] = distances[row * numInstances + column]; //set the distance for the other half of the pair we just computed
}
}