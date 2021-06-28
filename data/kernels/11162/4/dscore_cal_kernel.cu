#include "includes.h"

#define SIZ 20
#define num_inp 4

using namespace std;



typedef struct edge {
int first, second;
} edges;





__global__ void dscore_cal_kernel(double * dscores, int num_inputs, int size)
{
int i = blockIdx.x;
int j = threadIdx.x;
dscores[i*size + j] /= num_inputs;
}