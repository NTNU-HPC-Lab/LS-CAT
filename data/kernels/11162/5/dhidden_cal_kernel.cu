#include "includes.h"

#define SIZ 20
#define num_inp 4

using namespace std;



typedef struct edge {
int first, second;
} edges;





__global__ void dhidden_cal_kernel(double * a1,double * dhidden,int size)
{
int i = blockIdx.x;
int j = threadIdx.x;
if (a1[i*size + j] <= 0)
{
dhidden[i*size + j] = 0;
}
}