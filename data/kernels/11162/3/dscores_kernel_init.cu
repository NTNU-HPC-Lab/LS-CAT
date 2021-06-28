#include "includes.h"

#define SIZ 20
#define num_inp 4

using namespace std;



typedef struct edge {
int first, second;
} edges;





__global__ void dscores_kernel_init(int * y, double * dscores, int size)
{
int i = blockIdx.x;
dscores[i*size + y[i]] -= 1;
}