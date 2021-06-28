#include "includes.h"

#define SIZ 20
#define num_inp 4

using namespace std;



typedef struct edge {
int first, second;
} edges;





__global__ void grads_w2_kernel(double * grads_W2,double * W2,double reg, int size)
{
int i = blockIdx.x;
int j = threadIdx.x;
grads_W2[i*size + j] += W2[i*size + j] * reg;
}