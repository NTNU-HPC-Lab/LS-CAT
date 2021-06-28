#include "includes.h"
//CUDA reduction algorithm. simple approach
//Tom Dale
//11-20-18


using namespace std;
#define N 100000//number of input values
#define R 100//reduction factor
#define F (1+((N-1)/R))//how many values will be in the final output


//basicRun will F number of threads go through R number of values and put the average in z[tid]




__global__ void basicRun(double *a,double *z){
int tid = blockDim.x*blockIdx.x + threadIdx.x;
if(tid > F) return;
double avg=0;
for(int i= 0;i<R;i++){//get sum of input values in this threads domain
avg += a[i+tid*R];
}
z[tid]=avg/R;//divide sum by total number of input values to get average
}