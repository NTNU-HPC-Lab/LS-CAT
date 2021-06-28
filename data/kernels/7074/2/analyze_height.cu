#include "includes.h"
/*
cudaStructTest
testing/optimizing how to access/manipulate/return
structures in cuda.
*/



#define N 30
#define TRUE 1
#define FALSE 0
#define MAX_BLOCKS 65000
/*#define BLOCKS 2
#define THREADS 5*/

int cuda_setup(int computeCapability);

typedef struct{
int id;
int age;
int height;
} Person;


// Declare the Cuda kernels and any Cuda functions




__global__ void analyze_height(Person *people, int *statResults)
{
int id = threadIdx.x + blockIdx.x * blockDim.x;

if(id < N)
{
Person person = people[id];

if(person.height != 6)
{
statResults[id] = 1;
}
else
{
statResults[id] = 0;
}
}

}