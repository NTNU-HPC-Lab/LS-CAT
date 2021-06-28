#include "includes.h"
/* we need these includes for CUDA's random number stuff */

using namespace std;

#define MAX 26

//int a[1000]; //array of all possible password characters
int b[1000]; //array of attempted password cracks
unsigned long long tries = 0;
char alphabet[] = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' };
size_t result = 1000 * sizeof(float);

int *a = (int *) malloc(result);

__global__ void parallel_passwordCrack(int length,int*d_output,int *a)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
bool cracked = false;
char alphabetTable[] = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' };
int newB[1000];


__shared__ int nIter;
__shared__ int idT;
__shared__ long totalAttempt;

do{

if(idx == 0){
nIter = 0;
totalAttempt = 0;
}

newB[0]++;
for(int i =0; i<length; i++){
if (newB[i] >= 26 + alphabetTable[i]){
newB[i] -= 26;
newB[i+1]++;
}else break;
}

cracked=true;

for(int k=0; k<length; k++)
{
if(newB[k]!=a[k]){
cracked=false;
break;
}else
{
cracked = true;

}
}
if(cracked && nIter == 0){

idT = idx;
break;
}
else if(nIter){

break;
}

totalAttempt++;
}while(!cracked || !nIter);

if(idx == idT){
for(int i = 0; i< length; i++){

d_output[i] = newB[i];
}

}



}