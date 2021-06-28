#include "includes.h"

#define BOOL int
#define TRUE 1
#define FALSE 0
#define populationSize 128
#define chromosomeSize 10
#define maxGeneration 500
#define crossRate 0.8
#define mutationRate 0.01
#define eliteCount 0.05*populationSize



//typedef float float;
float LB[10] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}; //lower bound
float UB[10] = {5, 4, 5, 4, 5, 5, 5, 5, 5, 4}; //upper bound
float *a;  //Tzaihe
float *aa;  //yingliK
float *aaa; //Tyingli
int aRow;
int aaaRow;
float Dysum[9];

__device__ float c_LB[10]; //lower bound
__device__ float c_UB[10]; //upper bound
__device__ float *c_a;  //Tzaihe
__device__ float *c_aa;  //yingliK
__device__ float *c_aaa; //Tyingli
__device__ int c_aRow;
__device__ int c_aaaRow;
__device__ float c_Dysum[9];

float bestFitnessOfGen; //每一代的最优适应度
int bestIndexOfGen; //每一代的最优适应度位置
float aveFitnessOfGen[maxGeneration]; //每一代的平均最优适应度

float fval; //最终最优适应度
int G; //取得最终最优适应度的迭代次数
//BOOL elitism = TRUE; //是否精英选择

__global__ void mutationFcn(float *populationArray, curandState_t *states) {
//printf("mutationFcn\n");
int idx = threadIdx.x;
curandState_t s = states[idx];
curandState_t t = states[idx];
float ss = curand_uniform(&s);
int tt = curand(&t);

float scale = 0.5, shrink = 0.75;
scale -= scale * shrink * idx / maxGeneration;

//判断当前个体是否变异
if (ss < mutationRate){
for (int j = 0; j < chromosomeSize; j++) {
//判断当前染色体是否变异
if (tt % 2 != 0) {
float tmpChromosome;
do {
tmpChromosome = populationArray[idx * chromosomeSize + j] + scale * (c_UB[j] - c_LB[j]) * ss;
//判断是否越界
} while (tmpChromosome > c_UB[j] || tmpChromosome < c_LB[j]);
populationArray[idx * chromosomeSize + j] = tmpChromosome;
}
}
}
}