#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <ctime>
#include <algorithm>  // For time()
#include <cstdlib>
#include <chrono>
#include <unistd.h>

#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>


extern const int fSUMFLAG;
extern const int fKNAPSACKFLAG;

extern const int fAVGFLAG;
extern const int fMATCHFLAG;
extern const int fINVERSESUMFLAG;

extern const int fMAXIMIZE;
extern const int fMINIMIZE;

__device__ float getKnapsackFitness(float *,int,float*,float*,float,int,int);
__device__ float getAvgCount(float*,int,int,int);
__device__ float getSum(float*,int,int,int);
__device__ float getMatch(float*,float*,int,int,int);
__device__ void Crossover(float*,int,int,int,int,int,int);

class EABaseF
{
	public:
		virtual void initializePopulation()=0;
		virtual void getStatistics()=0;
		
};
/*=========================================BINARY GENOME=================================================*/

class FloatChromosome1D
{
	public:

		float *chromosome;
		int sizeofChromosome;
		float fitnessValue;
		FloatChromosome1D(int size){
			sizeofChromosome=size;
			chromosome = (float*) malloc(size*sizeof(float));
		}	
		void initializeChromosome(int size){
			sizeofChromosome=size;
			chromosome = (float*) malloc(size*sizeof(float));	
			fitnessValue=0.0;		
		}
		FloatChromosome1D(){

		}
};

class EAFloat: public EABaseF
{
	private:
		int id;
		float *chromosome;
		int *indices;
		int *cudaIndices; 
		
		int fitnessFlag;
		int minmaxflag; 
		int popSize;
		float mutationProbability=0.4;
		int *randomRange;
		FloatChromosome1D *population;
		float **Cudapopulation1D;
		float *Cudapopulation;
		int chromosomeSize;
		int populationSize;
		curandState* devStates;
		dim3 threads;
		dim3 blocks;

		//Statistics Variables
		double crossoverMutationTime=0.0;
		double fitnessCalculationTime=0.0;
		double sortingpopulationTime=0.0;
		double totalKernelTime=0.0;
		double totalMemoryTransferTime=0.0;
		double totalGpuTime=0.0;
		double totalCPUTime=0.0;

		


		
	public:

		void setFitnessFlag(int,int);		
		void shuffle(int);
		void evolve();
		void sortpop();
		float fitness(int);
		void init();
		EAFloat(int,int,float*);// (sizeofChromosomoe,sizeofpopulation,range[0,1])
		EAFloat(int,int,float**);// (sizeofChromosome,sizeofpopulation,population)
		void initializePopulation(){
			init();

		}
		void setParamKnapSack(float *kvalues,float *kweight,int chromosomeSize,float maxWeight);
		void setMatchParameter(float *kvalues,int chromosomeSize);

		void doCrossOver(int);
		void doMutation(int);
		void getStatistics(){
			printf("\n======================================================================================================\n");
			printf("\n===================================STATISTICS=========================================================\n");
			printf("\n======================================================================================================\n");
			printf("Total kernel Execution time : %f\n",totalKernelTime);
			printf("Total Memory tranfer time : %f\n", totalMemoryTransferTime);
			printf("Total Fitness calculation time : %f\n",fitnessCalculationTime);
		}
		int getChoromosomeSize();
		int getPopulationSize();
		void printpopulation();
		~EAFloat(){
			cudaFree(Cudapopulation);
		}

		

};
