#include "includes.h"


/* Program Parameters */
#define MAXN 15000  /* Max value of N */
#define TILE_WIDTH 32  /* Width of each block */
int N;  /* Matrix size */

/* Matrices */
float overall;
char buffer[10000];
char *pbuff;
int *classIdArray = (int *)malloc(sizeof(int)*26);
int *groupIdArray = (int *)malloc(sizeof(int)*26);
int *roomIdArray =(int *) malloc(sizeof(int)*26);
int *roomSizeAsArray = (int *)malloc(sizeof(int)*5);
int *groupSizeAsArray = (int *)malloc(sizeof(int)*11);
int *timeSlotIdArray = (int *)malloc(sizeof(int)*26);
int *profIdArray = (int *)malloc(sizeof(int)*26);
int *clashes=(int *)calloc(26*26,sizeof(int));

int   *dclassIdArray,*dgroupIdArray,*droomIdArray,*droomSizeAsArray,*dgroupSizeAsArray,*dtimeSlotIdArray,*dprofIdArray,*dclashes;

/* junk */
#define randm() 4|2[uid]&3


/* returns a seed for srand based on the time */
__global__ void calculateFitness(int *classIds, int *roomIds, int *roomCapacities, int *groupIds, int *groupSizes, int *timeSlotIds, int *profIds, int *clashes)
{
int i = threadIdx.x;
int j = blockIdx.x ;

if(classIds[i]==classIds[j] && roomCapacities[classIds[i]]<groupSizes[groupIds[classIds[i]]])
clashes[i*j+i]++;

if (roomIds[classIds[j]] == roomIds[classIds[i]] && timeSlotIds[classIds[i]] == timeSlotIds[classIds[j]]
&& classIds[i] != classIds[j]) {
clashes[i*j+i]++;
}

if (profIds[classIds[j]] == profIds[classIds[j]] && timeSlotIds[classIds[i]] == timeSlotIds[classIds[j]]
&& classIds[i] != classIds[j]) {
clashes[i*j+i]++;
}
}