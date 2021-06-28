#include "includes.h"
__global__ void setGroupsPointersDead(multipassConfig_t* mbk, unsigned numBuckets)
{
int index = TID;
if(index < numBuckets)
{
mbk->isNextDeads[index] = 1;
}

}