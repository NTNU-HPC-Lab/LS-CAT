#include "includes.h"
__global__ void groupSumKernel( float* input, int* firstOccurrences, int* otherOccurrences, int* otherOccurrencePositions, int dimensions, int numberIterations) {

int duplicateIndex = blockIdx.x;

int firstOccurrence = firstOccurrences[duplicateIndex];

int startWithinParameter = threadIdx.x * numberIterations;
int startFirstOccurrenceWithinBatch = firstOccurrence * dimensions;

int startFirstOccurrenceEntryIndex = startFirstOccurrenceWithinBatch + startWithinParameter;
int exclusiveEndFirstOccurrenceEntryIndex = min(startFirstOccurrenceEntryIndex + numberIterations, startFirstOccurrenceWithinBatch + dimensions);

int startOtherOccurrencePosition = otherOccurrencePositions[duplicateIndex];
int exclusiveEndOtherOccurrencePosition = otherOccurrencePositions[duplicateIndex + 1];

for(int otherOccurrencePosition = startOtherOccurrencePosition; otherOccurrencePosition < exclusiveEndOtherOccurrencePosition; otherOccurrencePosition++) {

int otherOccurrence = otherOccurrences[otherOccurrencePosition];

int firstOccurrenceEntryIndex = startFirstOccurrenceEntryIndex;
int otherOccurrenceEntryIndex = otherOccurrence * dimensions + startWithinParameter;

while(firstOccurrenceEntryIndex < exclusiveEndFirstOccurrenceEntryIndex) {

input[firstOccurrenceEntryIndex] += input[otherOccurrenceEntryIndex];
input[otherOccurrenceEntryIndex] = nanf("NaN");

firstOccurrenceEntryIndex++;
otherOccurrenceEntryIndex++;
}

}

}