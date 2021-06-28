#pragma once

__global__ void BlockPrefix(int *a, int k, int n);
__global__ void Compute(int *a, int k, int n);
/**
 * Compute the prefix sum of a
 * Array a is in device
 * 
 */
void PrefixSum(int *a, int n);
