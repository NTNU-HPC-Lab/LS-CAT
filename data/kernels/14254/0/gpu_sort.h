/*
 * gpu_sort.h
 *
 *  Created on: Mar 7, 2015
 *      Author: hugo.braun
 */

#ifndef GPU_SORT_H_
#define GPU_SORT_H_

void gpu_quicksort(int * data, int n) ;
void gpu_quicksort_benchmark(int * data, int n, int ntests);

#endif /* GPU_SORT_H_ */
