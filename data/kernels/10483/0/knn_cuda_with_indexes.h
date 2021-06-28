#ifndef KNN_CUDA_WITH_INDEXES_H
#define KNN_CUDA_WITH_INDEXES_H

namespace knn_cuda_with_indexes {

    /**
      * Prints the error message return during the memory allocation.
      *
      * @param error        error value return by the memory allocation function
      * @param memorySize   size of memory tried to be allocated
      */
    void printErrorMessage(cudaError_t error, int memorySize);

    /**
      * K nearest neighbor algorithm
      * - Initialize CUDA
      * - Allocate device memory
      * - Copy point sets (reference and query points) from host to device memory
      * - Compute the distances + indexes to the k nearest neighbors for each query point
      * - Copy distances from device to host memory
      *
      * @param ref_host      reference points ; pointer to linear matrix
      * @param ref_width     number of reference points ; width of the matrix
      * @param query_host    query points ; pointer to linear matrix
      * @param query_width   number of query points ; width of the matrix
      * @param height        dimension of points ; height of the matrices
      * @param k             number of neighbor to consider
      * @param dist_host     distances to k nearest neighbors ; pointer to linear matrix
      * @param ind_host      indexes of the k nearest neighbors ; pointer to linear matrix
      *
      */
    void knn(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host, int* ind_host);

    /**
     * Computes the Euclidean distances between points
     *
     * @param ref_host      reference points ; pointer to linear matrix
     * @param ref_width     number of reference points ; width of the matrix
     * @param query_host    query points ; pointer to linear matrix
     * @param query_width   number of query points ; width of the matrix
     * @param height        dimension of points ; height of the matrices
     * @param dist_host     distances of each element in query (col) to each element in ref (row)
     * @param inf           whether the norm is finite or not
     * @param norm          value of the norm. If inf is true, the sign of this indicates whether we are using positive or negative infinity
     */
    void computeDistances(float* ref_host, int ref_width, float* query_host, int query_width, int height, float* dist_host, bool inf, float norm);
}

#endif
