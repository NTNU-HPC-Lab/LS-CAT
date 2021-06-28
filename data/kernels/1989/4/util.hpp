#ifndef UTIL_HPP
#define UTIL_HPP 

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>

#ifndef NDEBUG
#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
#else
#define gpuErrchk(ans)                                                         \
  {}
#endif

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

template<typename T>
void fill_array(T *arr, int n, T v) {
    for (int i = 0; i < n; ++i) {
        arr[i] = v;
    }
}


template<typename T>
void print_array(T *arr, int n) {
    std::cout << '[';
    for (int i = 0; i < n - 1; ++i) {
        std::cout << arr[i] << ", ";
    }
    if (n > 0) {
        std::cout << arr[n - 1];
    }
    std::cout << "]" << std::endl;
}

template<typename T>
void print_mat(T *arr, int m, int n) {
  for (int j = 0; j < m; ++j) {
    std::cout << '[';
    for (int i = 0; i < n - 1; ++i) {
        std::cout << arr[j * n + i] << ", ";
    }
    if (n > 0) {
        std::cout << arr[j * n + n - 1];
    }
    std::cout << "]" << std::endl;
  }
}


#endif /* UTIL_HPP */
