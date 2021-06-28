#pragma once

#include <cudnn.h>
#include <curand.h>
#include <cublas_v2.h>

#include "config.h"

size_t const WORKSPACE_SIZE = size_t(128) * 1024 * 1024;

namespace dexe {

struct DEXE_API Handler {
  Handler();
  ~Handler();
  void init();

  static cudnnHandle_t &cudnn();
  static curandGenerator_t &curand();
  static cublasHandle_t &cublas();
  static char *workspace();
  static size_t workspace_size();
  static void set_workspace_size(size_t workspace_size);
  static void clear_workspace();

  static void set_device(int n);

  static Handler &get_handler();
  static void deinit();

  static float *one_float();
  static double *one_double();
  

  static void print_mem_info();

  static void sync();

  cudnnHandle_t h_cudnn;
  curandGenerator_t h_curand;
  cublasHandle_t h_cublas;
     
  char *s_workspace = nullptr;
  size_t workspace_size_ = WORKSPACE_SIZE;
    
  static Handler *s_handler;

  //pointers to cuda memory containing 1 (needed for some blas functions)
  float *one_float_ = 0;
  double *one_double_ = 0;
};

}
