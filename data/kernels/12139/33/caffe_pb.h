#pragma once

#ifndef CAFFE_EXPORTS
  #if defined libcaffe_EXPORTS && defined _MSC_VER
    #define CAFFE_EXPORTS __declspec(dllexport)
  #else
    #define CAFFE_EXPORTS
  #endif
#endif

#if defined proto_EXPORTS && defined _MSC_VER
#define PROTO_EXPORTS __declspec(dllexport)
#else
#define PROTO_EXPORTS
#endif

#ifdef STRICT // Boost 1.60+ on windows defines this
  #undef STRICT
#endif
#include "caffe/proto/caffe.pb.h"