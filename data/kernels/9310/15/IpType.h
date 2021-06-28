#pragma once

namespace ip
{

typedef signed char                 s8;
typedef signed short                s16;
typedef signed int                  s32;
typedef signed long long int        s64;
typedef unsigned char               u8;
typedef unsigned short              u16;
typedef unsigned int                u32;
typedef unsigned long long int      u64;
typedef float                       f32;
typedef double                      f64;


enum class PixelType
{
    U8 = 0,
    U16,
    U32,
    F32,
    F64,
    UNKNOWN,
};


} // namespace ip

