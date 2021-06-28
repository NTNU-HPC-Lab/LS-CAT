// Gkyl ------------------------------------------------------------------------
//
// Macros for internal use in CUDA LuaJIT bindings
//    _______     ___
// + 6 @ |||| # P ||| +
//------------------------------------------------------------------------------

#ifndef GKYL_CUDA_MACROS_H
#define GKYL_CUDA_MACROS_H

// Macros to declare/define functions to get CUDA objects
#define GET_CUDA_OBJECT(type, value) type get_##value() { return value; }
#define DECL_GET_CUDA_OBJECT(type, value) type get_##value()

#endif // GKYL_CUDA_MACROS_H
