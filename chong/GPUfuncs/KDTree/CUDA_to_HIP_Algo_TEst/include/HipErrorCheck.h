#ifndef HIP_ERROR_CHECK_H
#define HIP_ERROR_CHECK_H

#include <hip/hip_runtime.h>  // for hipError_t, hipSuccess, hipGetErrorString
#include <iostream>
#include <cstdlib>            // for std::exit

#define HIP_CHECK(err)                                                      \
  do {                                                                       \
    hipError_t _e = (err);                                                   \
    if (_e != hipSuccess) {                                                  \
      std::cerr                                                             \
        << "HIP error at " << __FILE__ << ":" << __LINE__                   \
        << " → " << hipGetErrorString(_e) << std::endl;                     \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

#endif // HIP_ERROR_CHECK_H
