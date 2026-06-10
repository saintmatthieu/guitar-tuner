cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
set(GUITAR_TUNER_ROOT ${CMAKE_CURRENT_LIST_DIR})

project(guitar-tuner VERSION 0.0.1)

option(NO_GTEST "Disable building/including GoogleTest (gtest/gmock)" OFF)

if(WIN32)
  set(PFFFT_DISABLE_LINK_WITH_M ON CACHE BOOL "" FORCE)
else()
  set(PFFFT_DISABLE_LINK_WITH_M OFF CACHE BOOL "" FORCE)
endif()

set(PFFFT_USE_BENCH_GREEN  OFF CACHE BOOL "" FORCE)
set(PFFFT_USE_BENCH_KISS   OFF CACHE BOOL "" FORCE)
set(PFFFT_USE_BENCH_POCKET OFF CACHE BOOL "" FORCE)
set(PFFFT_USE_FFTPACK      OFF CACHE BOOL "" FORCE)

if(NOT NO_GTEST)
  add_subdirectory(googletest)
endif()

add_subdirectory(pffft)
add_subdirectory(saint)

set(_pffft_unwanted_targets
  test_fft_factors
  test_pffft_cpp
  test_pffft_cpp_11
  test_fftpack_float
  test_fftpack_double
  test_pffft_float
  test_pffft_double
  test_pffastconv
  bench_pffft_float
  bench_pffft_double
  bench_pf_mixer_float
  bench_pf_conv_float
  example_cpp11_real_dbl_fwd
  example_cpp11_cplx_dbl_fwd
  example_c_cplx_dbl_fwd
  example_cpp98_real_flt_fwd
  example_cpp98_cplx_flt_fwd
  example_c_real_flt_fwd
)

foreach(tgt IN LISTS _pffft_unwanted_targets)
  if(TARGET ${tgt})
    set_target_properties(${tgt} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()
endforeach()

set(_silence_targets
  pf_conv_dispatcher
  pf_conv_arch_none
  pf_conv_arch_dflt
  pf_conv_arch_avx2
  pf_conv_arch_avx
  FFTPACK_DOUBLE
  PFFASTCONV
  PFFFT
  PFDSP
  FFTPACK_FLOAT
)

foreach(tgt IN LISTS _silence_targets)
  if(TARGET ${tgt})
    target_compile_options(${tgt} PRIVATE -w)
  endif()
endforeach()

if(NOT NO_GTEST)
  set(_gtest_targets
    gmock_main
    gtest_main
    gtest
    gmock
  )

  foreach(tgt IN LISTS _gtest_targets)
    if(TARGET ${tgt})
      target_compile_options(${tgt} PRIVATE -w)
    endif()
  endforeach()
endif()
