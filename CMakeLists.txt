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

if(NOT NO_GTEST)
  add_subdirectory(googletest)
endif()

add_subdirectory(pffft)
add_subdirectory(saint)

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
