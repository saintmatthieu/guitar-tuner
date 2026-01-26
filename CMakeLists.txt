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

target_compile_options(pf_conv_dispatcher PRIVATE -w)
target_compile_options(pf_conv_arch_none PRIVATE -w)
target_compile_options(pf_conv_arch_dflt PRIVATE -w)

if(TARGET pf_conv_arch_avx2)
  target_compile_options(pf_conv_arch_avx2 PRIVATE -w)
endif()

if(TARGET pf_conv_arch_avx)
  target_compile_options(pf_conv_arch_avx PRIVATE -w)
endif()

if(NOT NO_GTEST)
  target_compile_options(gmock_main PRIVATE -w)
  target_compile_options(gtest_main PRIVATE -w)
  target_compile_options(gtest PRIVATE -w)
  target_compile_options(gmock PRIVATE -w)
endif()

target_compile_options(FFTPACK_DOUBLE PRIVATE -w)
target_compile_options(PFFASTCONV PRIVATE -w)
target_compile_options(PFFFT PRIVATE -w)
target_compile_options(PFDSP PRIVATE -w)
target_compile_options(FFTPACK_FLOAT PRIVATE -w)
