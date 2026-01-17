cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

project(guitar-tuner VERSION 0.0.1)

set(PFFFT_DISABLE_LINK_WITH_M ON CACHE BOOL "" FORCE)

add_subdirectory(googletest)
add_subdirectory(pffft)
add_subdirectory(saint)

target_compile_options(pf_conv_dispatcher PRIVATE -w)
target_compile_options(pf_conv_arch_none PRIVATE -w)
target_compile_options(pf_conv_arch_dflt PRIVATE -w)
target_compile_options(pf_conv_arch_avx2 PRIVATE -w)
target_compile_options(pf_conv_arch_avx PRIVATE -w)
target_compile_options(gmock_main PRIVATE -w)
target_compile_options(FFTPACK_DOUBLE PRIVATE -w)
target_compile_options(PFFASTCONV PRIVATE -w)
target_compile_options(PFFFT PRIVATE -w)
target_compile_options(PFDSP PRIVATE -w)
target_compile_options(gtest_main PRIVATE -w)
target_compile_options(gtest PRIVATE -w)
target_compile_options(gmock PRIVATE -w)
target_compile_options(FFTPACK_FLOAT PRIVATE -w)
