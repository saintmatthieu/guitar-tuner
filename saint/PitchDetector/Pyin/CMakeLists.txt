# pYIN pitch-detector wrapper, for benchmarking and the TestApp only — never part
# of the production library (pYIN is GPL-2.0-or-later; see pyin-integration.md at
# the repo root). The upstream sources are fetched at build time and are not
# committed to this repository.
option(SAINT_WITH_PYIN
  "Build the pYIN pitch detector for the benchmark and the TestApp" OFF)

if(NOT SAINT_WITH_PYIN)
  return()
endif()

include(FetchContent)
# The pYIN reference (Mauch & Dixon) is a Vamp plugin with no CMake build, so we
# only populate it and compile the algorithm core ourselves — no Vamp SDK, no note
# segmentation. Pinned to release v1.2, whose SparseHMM already implements the
# fixed-lag online Viterbi decoding our streaming wrapper relies on.
FetchContent_Declare(pyin
  GIT_REPOSITORY https://github.com/c4dm/pyin.git
  GIT_TAG v1.2
)
FetchContent_GetProperties(pyin)
if(NOT pyin_POPULATED)
  FetchContent_Populate(pyin)
endif()

# The pYIN algorithm core + our dependency shims. Built as a separate library so
# the upstream GPL sources (and our FFT shim) compile without the project's strict
# warning set, which the wrapper below keeps. The shims/ directory provides
# vamp-sdk/FFT.h and an empty boost/math/distributions.hpp (see those files); it
# is marked SYSTEM so upstream warnings stay out of our build.
add_library(pyin_core STATIC
  shims/vamp-sdk/FFT.cpp
  ${pyin_SOURCE_DIR}/Yin.cpp
  ${pyin_SOURCE_DIR}/YinUtil.cpp
  ${pyin_SOURCE_DIR}/MonoPitchHMM.cpp
  ${pyin_SOURCE_DIR}/SparseHMM.cpp
)
target_include_directories(pyin_core SYSTEM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/shims
  ${pyin_SOURCE_DIR}
)

add_library(PyinPitchDetector STATIC
  PyinPitchDetector.cpp
  PyinPitchDetector.h
)

target_compile_options(PyinPitchDetector PRIVATE ${SAINT_ANNOYING_WARNINGS})

target_compile_definitions(PyinPitchDetector PUBLIC SAINT_WITH_PYIN)

target_include_directories(PyinPitchDetector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(PyinPitchDetector
  PUBLIC
    PitchDetector
  PRIVATE
    pyin_core
)
