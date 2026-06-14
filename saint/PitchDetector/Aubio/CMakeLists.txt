# aubio pitch-detector wrappers, for benchmarking and the TestApp only — never
# part of the production library (aubio is GPL-3.0; see aubio-integration.md at
# the repo root).
option(SAINT_WITH_AUBIO
  "Build the aubio pitch detectors for the benchmark and the TestApp" OFF)

if(NOT SAINT_WITH_AUBIO)
  return()
endif()

include(FetchContent)
# Pinned to a commit that ships the CMake build (the released 0.4.9 tag predates
# it). aubio's top-level CMakeLists also builds examples and tests we don't need,
# so SOURCE_SUBDIR points the build straight at src/, which defines the `aubio`
# target alone.
FetchContent_Declare(aubio
  GIT_REPOSITORY https://github.com/aubio/aubio.git
  GIT_TAG ad5cf975aed08cc4562dd008cf9f83b12b82ffb8
  SOURCE_SUBDIR src
)
FetchContent_MakeAvailable(aubio)

add_library(AubioPitchDetector STATIC
  AubioPitchDetector.cpp
  AubioPitchDetector.h
)

target_compile_options(AubioPitchDetector PRIVATE ${SAINT_ANNOYING_WARNINGS})

target_compile_definitions(AubioPitchDetector PUBLIC SAINT_WITH_AUBIO)

target_include_directories(AubioPitchDetector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# aubio's `aubio` target exposes no public include directories, so point at its
# umbrella header (src/aubio.h) ourselves. SYSTEM keeps its warnings out of ours.
target_include_directories(AubioPitchDetector SYSTEM PUBLIC ${aubio_SOURCE_DIR}/src)

target_link_libraries(AubioPitchDetector
  PUBLIC
    PitchDetector
    aubio
)
