# PESTO (ONNX) pitch-detector wrapper, for benchmarking and the TestApp only —
# never part of the production library (PESTO is LGPL-3.0; see
# pesto-integration.md at the repo root).
option(SAINT_WITH_PESTO
  "Build the PESTO (ONNX) pitch detector for the benchmark and the TestApp" OFF)

if(NOT SAINT_WITH_PESTO)
  return()
endif()

include(FetchContent)
set(SAINT_ONNXRUNTIME_VERSION 1.26.0)
if(APPLE)
  set(saint_ort_platform osx-arm64)
elseif(UNIX)
  set(saint_ort_platform linux-x64)
else()
  message(FATAL_ERROR
    "SAINT_WITH_PESTO: no prebuilt onnxruntime URL configured for this platform")
endif()
FetchContent_Declare(onnxruntime_prebuilt
  URL https://github.com/microsoft/onnxruntime/releases/download/v${SAINT_ONNXRUNTIME_VERSION}/onnxruntime-${saint_ort_platform}-${SAINT_ONNXRUNTIME_VERSION}.tgz
)
FetchContent_MakeAvailable(onnxruntime_prebuilt)

find_library(SAINT_ONNXRUNTIME_LIB onnxruntime
  PATHS ${onnxruntime_prebuilt_SOURCE_DIR}/lib NO_DEFAULT_PATH REQUIRED)
add_library(onnxruntime_prebuilt_lib SHARED IMPORTED)
set_target_properties(onnxruntime_prebuilt_lib PROPERTIES
  IMPORTED_LOCATION ${SAINT_ONNXRUNTIME_LIB}
  INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_prebuilt_SOURCE_DIR}/include
)
# Consumers (executables) need this on their BUILD_RPATH to find the .so at runtime.
set(SAINT_ONNXRUNTIME_LIB_DIR ${onnxruntime_prebuilt_SOURCE_DIR}/lib CACHE INTERNAL "")

add_library(PestoPitchDetector STATIC
  PestoPitchDetector.cpp
  PestoPitchDetector.h
)

target_compile_options(PestoPitchDetector PRIVATE ${SAINT_ANNOYING_WARNINGS})

target_compile_definitions(PestoPitchDetector PUBLIC
  SAINT_WITH_PESTO
  SAINT_PESTO_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models"
)

target_include_directories(PestoPitchDetector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(PestoPitchDetector
  PUBLIC
    PitchDetector
    onnxruntime_prebuilt_lib
)
