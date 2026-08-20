# Reading recordings back (the WAV reader and the ReplayPitchDetector decorator) is debug/test
# tooling, kept out of the production PitchDetector library. It is shared by the offline ReplayApp
# and the recording tests, so it is a library and is built unconditionally (the ReplayApp is built
# even in NO_GTEST configurations).
add_library(RecordingReplay)

target_compile_options(RecordingReplay PRIVATE ${SAINT_ANNOYING_WARNINGS})

target_sources(RecordingReplay
  PUBLIC
    RecordingFileReader.cpp
    ReplayPitchDetector.cpp
)

target_link_libraries(RecordingReplay
  PUBLIC
    PitchDetector
)

target_include_directories(RecordingReplay
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(NOT NO_GTEST)
  add_executable(PitchDetectorImplTests
    BenchmarkAlgorithms.cpp
    PitchDetectorImplTests.cpp
  )

  target_compile_options(PitchDetectorImplTests PRIVATE ${SAINT_ANNOYING_WARNINGS})

  target_link_libraries(PitchDetectorImplTests
    PRIVATE
      PitchDetector
      TestUtils
      gtest_main
      gmock
  )

  if(SAINT_WITH_PESTO)
    target_link_libraries(PitchDetectorImplTests PRIVATE PestoPitchDetector)
    set_target_properties(PitchDetectorImplTests PROPERTIES
      BUILD_RPATH ${SAINT_ONNXRUNTIME_LIB_DIR})
  endif()

  if(SAINT_WITH_AUBIO)
    target_link_libraries(PitchDetectorImplTests PRIVATE AubioPitchDetector)
  endif()

  if(SAINT_WITH_PYIN)
    target_link_libraries(PitchDetectorImplTests PRIVATE PyinPitchDetector)
  endif()

  add_executable(OnsetDetectorCalibrationTests
    OnsetDetectorCalibrationTests.cpp
  )

  target_compile_options(OnsetDetectorCalibrationTests PRIVATE ${SAINT_ANNOYING_WARNINGS})

  target_link_libraries(OnsetDetectorCalibrationTests
    PRIVATE
      PitchDetector
      TestUtils
      gtest_main
      gmock
  )

  add_executable(PreprocessorTests
    PreprocessorTests.cpp
  )

  target_compile_options(PreprocessorTests PRIVATE ${SAINT_ANNOYING_WARNINGS})

  target_link_libraries(PreprocessorTests
    PRIVATE
      PitchDetector
      TestUtils
      gtest_main
      gmock
  )

  add_executable(RecordingPitchDetectorTests
    RecordingPitchDetectorTests.cpp
  )

  target_compile_options(RecordingPitchDetectorTests PRIVATE ${SAINT_ANNOYING_WARNINGS})

  target_link_libraries(RecordingPitchDetectorTests
    PRIVATE
      PitchDetector
      RecordingReplay
      TestUtils
      gtest_main
      gmock
  )

  add_executable(UpsamplerTests
    UpsamplerTests.cpp
  )

  target_compile_options(UpsamplerTests PRIVATE ${SAINT_ANNOYING_WARNINGS})

  target_link_libraries(UpsamplerTests
    PRIVATE
      PitchDetector
      TestUtils
      gtest_main
      gmock
  )
endif()
