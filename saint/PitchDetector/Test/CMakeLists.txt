add_executable(PitchDetectorImplTests
  BenchmarkAlgorithms.cpp
  PitchDetectorImplTests.cpp
  PitchDetectorImplTestWrapper.cpp
)

target_compile_options(PitchDetectorImplTests PRIVATE ${SAINT_ANNOYING_WARNINGS})

target_link_libraries(PitchDetectorImplTests
  PRIVATE
    PitchDetector
    TestUtils
    gtest_main
    gmock
)

add_executable(AutocorrPeakCensusTests
  AutocorrPeakCensusTests.cpp
)

target_compile_options(AutocorrPeakCensusTests PRIVATE ${SAINT_ANNOYING_WARNINGS})

target_link_libraries(AutocorrPeakCensusTests
  PRIVATE
    PitchDetector
    TestUtils
    gtest_main
    gmock
)

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

