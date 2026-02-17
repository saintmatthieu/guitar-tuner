add_executable(PitchDetectorImplTests
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
