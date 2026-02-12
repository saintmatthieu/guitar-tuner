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
