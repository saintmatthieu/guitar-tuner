add_library(PitchDetector)

target_compile_options(PitchDetector PRIVATE ${SAINT_ANNOYING_WARNINGS})

target_sources(PitchDetector
  PUBLIC
    Cepstrum.cpp
    PitchDetectorImpl.cpp
    DummyFormantShifterLogger.cpp
)

target_link_libraries(PitchDetector
  PUBLIC
    PFFFT
    Utils
)

target_include_directories(PitchDetector
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_SOURCE_DIR}/Ring-Buffer
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(PitchDetector SYSTEM
  PUBLIC
    ${CMAKE_SOURCE_DIR}/pffft
)

add_executable(PitchDetectorImplTests
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
