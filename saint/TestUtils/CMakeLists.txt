add_library(TestUtils)

target_compile_options(TestUtils PRIVATE ${SAINT_ANNOYING_WARNINGS})

# fetch https://github.com/libsndfile/libsndfile.git
include(FetchContent)
FetchContent_Declare(
  sndfile
  GIT_REPOSITORY https://github.com/libsndfile/libsndfile.git
  GIT_TAG        1.2.2
)
FetchContent_MakeAvailable(sndfile)

target_sources(TestUtils
  PUBLIC
    testUtils.cpp
    FormantShifterLogger.cpp
)

target_include_directories(TestUtils
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(TestUtils
  PUBLIC
    sndfile
    PitchDetector
)
