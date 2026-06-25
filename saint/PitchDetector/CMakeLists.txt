add_library(PitchDetector)

target_compile_options(PitchDetector PRIVATE ${SAINT_ANNOYING_WARNINGS})

target_sources(PitchDetector
  PUBLIC
    Cepstrum.cpp
    AutocorrEstimateDisambiguator.cpp
    AutocorrPitchDetector.cpp
    PitchDetectorFactory.cpp
    PitchDetectorImpl.cpp
    PitchDetectorUtils.cpp
    PitchDetectorMedianFilter.cpp
    Preprocessor.cpp
    DummyPitchDetectorLogger.cpp
    FrequencyDomainTransformer.cpp
    OnsetDetector.cpp
    PitchDetectionHolder.cpp
    PitchDetectionSmoother.cpp
    Upsampler.cpp
    Recording/IssueReportingPitchDetector.cpp
    Recording/PitchDetectorRecording.cpp
    Recording/RecordingPitchDetector.cpp
)

target_link_libraries(PitchDetector
  PUBLIC
    PFFFT
    Utils
)

target_include_directories(PitchDetector
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(PitchDetector SYSTEM
  PUBLIC
    ${GUITAR_TUNER_ROOT}/pffft
)

add_subdirectory(Pesto)
add_subdirectory(Aubio)
add_subdirectory(Pyin)

# Always added: Test/ also defines the RecordingReplay library (the WAV reader +
# ReplayPitchDetector), which the ReplayApp needs even in NO_GTEST builds. The gtest
# executables inside are guarded by NOT NO_GTEST.
add_subdirectory(Test)
