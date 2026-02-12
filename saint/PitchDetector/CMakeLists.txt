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
    DummyPitchDetectorLogger.cpp
    FrequencyDomainTransformer.cpp
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

add_subdirectory(Test)