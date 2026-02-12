add_executable(TestApp)

target_compile_options(TestApp PRIVATE ${SAINT_ANNOYING_WARNINGS})

target_sources(TestApp
  PRIVATE
    main.cpp
    AlsaAudioInput.cpp
    AlsaAudioInput.h
    TunerDisplay.cpp
    TunerDisplay.h
)

target_link_libraries(TestApp
  PRIVATE
    PitchDetector
    asound
)

target_include_directories(TestApp
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
