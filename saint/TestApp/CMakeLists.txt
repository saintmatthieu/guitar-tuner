# Offline replay of issue recordings - no audio input needed
add_executable(ReplayApp)

target_compile_options(ReplayApp PRIVATE ${SAINT_ANNOYING_WARNINGS})

target_sources(ReplayApp
  PRIVATE
    replayMain.cpp
    TunerDisplay.cpp
    TunerDisplay.h
)

target_link_libraries(ReplayApp
  PRIVATE
    PitchDetector
)

target_include_directories(ReplayApp
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Real-time tuner app (requires ALSA)
find_library(ALSA_LIBRARY asound)
if(ALSA_LIBRARY)
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
else()
  message(STATUS "ALSA not found, skipping TestApp build")
endif()
