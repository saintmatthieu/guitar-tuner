# Detect the platform audio backend up front (ALSA on Linux, CoreAudio on
# macOS); it gates the real-time TestApp and the ReplayApp's audio playback.
if(APPLE)
  set(SAINT_AUDIO_SOURCES CoreAudioInput.cpp CoreAudioInput.h)
  find_library(AUDIOTOOLBOX_FRAMEWORK AudioToolbox)
  set(SAINT_AUDIO_LIBRARIES ${AUDIOTOOLBOX_FRAMEWORK})
  set(SAINT_BUILD_TESTAPP TRUE)
else()
  find_library(ALSA_LIBRARY asound)
  if(ALSA_LIBRARY)
    set(SAINT_AUDIO_SOURCES AlsaAudioInput.cpp AlsaAudioInput.h)
    set(SAINT_AUDIO_LIBRARIES asound)
    set(SAINT_BUILD_TESTAPP TRUE)
  else()
    set(SAINT_BUILD_TESTAPP FALSE)
    message(STATUS "ALSA not found, skipping TestApp build")
  endif()
endif()

# Offline replay of issue recordings - no audio *input* needed, but it can play
# the recording back through the speakers in real-time mode (Linux/ALSA only).
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
    RecordingReplay
)

target_include_directories(ReplayApp
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Real-time audio playback for the ReplayApp (Linux/ALSA only for now).
if(NOT APPLE AND ALSA_LIBRARY)
  target_sources(ReplayApp PRIVATE AlsaAudioOutput.cpp AlsaAudioOutput.h)
  target_link_libraries(ReplayApp PRIVATE asound)
  target_compile_definitions(ReplayApp PRIVATE SAINT_REPLAY_WITH_ALSA)
endif()

if(SAINT_BUILD_TESTAPP)
  add_executable(TestApp)

  target_compile_options(TestApp PRIVATE ${SAINT_ANNOYING_WARNINGS})

  target_sources(TestApp
    PRIVATE
      main.cpp
      AudioInput.h
      ${SAINT_AUDIO_SOURCES}
      TunerDisplay.cpp
      TunerDisplay.h
  )

  target_link_libraries(TestApp
    PRIVATE
      PitchDetector
      ${SAINT_AUDIO_LIBRARIES}
  )

  target_include_directories(TestApp
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )

  if(SAINT_WITH_PESTO)
    target_link_libraries(TestApp PRIVATE PestoPitchDetector)
    set_target_properties(TestApp PROPERTIES
      BUILD_RPATH ${SAINT_ONNXRUNTIME_LIB_DIR})
  endif()

  if(SAINT_WITH_AUBIO)
    target_link_libraries(TestApp PRIVATE AubioPitchDetector)
  endif()
endif()
