add_subdirectory(PitchDetector)

if(NOT NO_GTEST)
  add_subdirectory(TestUtils)
endif()

add_subdirectory(Utils)

# Tuner apps: offline replay (ReplayApp) and real-time tuner (TestApp; ALSA on Linux, CoreAudio on macOS)
if(UNIX)
  add_subdirectory(TestApp)
endif()