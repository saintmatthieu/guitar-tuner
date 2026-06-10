add_subdirectory(PitchDetector)

if(NOT NO_GTEST)
  add_subdirectory(TestUtils)
endif()

add_subdirectory(Utils)

# Tuner apps: offline replay (ReplayApp) and real-time tuner (TestApp, Linux only, requires ALSA)
if(UNIX AND NOT APPLE)
  add_subdirectory(TestApp)
endif()