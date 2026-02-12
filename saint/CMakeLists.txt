add_subdirectory(PitchDetector)

if(NOT NO_GTEST)
  add_subdirectory(TestUtils)
endif()

add_subdirectory(Utils)

# Real-time tuner app (Linux only, requires ALSA)
if(UNIX AND NOT APPLE)
  find_library(ALSA_LIBRARY asound)
  if(ALSA_LIBRARY)
    add_subdirectory(TestApp)
  else()
    message(STATUS "ALSA not found, skipping TestApp build")
  endif()
endif()