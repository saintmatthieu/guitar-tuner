add_library(Utils)

target_compile_options(Utils PRIVATE ${SAINT_ANNOYING_WARNINGS})

target_sources(Utils
  PUBLIC
    CommonTypes.cpp
    Utils.cpp
)

target_include_directories(Utils
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(NOT NO_GTEST)
  add_executable(UtilsTests)

  target_sources(UtilsTests
    PRIVATE
      UtilsTests.cpp
  )

  target_link_libraries(UtilsTests
    PRIVATE
      Utils
      gtest_main
  )
endif()
