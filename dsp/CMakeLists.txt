cmake_minimum_required(VERSION 3.18)
project(voiceanon_dsp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Portable real-time DSP core (also compiled into the Android app via the NDK).
add_library(voiceanon_dsp STATIC
  src/fft.cpp
  src/biquad.cpp
  src/noise_suppressor.cpp
  src/pitch_tracker.cpp
  src/psola.cpp
  src/limiter.cpp
  src/engine.cpp
  src/presets.cpp
  src/analysis.cpp)
target_include_directories(voiceanon_dsp PUBLIC include)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(voiceanon_dsp PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

option(VOICEANON_BUILD_HOST_TOOLS "Build host tests and tools" ON)
if(VOICEANON_BUILD_HOST_TOOLS AND NOT ANDROID)
  enable_testing()
  add_library(voiceanon_testing STATIC testing/synth.cpp testing/wav.cpp)
  target_include_directories(voiceanon_testing PUBLIC testing)
  target_link_libraries(voiceanon_testing PUBLIC voiceanon_dsp)

  add_executable(voiceanon_tests tests/test_main.cpp tests/test_components.cpp tests/test_engine.cpp)
  target_link_libraries(voiceanon_tests PRIVATE voiceanon_testing)
  add_test(NAME voiceanon_tests COMMAND voiceanon_tests)

  add_executable(voiceanon_bench tools/bench.cpp)
  target_link_libraries(voiceanon_bench PRIVATE voiceanon_testing)

  add_executable(voiceanon_eval tools/eval.cpp)
  target_link_libraries(voiceanon_eval PRIVATE voiceanon_testing)
endif()
