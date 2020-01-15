cmake_minimum_required(VERSION 2.8.12)

project(libbubbler_example)
enable_language(CXX)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

find_package(libbubbler REQUIRED)
find_package(effpotential REQUIRED)
find_package(CASADI REQUIRED)

# List source files so tests can access them (surely there's a better way?)
set(SRC_FILES ${PROJECT_SOURCE_DIR}/src/SimpleBounce.cpp
              ${PROJECT_SOURCE_DIR}/src/FiniteTempPotential.cpp
              ${PROJECT_SOURCE_DIR}/src/GenericPotential.cpp
)

# Set up the tests
add_subdirectory(tests)

