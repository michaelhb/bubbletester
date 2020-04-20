cmake_minimum_required(VERSION 2.8.12)

project(libbubbler_example)
enable_language(CXX)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

find_package(libbubbler REQUIRED)
find_package(effpotential REQUIRED)
find_package(CASADI REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CXSC REQUIRED)

# List source files so tests can access them (surely there's a better way?)
set(SRC_FILES ${PROJECT_SOURCE_DIR}/src/SimpleBounce.cpp
              ${PROJECT_SOURCE_DIR}/src/FiniteTempPotential.cpp
              ${PROJECT_SOURCE_DIR}/src/GenericPotential.cpp
              ${PROJECT_SOURCE_DIR}/src/Rotation.cpp
              ${PROJECT_SOURCE_DIR}/src/CTFiniteTempModel.cpp
)

# Set up the tests
add_subdirectory(tests)

# TEMP cxsc test
add_executable(csxc_ex ${PROJECT_SOURCE_DIR}/src/example.cpp)
target_link_libraries(csxc_ex ${CSXC_LIBRARIES})
target_include_directories(csxc_ex PUBLIC ${CXSC_INCLUDE_DIR})
set_target_properties(csxc_ex PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

