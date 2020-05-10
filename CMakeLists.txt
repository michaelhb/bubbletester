cmake_minimum_required(VERSION 2.8.12)

project(libbubbler_example)
enable_language(CXX)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

find_package(libbubbler REQUIRED)
find_package(effpotential REQUIRED)
find_package(CASADI REQUIRED)
find_package(Eigen3 REQUIRED)

# List source files so tests can access them (surely there's a better way?)
set(SRC_FILES ${PROJECT_SOURCE_DIR}/src/SimpleBounce.cpp
              ${PROJECT_SOURCE_DIR}/src/FiniteTempPotential.cpp
              ${PROJECT_SOURCE_DIR}/src/GenericPotential.cpp
              ${PROJECT_SOURCE_DIR}/src/Rotation.cpp
              ${PROJECT_SOURCE_DIR}/src/CTFiniteTempModel.cpp
)

# Set up the tests
add_subdirectory(tests)

# Configure LGR frame generator
add_executable(LGRFrameGenerator ${PROJECT_SOURCE_DIR}/src/LGRFrameGenerator.cpp ${SRC_FILES})
target_link_libraries(LGRFrameGenerator effpotential ${LIBBUBBLER_LIBRARIES} ${CASADI_LIBRARIES})
target_include_directories(LGRFrameGenerator PUBLIC ${LIBBUBBLER_INCLUDE_DIRS} ${CASADI_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/include)
set_target_properties(LGRFrameGenerator PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)