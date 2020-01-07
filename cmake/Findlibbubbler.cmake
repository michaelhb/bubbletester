# - Try to find libbubbler
# Once done, this will define
#
#  LIBBUBBLER_FOUND
#  LIBBUBBLER_INCLUDE_DIRS
#  LIBBUBBLER_LIBRARIES
#  LIBBUBBLER_DEFINITIONS

# Find the dirs where BubbleProfiler headers / lib installed
find_path(LIBBUBBLER_INCLUDE_DIR NAMES BubbleProfiler/potential.hpp)
find_library(LIBBUBBLER_LIBRARY NAMES libbubbler.a)

# Find all the dependencies required to use libbubbler
find_package(Boost 1.53.0 REQUIRED
  COMPONENTS
  filesystem
  program_options
  system
  )
find_package(CLN 1.2.2 REQUIRED)
find_package(Eigen3 3.1 REQUIRED)
find_package(GiNaC REQUIRED)
find_package(GSL 1.15 REQUIRED)
find_package(NLopt REQUIRED)
find_package(Threads REQUIRED)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBBUBBLER DEFAULT_MSG LIBBUBBLER_INCLUDE_DIR LIBBUBBLER_LIBRARY)

if (LIBBUBBLER_FOUND)
    set(LIBBUBBLER_LIBRARIES 
      ${LIBBUBBLER_LIBRARY} 
      ${NLopt_LIBRARIES}
      ${GINAC_LIBRARIES}
      ${CLN_LIBRARIES}
      ${Boost_LIBRARIES}
      ${GSL_LIBRARIES}
      ${CMAKE_DL_LIBS}
    )
    set(LIBBUBBLER_INCLUDE_DIRS 
      ${LIBBUBBLER_INCLUDE_DIR} 
      ${Boost_INCLUDE_DIRS}
      ${CLN_INCLUDE_DIRS}
      ${EIGEN3_INCLUDE_DIR}
      ${GINAC_INCLUDE_DIRS}
      ${GSL_INCLUDE_DIRS}
      ${NLopt_INCLUDE_DIRS}
    )
    set(LIBBUBBLER_DEFINITIONS )
endif()

