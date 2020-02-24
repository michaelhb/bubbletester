add_executable(CTAnalyticPotential TestCTAnalyticPotential.cpp ${SRC_FILES})
target_link_libraries(CTAnalyticPotential effpotential ${LIBBUBBLER_LIBRARIES})
target_include_directories(CTAnalyticPotential PUBLIC ${LIBBUBBLER_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
set_target_properties(CTAnalyticPotential PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)