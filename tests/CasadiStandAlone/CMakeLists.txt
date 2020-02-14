add_executable(CasadiTest CasadiTest.cpp)
target_link_libraries(CasadiTest ${CASADI_LIBRARIES})
target_include_directories(CasadiTest PUBLIC ${CASADI_INCLUDE_DIR})
set_target_properties(CasadiTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)