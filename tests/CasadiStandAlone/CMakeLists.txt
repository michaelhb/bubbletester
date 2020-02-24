add_executable(CasadiMaupertuis CasadiMaupertuis.cpp)
target_link_libraries(CasadiMaupertuis ${CASADI_LIBRARIES})
target_include_directories(CasadiMaupertuis PUBLIC ${CASADI_INCLUDE_DIR})
set_target_properties(CasadiMaupertuis PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(TestCasadiPotential TestCasadiPotential.cpp)
target_link_libraries(TestCasadiPotential ${CASADI_LIBRARIES})
target_include_directories(TestCasadiPotential PUBLIC ${CASADI_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/include)
set_target_properties(TestCasadiPotential PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)