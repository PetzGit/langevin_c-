add_executable(cldiff_tests
    test_polyhedron.cpp
    test_rng.cpp
    test_sampler.cpp
)
target_link_libraries(cldiff_tests PRIVATE cldiff Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(cldiff_tests)
