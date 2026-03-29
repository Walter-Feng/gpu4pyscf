set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

add_library(ovlp SHARED
				overlap.cu
				dipole.cu
)

set_target_properties(ovlp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
        CUDA_SEPARABLE_COMPILATION ON
)
