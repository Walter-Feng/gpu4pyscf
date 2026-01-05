set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -DLIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS")# -maxrregcount=128")

add_library(aft SHARED
        drivers.cu
				screen.cu
				# eval_xc.cu
				# eval_xc_grad.cu
)

set_target_properties(aft PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
        CUDA_SEPARABLE_COMPILATION ON
)
