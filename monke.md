How to Run

CPU Benchmark
cd build
./apriltag_timing --compare-halide ../EleFrontCam_input_2025-04-16T075112283_P-12-NEWTON.jpg --runs 20

GPU Benchmark (OpenCL)

cd build
HL_JIT_TARGET=host-opencl ./apriltag_timing --compare-halide ../EleFrontCam_input_2025-04-16T075112283_P-12-NEWTON.jpg --runs 20