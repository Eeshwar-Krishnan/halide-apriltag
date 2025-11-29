#include <Halide.h>
#include <cstdio>
#include <memory>
#include <mutex>

extern "C" {
#include "apriltag/apriltag.h"
#include "apriltag/common/image_u8.h"
#include "apriltag/common/matd.h"
}

using Halide::Func;
using Halide::ImageParam;
using Halide::Param;
using Halide::Var;
using Halide::BoundaryConditions::repeat_edge;

namespace {

class DecodePipeline {
public:
    DecodePipeline()
        : input_(Halide::type_of<uint8_t>(), 2, "input"),
          H_(Halide::type_of<double>(), 2, "H"),
          samples_x_(Halide::type_of<float>(), 1, "samples_x"),
          samples_y_(Halide::type_of<float>(), 1, "samples_y") {}

    void compile_once() {
        std::call_once(init_flag_, [&]() { build(); });
    }

    void run(const Halide::Buffer<uint8_t> &input_buf,
             const Halide::Buffer<double> &H_buf,
             const Halide::Buffer<float> &sx_buf,
             const Halide::Buffer<float> &sy_buf,
             Halide::Buffer<uint8_t> &output_buf) {
        compile_once();

        input_.set(input_buf);
        H_.set(H_buf);
        samples_x_.set(sx_buf);
        samples_y_.set(sy_buf);

        pipeline_->realize(output_buf);
    }

private:
    void build() {
        Var x("x");

        // Input image with boundary conditions
        Func clamped = repeat_edge(input_, {{0, input_.width()}, {0, input_.height()}});

        // Homography matrix H (3x3)
        // H(0,0) H(0,1) H(0,2)
        // H(1,0) H(1,1) H(1,2)
        // H(2,0) H(2,1) H(2,2)
        
        // Sample coordinates (in tag space)
        Halide::Expr sx = samples_x_(x);
        Halide::Expr sy = samples_y_(x);

        // Project using Homography
        // z = H20*sx + H21*sy + H22
        // px = (H00*sx + H01*sy + H02) / z
        // py = (H10*sx + H11*sy + H12) / z
        
        Halide::Expr H00 = H_(0, 0), H01 = H_(1, 0), H02 = H_(2, 0);
        Halide::Expr H10 = H_(0, 1), H11 = H_(1, 1), H12 = H_(2, 1);
        Halide::Expr H20 = H_(0, 2), H21 = H_(1, 2), H22 = H_(2, 2);

        Halide::Expr z = H20 * sx + H21 * sy + H22;
        Halide::Expr px = (H00 * sx + H01 * sy + H02) / z;
        Halide::Expr py = (H10 * sx + H11 * sy + H12) / z;

        // Bilinear interpolation
        // We need to sample at (px, py)
        // Halide's linear_interp might be useful, or manual.
        // Manual:
        Halide::Expr x_f = Halide::floor(px - 0.5f);
        Halide::Expr y_f = Halide::floor(py - 0.5f);
        Halide::Expr x_i = Halide::cast<int>(x_f);
        Halide::Expr y_i = Halide::cast<int>(y_f);
        
        Halide::Expr u = px - 0.5f - x_f;
        Halide::Expr v = py - 0.5f - y_f;
        
        // Sample 4 neighbors
        Halide::Expr p00 = clamped(x_i, y_i);
        Halide::Expr p10 = clamped(x_i + 1, y_i);
        Halide::Expr p01 = clamped(x_i, y_i + 1);
        Halide::Expr p11 = clamped(x_i + 1, y_i + 1);
        
        Halide::Expr val = Halide::lerp(
            Halide::lerp(p00, p10, u),
            Halide::lerp(p01, p11, u),
            v
        );

        Func output("output");
        output(x) = Halide::cast<uint8_t>(val);

        // Schedule
        // Vectorize 32 to use wider SIMD if available.
        // Since this is 1D and likely small, parallelization might add overhead.
        output.compute_root().vectorize(x, 32);

        pipeline_ = std::make_unique<Halide::Pipeline>(output);
        Halide::Target target = Halide::get_host_target();
        pipeline_->compile_jit(target);
    }

    ImageParam input_;
    ImageParam H_;
    ImageParam samples_x_, samples_y_;
    std::unique_ptr<Halide::Pipeline> pipeline_;
    std::once_flag init_flag_;
};

DecodePipeline &get_decode_pipeline() {
    static DecodePipeline pipeline;
    return pipeline;
}

} // namespace

extern "C" void halide_decode_sample(
    uint8_t *input_buf, int w, int h, int s,
    double *H_buf,
    float *sx_buf, float *sy_buf, int num_samples,
    uint8_t *output_buf)
{
    DecodePipeline &pipeline = get_decode_pipeline();
    pipeline.compile_once();

    Halide::Buffer<uint8_t> input(input_buf, w, h);
    input.raw_buffer()->dim[0].stride = 1;
    input.raw_buffer()->dim[1].stride = s;

    Halide::Buffer<double> H(H_buf, 3, 3); // 3x3 matrix
    
    Halide::Buffer<float> sx(sx_buf, num_samples);
    Halide::Buffer<float> sy(sy_buf, num_samples);
    
    Halide::Buffer<uint8_t> output(output_buf, num_samples);

    pipeline.run(input, H, sx, sy, output);
}
