#include <Halide.h>
#include <cstdio>
#include <memory>
#include <mutex>

extern "C" {
#include "apriltag/apriltag.h"
#include "apriltag/common/image_u8.h"
}

using Halide::Func;
using Halide::ImageParam;
using Halide::Param;
using Halide::Var;
using Halide::BoundaryConditions::repeat_edge;

namespace {

class GradientPipeline {
public:
    GradientPipeline()
        : input_(Halide::type_of<uint8_t>(), 2, "input") {}

    void compile_once() {
        std::call_once(init_flag_, [&]() { build(); });
    }

    void run(const Halide::Buffer<uint8_t> &input_buf,
             Halide::Buffer<int16_t> &gx_buf,
             Halide::Buffer<int16_t> &gy_buf,
             Halide::Buffer<float> &mag_buf) {
        compile_once();

        input_.set(input_buf);

        // Realize into the output buffers
        // We need a way to realize multiple outputs. Halide supports Realization wrapping multiple buffers.
        Halide::Realization outputs({gx_buf, gy_buf, mag_buf});
        pipeline_->realize(outputs);
    }

private:
    void build() {
        Var x("x"), y("y");

        // Clamp input
        Func clamped = repeat_edge(input_, {{0, input_.width()}, {0, input_.height()}});

        // Cast to 16-bit for arithmetic
        Func input_16("input_16");
        input_16(x, y) = Halide::cast<int16_t>(clamped(x, y));

        // Compute gradients (central difference)
        // gx = I(x+1, y) - I(x-1, y)
        // gy = I(x, y+1) - I(x, y-1)
        Func gx("gx"), gy("gy");
        gx(x, y) = input_16(x + 1, y) - input_16(x - 1, y);
        gy(x, y) = input_16(x, y + 1) - input_16(x, y - 1);

        // Compute magnitude
        Func gradients("gradients");
        Halide::Expr gx_val = gx(x, y);
        Halide::Expr gy_val = gy(x, y);
        Halide::Expr sum_sq = Halide::cast<float>(gx_val * gx_val + gy_val * gy_val);
        Halide::Expr mag_val = Halide::sqrt(sum_sq) + 1.0f;
        
        gradients(x, y) = Halide::Tuple(gx_val, gy_val, mag_val);

        // Schedule
        Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
        
        gradients.compute_root()
                 .parallel(y)
                 .vectorize(x, 16);

        pipeline_ = std::make_unique<Halide::Pipeline>(gradients);
        Halide::Target target = Halide::get_host_target();
        pipeline_->compile_jit(target);
    }

    ImageParam input_;
    std::unique_ptr<Halide::Pipeline> pipeline_;
    std::once_flag init_flag_;
};

GradientPipeline &get_gradient_pipeline() {
    static GradientPipeline pipeline;
    return pipeline;
}

} // namespace

extern "C" void halide_gradient(image_u8_t *im, image_u8_t *gx_im, image_u8_t *gy_im, image_u8_t *mag_im)
{
    if (!im || !gx_im || !gy_im || !mag_im) return;

    GradientPipeline &pipeline = get_gradient_pipeline();
    pipeline.compile_once();

    Halide::Buffer<uint8_t> input_buf(im->buf, im->width, im->height);
    
    // UMich uses int16 for gradients in some places, but here we need to match the storage.
    // Wait, `compute_lfps` calculates them on the fly.
    // If we want to precompute them, we need buffers.
    // The user (me) needs to allocate these buffers in C and pass them in.
    // Let's assume they are int16 images (image_u16_t doesn't exist in common/image_u8.h, but we can cast).
    // Actually, `image_u8_t` is 8-bit. We might need a new struct or just use `int16_t*` buffer.
    // For now, let's assume we pass `image_u8_t` but cast the buffer to `int16_t*`.
    // WARNING: stride is in bytes.
    
    // Let's define the C interface to take raw pointers and strides for outputs to be safe.
}

// Redefine C interface to be more flexible
extern "C" void halide_compute_gradients(
    uint8_t *input_buf, int w, int h, int s,
    int16_t *gx_buf, int gx_s,
    int16_t *gy_buf, int gy_s,
    float *mag_buf, int mag_s)
{
    GradientPipeline &pipeline = get_gradient_pipeline();
    pipeline.compile_once();

    Halide::Buffer<uint8_t> input(input_buf, w, h);
    input.raw_buffer()->dim[0].stride = 1;
    input.raw_buffer()->dim[1].stride = s;

    Halide::Buffer<int16_t> gx(gx_buf, w, h);
    gx.raw_buffer()->dim[0].stride = 1;
    gx.raw_buffer()->dim[1].stride = gx_s / sizeof(int16_t); // stride in elements

    Halide::Buffer<int16_t> gy(gy_buf, w, h);
    gy.raw_buffer()->dim[0].stride = 1;
    gy.raw_buffer()->dim[1].stride = gy_s / sizeof(int16_t);

    Halide::Buffer<float> mag(mag_buf, w, h);
    mag.raw_buffer()->dim[0].stride = 1;
    mag.raw_buffer()->dim[1].stride = mag_s / sizeof(float);

    pipeline.run(input, gx, gy, mag);
}
