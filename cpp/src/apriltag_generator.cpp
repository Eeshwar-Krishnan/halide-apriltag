#include "Halide.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace Halide;


Func gaussian_down(Func in, Expr width, Expr height) {
    Var x, y;
    Func blur_x("blur_x"), blur_y("blur_y");
    blur_x(x, y) = (in(clamp(x-1, 0, width-1), y) + 2.0f*in(x, y) + in(clamp(x+1, 0, width-1), y)) / 4.0f;
    blur_y(x, y) = (blur_x(x, clamp(y-1, 0, height-1)) + 2.0f*blur_x(x, y) + blur_x(x, clamp(y+1, 0, height-1))) / 4.0f;
    Func down("down");
    down(x, y) = select(blur_y(clamp(2*x, 0, width-1), clamp(2*y, 0, height-1)) >= 50.0f, 50, 0);
    Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
    down.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
    return down;
}

Func laplacian_delta(Func g_low, Func g_high) {
    Var x, y;
    Func up("up");
    up(x, y) = g_high(x/2, y/2);  // simple nearest; use bilinear if you prefer
    Func lap("lap");
    lap(x, y) = g_low(x, y) - up(x, y);
    Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
    lap.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
    return lap;
}

Func flood_once(Func img, Func seed_labels) {
    Var x, y;
    RDom n(-1, 3, -1, 3);
    Func next("next");
    next(x, y) = select(
        img(x, y) > 0,
        min(seed_labels(x, y), minimum(seed_labels(x+n.x, y+n.y))),
        0
    );
    return next;
}

Func hierarchical_flood(Func input, int levels, int passes_per_level) {
    Var x, y;

    // ---- Build Gaussian pyramid ----
    std::vector<Func> G(levels);
    G[0] = input;
    for (int i = 1; i < levels; i++) {
        //G[i] = gaussian_down(G[i-1]);
    }

    // ---- Build Laplacian deltas ----
    std::vector<Func> L(levels-1);
    for (int i = 0; i < levels-1; i++) {
        L[i] = laplacian_delta(G[i], G[i+1]);
    }

    // ---- Coarse flood ----
    Func labels_coarse("labels_coarse");

    // initial labeling (unique ID per pixel)
    labels_coarse(x, y) = select(
        G[levels-1](x, y) > 0,
        (y << 16) + x + 1,  // unique label without needing image width
        0
    );

    for (int iter = 0; iter < 3; iter++) {
        labels_coarse = flood_once(G[levels-1], labels_coarse);
    }

    // ---- Propagate upward ----
    Func cur = labels_coarse;

    for (int level = levels-2; level >= 0; level--) {
        Func up("up_" + std::to_string(level));
        up(x, y) = cur(x/2, y/2);

        Expr edge_mask = abs(L[level](x, y)) > 0.05f;

        Func refine = up;
        for (int pass = 0; pass < passes_per_level; pass++) {
            RDom n(-1, 3, -1, 3);
            Func tmp("refine_" + std::to_string(level) + "_" + std::to_string(pass));
            tmp(x, y) = select(
                edge_mask,
                min(refine(x, y), minimum(refine(x+n.x, y+n.y))),
                refine(x, y)
            );
            refine = tmp;
        }

        cur = refine;
    }

    return cur;
}

namespace {

Func build_binary_pipeline(ImageParam input) {
    Var x("x"), y("y");

    const int tile_size = 4;

    Expr width = input.width();
    Expr height = input.height();
    Expr tiles_x = (width + tile_size - 1) / tile_size;
    Expr tiles_y = (height + tile_size - 1) / tile_size;

    Func tile_min("tile_min"), tile_max("tile_max");
    Var tx("tx"), ty("ty");

    tile_min(tx, ty) = cast<float>(255.0f);
    tile_max(tx, ty) = cast<float>(0.0f);

    RDom r_tile(0, tile_size, 0, tile_size);
    Expr px = clamp(tx * tile_size + r_tile.x, 0, width - 1);
    Expr py = clamp(ty * tile_size + r_tile.y, 0, height - 1);
    tile_min(tx, ty) = min(tile_min(tx, ty), input(px, py));
    tile_max(tx, ty) = max(tile_max(tx, ty), input(px, py));

    Func binary("binary");
    Expr tile_x = clamp(x / tile_size, 0, tiles_x - 1);
    Expr tile_y = clamp(y / tile_size, 0, tiles_y - 1);

    Expr min_val = cast<float>(255.0f);
    Expr max_val = cast<float>(0.0f);
    for (int dy = -1; dy <= 1; ++dy) {
        Expr ty_n = clamp(tile_y + dy, 0, tiles_y - 1);
        for (int dx = -1; dx <= 1; ++dx) {
            Expr tx_n = clamp(tile_x + dx, 0, tiles_x - 1);
            min_val = min(min_val, tile_min(tx_n, ty_n));
            max_val = max(max_val, tile_max(tx_n, ty_n));
        }
    }

    Expr threshold = min(max((min_val + max_val) * 0.5f, 100.0f), 50.0f);
    binary(x, y) = cast<uint8_t>(select(input(x, y) > threshold && (max_val - min_val) > 100, 50, 0));

    tile_min.compute_root();
    tile_max.compute_root();
    tile_min.update().parallel(ty);
    tile_max.update().parallel(ty);

    binary.compute_root();
    binary.parallel(y).vectorize(x, 16);

    return binary;
}

Func build_edge_pipeline(ImageParam binary_in) {
    Var x("x"), y("y");

    Expr max_x = binary_in.dim(0).extent() - 1;
    Expr max_y = binary_in.dim(1).extent() - 1;

    Expr center = binary_in(x, y);
    Expr left = binary_in(clamp(x - 1, 0, max_x), y);
    Expr right = binary_in(clamp(x + 1, 0, max_x), y);
    Expr up = binary_in(x, clamp(y - 1, 0, max_y));
    Expr down = binary_in(x, clamp(y + 1, 0, max_y));

    Expr is_edge = (center != left) || (center != right) || (center != up) || (center != down);

    Func edge("edge");
    edge(x, y) = select(is_edge, 1, 0);

    edge.compute_root();
    edge.parallel(y).vectorize(x, 16);

    // Temporarily disable GPU tiling to isolate the issue
    // Var xi("xi"), yi("yi");
    // edge.gpu_tile(x, y, xi, yi, 16, 16);

    return edge;
}

Func build_density_pipeline(ImageParam edge_in, Param<int> n, Param<float> threshold_ratio) {
    Var x("x"), y("y");
    Var xi("xi"), yi("yi");

    Expr max_x = edge_in.dim(0).extent() - 1;
    Expr max_y = edge_in.dim(1).extent() - 1;

    // Ultra-fast separable density filter using n×n neighborhood
    // First pass: horizontal sum over n pixels
    Func hsum("hsum");

    // Build horizontal sum using a loop-like approach for arbitrary n
    Expr hsum_val = cast<int32_t>(0);
    for (int i = 0; i < 7; ++i) { // Support up to 7x7 neighborhood
        Expr offset = i - 3; // Center around 0, support -3 to +3
        Expr nx = clamp(x + offset, 0, max_x);
        Expr pixel_binary = select(edge_in(nx, y) > 0, 1, 0);
        // Only include pixels within the actual neighborhood size
        Expr within_neighborhood = (i >= (3 - n/2)) && (i <= (3 + n/2));
        hsum_val = hsum_val + select(within_neighborhood, pixel_binary, 0);
    }

    hsum(x, y) = hsum_val;

    // Second pass: vertical sum over n pixels using horizontal sums
    Func density_sum("density_sum");

    // Build vertical sum using horizontal sums
    Expr vsum_val = cast<int32_t>(0);
    for (int i = 0; i < 7; ++i) { // Support up to 7x7 neighborhood
        Expr offset = i - 3; // Center around 0, support -3 to +3
        Expr ny = clamp(y + offset, 0, max_y);
        // Only include pixels within the actual neighborhood size
        Expr within_neighborhood = (i >= (3 - n/2)) && (i <= (3 + n/2));
        vsum_val = vsum_val + select(within_neighborhood, hsum(x, ny), 0);
    }

    density_sum(x, y) = vsum_val;

    // Calculate threshold based on ratio of active neighbors
    Expr max_possible_neighbors = cast<float>(n * n);
    Expr threshold_count = cast<int32_t>(max_possible_neighbors * threshold_ratio);

    // Apply density threshold (remove pixels with too many active neighbors)
    Func density("density");
    density(x, y) = select(density_sum(x, y) > threshold_count, 0, edge_in(x, y));

    // Optimize scheduling for maximum performance
    hsum.compute_root();
    density_sum.compute_root();
    density.compute_root();

    return density;
}

Func build_lut_pipeline(ImageParam density_in) {
    Var x("x"), y("y");

    Expr width = cast<int32_t>(density_in.dim(0).extent());

    Func lut("lut");
    Expr idx = cast<int32_t>(x) + cast<int32_t>(y) * width;
    lut(x, y) = select(density_in(x, y) > 0, idx + 1, 0);

    lut.compute_root();

    // Temporarily disable GPU tiling to debug assertion error
    // Var xi("xi"), yi("yi");
    // lut.gpu_tile(x, y, xi, yi, 16, 16);

    return lut;
}

Func build_full_pipeline(ImageParam input) {
    Var x("x"), y("y");
    Var xo("xo"), yo("yo"), xi("xi"), yi("yi"); // Add GPU tile variables

    const int tile_size = 4;

    Expr width = input.width();
    Expr height = input.height();
    Expr tiles_x = (width + tile_size - 1) / tile_size;
    Expr tiles_y = (height + tile_size - 1) / tile_size;

    Func tile_min("tile_min"), tile_max("tile_max");
    Var tx("tx"), ty("ty");

    tile_min(tx, ty) = cast<float>(255.0f);
    tile_max(tx, ty) = cast<float>(0.0f);

    RDom r_tile(0, tile_size, 0, tile_size);
    Expr px = clamp(tx * tile_size + r_tile.x, 0, width - 1);
    Expr py = clamp(ty * tile_size + r_tile.y, 0, height - 1);
    tile_min(tx, ty) = min(tile_min(tx, ty), input(px, py));
    tile_max(tx, ty) = max(tile_max(tx, ty), input(px, py));

    Func binary("binary");
    Expr tile_x = clamp(x / tile_size, 0, tiles_x - 1);
    Expr tile_y = clamp(y / tile_size, 0, tiles_y - 1);

    Expr min_val = cast<float>(255.0f);
    Expr max_val = cast<float>(0.0f);
    for (int dy = -1; dy <= 1; ++dy) {
        Expr ty_n = clamp(tile_y + dy, 0, tiles_y - 1);
        for (int dx = -1; dx <= 1; ++dx) {
            Expr tx_n = clamp(tile_x + dx, 0, tiles_x - 1);
            min_val = min(min_val, tile_min(tx_n, ty_n));
            max_val = max(max_val, tile_max(tx_n, ty_n));
        }
    }

    Expr threshold = max((min_val + max_val) * 0.5f, 10.0f);
    binary(x, y) = cast<int32_t>(select(input(x, y) > 75, 50, 0));

    // tile_min.compute_root();
    // tile_max.compute_root();
    // //tile_min.update().parallel(ty);
    // //tile_max.update().parallel(ty);

    // //binary.parallel(y).vectorize(x, 16);

    // // GPU scheduling for tile functions
    // tile_min.compute_root().gpu_tile(tx, ty, xo, yo, xi, yi, 16, 16);
    // tile_max.compute_root().gpu_tile(tx, ty, xo, yo, xi, yi, 16, 16);

    //binary.compute_root();

    // GPU scheduling for binary function
    binary.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
    
    tile_min.compute_at(binary, xo).store_at(binary, xo);
    tile_max.compute_at(binary, xo).store_at(binary, xo);

    Var x2("x2"), y2("y2");

    Expr max_x = input.width() - 1;
    Expr max_y = input.height() - 1;

    Expr center = binary(x2, y2);
    Expr left = binary(clamp(x2 - 1, 0, max_x), y2);
    Expr right = binary(clamp(x2 + 1, 0, max_x), y2);
    Expr up = binary(x2, clamp(y2 - 1, 0, max_y));
    Expr down = binary(x2, clamp(y2 + 1, 0, max_y));

    Expr is_edge_x_1 = (center != left) || (center != right);
    Expr is_edge_y_1 = (center != up) || (center != down);

    Func edge("edge");
    //edge(x2, y2) = select(is_edge, 1, 0);
    edge(x2, y2) = binary(x2, y2);

    Func edge_x_1("edge_x_1");
    edge_x_1(x2, y2) = select(is_edge_x_1, 1, 0);
    Func edge_y_1("edge_y_1");
    edge_y_1(x2, y2) = select(is_edge_y_1, 1, 0);

    edge_x_1.compute_root().gpu_tile(x2, y2, xo, yo, xi, yi, 16, 16);
    edge_y_1.compute_root().gpu_tile(x2, y2, xo, yo, xi, yi, 16, 16);

    Func R;
    Expr a = edge_x_1(x2,y2)*edge_x_1(x2,y2);
    Expr b = edge_y_1(x2,y2)*edge_y_1(x2,y2);
    Expr c = edge_x_1(x2,y2)*edge_y_1(x2,y2);
    Expr det = a*b - c*c;
    Expr trace = a + b;

    R(x2, y2) = cast<int32_t>((det - 0.04f * trace * trace)*100.0f);
    //R(x2, y2) = cast<int32_t>(edge_x_1(x2, y2) + edge_y_1(x2, y2));

    R.compute_root().gpu_tile(x2, y2, xo, yo, xi, yi, 16, 16);

    Func iter_f;
    iter_f = R;

    Func iter_fs[100];
    iter_fs[0] = iter_f;

    RDom nloc(-1, 3, -1, 3);
    for(int i = 0; i < 100; i++) {
        Func iter_f_tmp("iter_f_tmp_" + std::to_string(i));
        iter_f_tmp(x2, y2) = select(
            iter_f(x2, y2) != 0,
            max(iter_f(x2, y2), maximum(iter_f(clamp(x2 + nloc.x, 0, width - 1), clamp(y2 + nloc.y, 0, height - 1)))),
            0
        );

        //iter_f_tmp.compute_root().gpu_tile(x2, y2, xo, yo, xi, yi, 16, 16);
        iter_fs[i] = iter_f_tmp;
        iter_f = iter_f_tmp;
    }

    for(int i = 98; i >= 0; i--) {
        //iter_fs[i].compute_at(iter_f, xi);
        iter_fs[i].compute_root().gpu_tile(x2, y2, xo, yo, xi, yi, 32, 8);
    }

    iter_f.compute_root().gpu_tile(x2, y2, xo, yo, xi, yi, 16, 16);

    //return iter_f;

    //edge.compute_root();
    //edge.parallel(y2).vectorize(x2, 16);
    edge.compute_root().gpu_tile(x2, y2, xo, yo, xi, yi, 16, 16);

    int levels = 6;
    Func pyramid[levels];
    pyramid[0] = R;
    for (int i = 1; i < levels; i++) {
        pyramid[i] = gaussian_down(pyramid[i-1], input.width(), input.height());
    }

    Func laplacians[levels-1];
    for (int i = 0; i < levels-1; i++) {
        laplacians[i] = laplacian_delta(pyramid[i], pyramid[i+1]);
    }

    Var x3("x3"), y3("y3");
    Func level_labels("level_labels");
    level_labels(x3, y3) = select(pyramid[levels-1](x3, y3) > 0, (((x3 * (2 * (levels + 1)) + (y3 * (2 * (levels + 1))))) + (width * height * width * height)) * R(x3, y3), 0);

    int passes_per_level = 2;
    RDom n(-1, 3, -1, 3);
    Func level_labels_iter = level_labels;

    for(int j = 0; j < passes_per_level; j++) {
        Func level_labels_iter_tmp("level_labels_iter_tmp_start_" + std::to_string(j));
        level_labels_iter_tmp(x3, y3) = select(
            level_labels_iter(x3, y3) != 0,
            max(level_labels_iter(x3, y3), maximum(level_labels_iter(clamp(x3 + n.x, 0, width - 1), clamp(y3 + n.y, 0, height - 1)))),
            0
        );
        level_labels_iter_tmp.compute_root().gpu_tile(x3, y3, xo, yo, xi, yi, 16, 16);
        //level_labels_iter_tmp.compute_at(level_labels_iter, xo).store_at(level_labels_iter, xo);
        level_labels_iter = level_labels_iter_tmp;
    }

    passes_per_level = 10;

    for(int i = levels-1; i > 0; i--) {

        Func upscaled_tmp("upscaled_tmp_" + std::to_string(i));
        upscaled_tmp(x3, y3) = level_labels_iter(x3/2, y3/2);

        Func upscaled_propagate("upscaled_propagate_" + std::to_string(i));
        upscaled_propagate(x3, y3) = select(
            laplacians[i-1](x3, y3) > 0,
            max(((x3 * (2 * (i))) + (y3 * (2 * (i)) * width)) + (width * height * i), maximum(upscaled_tmp(clamp(x3 + n.x, 0, width - 1), clamp(y3 + n.y, 0, height - 1)))),
            upscaled_tmp(x3, y3)
        );
        

        upscaled_propagate.compute_root().gpu_tile(x3, y3, xo, yo, xi, yi, 16, 16);

        //upscaled_propagate.compute_at(level_labels_iter, xo).store_at(level_labels_iter, xo);

        Func level_labels_tmp = upscaled_propagate;

        if(i == 1) {
            passes_per_level = 10;
        }

        for(int j = 0; j < passes_per_level; j++) {
            Func level_labels_iter_tmp("level_labels_iter_tmp_" + std::to_string(i) + "_" + std::to_string(j));
            level_labels_iter_tmp(x3, y3) = select(
                level_labels_tmp(x3, y3) != 0,
                max(level_labels_tmp(x3, y3), maximum(level_labels_tmp(clamp(x3 + n.x, 0, width - 1), clamp(y3 + n.y, 0, height - 1)))),
                0
            );
            level_labels_iter_tmp.compute_root().gpu_tile(x3, y3, xo, yo, xi, yi, 16, 16);
            //level_labels_iter_tmp.compute_at(level_labels_tmp, xo).store_at(level_labels_tmp, xo);
            //upscaled_propagate.compute_at(level_labels_iter_tmp, Var::outermost());
            level_labels_tmp = level_labels_iter_tmp;
        }

        Func upscaled_propagate2("upscaled_propagate2_" + std::to_string(i));
        upscaled_propagate2(x3, y3) = select(
            laplacians[i-1](x3, y3) < 0,
            0,
            level_labels_tmp(x3, y3)
        );

        level_labels_iter = upscaled_propagate2;
        level_labels_iter.compute_root().gpu_tile(x3, y3, xo, yo, xi, yi, 16, 16);

        if(i == 0) {
            return level_labels_iter;
        }

    }

    return level_labels_iter;

    // Func edge_downsampled1("edge_downsampled1");
    // edge_downsampled1 = gaussian_down(edge, input.width(), input.height());

    // Func laplace_delta1("laplace_delta1");
    // laplace_delta1 = laplacian_delta(edge, edge_downsampled1);

    // Func edge_downsampled("edge_downsampled");
    // edge_downsampled = gaussian_down(edge_downsampled1, input.width() / 2, input.height() / 2);

    // Func laplace_delta2("laplace_delta2");
    // laplace_delta2 = laplacian_delta(edge_downsampled1, edge_downsampled);

    // Func labels("labels");

    // //labels(x3, y3) = 0;

    // //labels.compute_root();

    // labels(x3, y3) = select(edge_downsampled(x3, y3) > 0, (x3) + (y3 * width), 0);

    // //labels.parallel(y3).vectorize(x3, 16);
    // //labels.compute_root();
    // labels.compute_root().gpu_tile(x3, y3, xo, yo, xi, yi, 16, 16);

    // //labels_downsampled.compute_root().gpu_tile(x3, y3, xo, yo, xi, yi, 16, 16);

    // RDom n0(-1, 3, -1, 3);

    // Func labels_propagate("labels_propagate");
    // labels_propagate(x3, y3) = labels(x3, y3);

    // Func labels_propagate_iter = labels_propagate;
    // int iters = 25;

    // for (int i = 0; i < iters; i++) {
    //     Func g("labels_propagate_iter_" + std::to_string(i));
    //     g(x3, y3) = select(
    //         labels_propagate_iter(x3, y3) != 0,
    //         max(labels_propagate_iter(x3, y3), maximum(labels_propagate_iter(clamp(x3 + n0.x, 0, max_x), clamp(y3 + n0.y, 0, max_y)))),
    //         0
    //     );
    //     labels_propagate_iter = g;
    //     labels_propagate_iter.compute_root().gpu_tile(x3, y3, xo, yo, xi, yi, 32, 32);
    // }


    // Func labels_up1("labels_up1");
    // Var x4("x4"), y4("y4");

    // labels_up1(x4, y4) = labels_propagate_iter(x4/2, y4/2);

    // Func labels_up1_propagate("labels_up1_propagate");

    // labels_up1_propagate(x4, y4) = select(
    //     laplace_delta2(x4, y4) > 0,
    //     1,
    //     labels_up1(x4, y4)
    // );

    // Func labels_up1_propagate2("labels_up1_propagate2");

    // labels_up1_propagate2(x4, y4) = select(
    //     laplace_delta2(x4, y4) < 0,
    //     0,
    //     labels_up1_propagate(x4, y4)
    // );

    // labels_up1_propagate2.compute_root().gpu_tile(x4, y4, xo, yo, xi, yi, 32, 32);

    // Func f = labels_up1_propagate2;

    // iters = 25;

    // for (int i = 0; i < iters; i++) {
    //     Func g("labels_up1_propagate_iter_" + std::to_string(i));
    //     g(x4, y4) = select(
    //         f(x4, y4) != 0,
    //         max(f(x4, y4), maximum(f(clamp(x4 + n0.x, 0, max_x), clamp(y4 + n0.y, 0, max_y)))),
    //         0
    //     );
    //     f = g;
    //     f.compute_root().gpu_tile(x4, y4, xo, yo, xi, yi, 32, 32);
    // }



    // Func labels_up2("labels_up12");

    // labels_up2(x4, y4) = f(x4/2, y4/2);

    // Func labels_up2_propagate("labels_up2_propagate");

    // labels_up2_propagate(x4, y4) = select(
    //     laplace_delta1(x4, y4) > 0,
    //     max(1, maximum(labels_up2(clamp(x4 + n0.x, 0, max_x), clamp(y4 + n0.y, 0, max_y)))),
    //     labels_up2(x4, y4)
    // );

    // Func labels_up2_propagate2("labels_up2_propagate2");

    // labels_up2_propagate2(x4, y4) = select(
    //     laplace_delta1(x4, y4) < 0,
    //     0,
    //     labels_up2(x4, y4)
    // );
}

// =============================================================================
// AprilTag-style Adaptive Thresholding Pipeline
// =============================================================================
// Like the original AprilTag: uses local min/max for adaptive thresholding
// No manual tuning required - automatically adapts to image contrast

constexpr int TILE_SIZE = 4;  // Local neighborhood for adaptive threshold
constexpr int MIN_CONTRAST = 30;  // Minimum contrast to consider an edge (fixed, robust)

// Input is now RGB uint8 (3-channel), grayscale conversion is done on GPU
Func build_adaptive_threshold_pipeline(ImageParam input_rgb) {
    Var x("x"), y("y"), c("c");
    Var xo("xo"), yo("yo"), xi("xi"), yi("yi");

    Expr width = input_rgb.width();
    Expr height = input_rgb.height();
    Expr max_x = width - 1;
    Expr max_y = height - 1;

    // Step 0: Convert RGB to grayscale on GPU
    // Standard luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
    Func gray("gray");
    Expr red = cast<float>(input_rgb(clamp(x, 0, max_x), clamp(y, 0, max_y), 0));
    Expr green = cast<float>(input_rgb(clamp(x, 0, max_x), clamp(y, 0, max_y), 1));
    Expr blue = cast<float>(input_rgb(clamp(x, 0, max_x), clamp(y, 0, max_y), 2));
    gray(x, y) = 0.299f * red + 0.587f * green + 0.114f * blue;
    
    // Schedule grayscale conversion on GPU
    gray.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);

    // Step 1: Clamp input (now using grayscale)
    Func clamped("clamped");
    clamped(x, y) = gray(clamp(x, 0, max_x), clamp(y, 0, max_y));

    // Step 2: Compute local min/max in TILE_SIZE x TILE_SIZE neighborhoods
    // This is the key to adaptive thresholding - no manual tuning needed
    Var tx("tx"), ty("ty");
    Expr tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
    Expr tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
    
    RDom r(0, TILE_SIZE, 0, TILE_SIZE, "tile_minmax");
    
    Func tile_min("tile_min"), tile_max("tile_max");
    tile_min(tx, ty) = 255.0f;
    tile_max(tx, ty) = 0.0f;
    
    Expr px = clamp(tx * TILE_SIZE + r.x, 0, max_x);
    Expr py = clamp(ty * TILE_SIZE + r.y, 0, max_y);
    tile_min(tx, ty) = min(tile_min(tx, ty), clamped(px, py));
    tile_max(tx, ty) = max(tile_max(tx, ty), clamped(px, py));

    // Step 3: For each pixel, look up the min/max from surrounding tiles
    // and compute adaptive threshold
    Expr tile_x = clamp(x / TILE_SIZE, 0, tiles_x - 1);
    Expr tile_y = clamp(y / TILE_SIZE, 0, tiles_y - 1);
    
    // Sample 3x3 neighborhood of tiles for smoother thresholding
    Func local_min("local_min"), local_max("local_max");
    Expr lmin = cast<float>(255.0f);
    Expr lmax = cast<float>(0.0f);
    
    for (int dy = -1; dy <= 1; ++dy) {
        Expr ty_n = clamp(tile_y + dy, 0, tiles_y - 1);
        for (int dx = -1; dx <= 1; ++dx) {
            Expr tx_n = clamp(tile_x + dx, 0, tiles_x - 1);
            lmin = min(lmin, tile_min(tx_n, ty_n));
            lmax = max(lmax, tile_max(tx_n, ty_n));
        }
    }
    
    local_min(x, y) = lmin;
    local_max(x, y) = lmax;

    // Step 4: Adaptive threshold
    // threshold = (min + max) / 2
    // Only mark as foreground if contrast is sufficient
    Func threshold("threshold");
    Expr thresh = (local_min(x, y) + local_max(x, y)) / 2.0f;
    Expr contrast = local_max(x, y) - local_min(x, y);
    
    // Binary: 255 if dark (below threshold), 0 if light, with contrast check
    Func binary("binary");
    binary(x, y) = cast<uint8_t>(select(
        contrast > MIN_CONTRAST && clamped(x, y) < thresh,
        255,
        0
    ));

    // Output the binary image directly - let CPU find boundaries of connected components
    // This is more like the original AprilTag algorithm
    
    // Schedule for GPU
    tile_min.compute_root().gpu_tile(tx, ty, xo, yo, xi, yi, 16, 16);
    tile_min.update().unscheduled();
    tile_max.compute_root().gpu_tile(tx, ty, xo, yo, xi, yi, 16, 16);
    tile_max.update().unscheduled();
    
    binary.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);

    return binary;
}

} // namespace

int main(int argc, char **argv) {
    try {
        using Clock = std::chrono::steady_clock;
        auto to_ms = [](Clock::duration d) {
            return std::chrono::duration<double, std::milli>(d).count();
        };
        std::vector<std::pair<std::string, double>> timings;
        const auto program_start = Clock::now();

        Target target = get_target_from_environment();

        // =========================================================================
        // AprilTag-style Adaptive Thresholding Pipeline (no tuning required)
        // Accepts RGB uint8 input, performs grayscale conversion on GPU
        // =========================================================================
        auto stage_start = Clock::now();
        ImageParam input_rgb(UInt(8), 3, "input_rgb");  // RGB input, grayscale done on GPU
        
        Func edge_detect = build_adaptive_threshold_pipeline(input_rgb);
        auto stage_end = Clock::now();
        timings.emplace_back("build_adaptive_threshold_pipeline", to_ms(stage_end - stage_start));
        
        std::vector<Argument> edge_detect_args = {input_rgb};

        // Configure GPU target
        Target gpu_target = target;
        gpu_target.set_feature(Target::Metal);
        //gpu_target.set_feature(Target::Profile);

        std::cout << "Compiling adaptive threshold pipeline..." << std::endl;
        stage_start = Clock::now();
        edge_detect.compile_to_static_library("apriltag_edge_detect", edge_detect_args, "atag_edge_detect", gpu_target);
        stage_end = Clock::now();
        timings.emplace_back("compile_to_static_library", to_ms(stage_end - stage_start));

        stage_start = Clock::now();
        edge_detect.compile_to_header("apriltag_edge_detect.h", edge_detect_args, "atag_edge_detect", gpu_target);
        stage_end = Clock::now();
        timings.emplace_back("compile_to_header", to_ms(stage_end - stage_start));

        std::cout << "All pipelines compiled successfully!" << std::endl;

        const auto program_end = Clock::now();
        std::cout << "\n--- Timing Summary (ms) ---" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        for (const auto& entry : timings) {
            std::cout << "  " << std::left << std::setw(32) << entry.first
                      << " : " << entry.second << std::endl;
        }
        std::cout << "  " << std::left << std::setw(32) << "total_wall_time"
                  << " : " << to_ms(program_end - program_start) << std::endl;
        return 0;
    } catch (const Halide::CompileError &e) {
        std::cerr << "Halide compilation error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
