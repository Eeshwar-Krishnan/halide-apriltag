#include "HalideBuffer.h"
#include "halide_image_io.h"

#include "apriltag_edge_detect.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

using Halide::Runtime::Buffer;
using namespace Halide::Tools;

// =============================================================================
// Data Structures
// =============================================================================

struct Point2D {
    float x, y;
    Point2D(float x = 0, float y = 0) : x(x), y(y) {}
    Point2D operator+(const Point2D& o) const { return {x + o.x, y + o.y}; }
    Point2D operator-(const Point2D& o) const { return {x - o.x, y - o.y}; }
    Point2D operator*(float s) const { return {x * s, y * s}; }
    float dot(const Point2D& o) const { return x * o.x + y * o.y; }
    float cross(const Point2D& o) const { return x * o.y - y * o.x; }
    float norm() const { return std::sqrt(x * x + y * y); }
    Point2D normalized() const { float n = norm(); return n > 0 ? Point2D(x/n, y/n) : Point2D(); }
};

struct Quad {
    Point2D corners[4];
    float confidence;
    
    bool is_valid() const {
        float side_lengths[4];
        for (int i = 0; i < 4; i++) {
            side_lengths[i] = (corners[(i+1)%4] - corners[i]).norm();
        }
        
        float min_side = *std::min_element(side_lengths, side_lengths + 4);
        float max_side = *std::max_element(side_lengths, side_lengths + 4);
        
        if (min_side < 15.0f) return false;
        if (max_side / min_side > 4.0f) return false;
        
        // Check convexity
        for (int i = 0; i < 4; i++) {
            Point2D v1 = corners[(i+1)%4] - corners[i];
            Point2D v2 = corners[(i+2)%4] - corners[(i+1)%4];
            if (v1.cross(v2) < 0) return false;
        }
        return true;
    }
};

// =============================================================================
// Union-Find for Connected Components
// =============================================================================

class UnionFind {
public:
    std::vector<int> parent, rank_vec;
    
    UnionFind(int n) : parent(n), rank_vec(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }
    
    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }
    
    void unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return;
        if (rank_vec[px] < rank_vec[py]) std::swap(px, py);
        parent[py] = px;
        if (rank_vec[px] == rank_vec[py]) rank_vec[px]++;
    }
};

// =============================================================================
// Convex Hull (Andrew's Monotone Chain)
// =============================================================================

std::vector<Point2D> convex_hull(std::vector<Point2D> points) {
    int n = points.size();
    if (n < 3) return points;
    
    std::sort(points.begin(), points.end(), [](const Point2D& a, const Point2D& b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    
    std::vector<Point2D> hull;
    
    // Lower hull
    for (int i = 0; i < n; i++) {
        while (hull.size() >= 2) {
            Point2D a = hull[hull.size() - 2];
            Point2D b = hull[hull.size() - 1];
            if ((b - a).cross(points[i] - a) <= 0) hull.pop_back();
            else break;
        }
        hull.push_back(points[i]);
    }
    
    // Upper hull
    int lower_size = hull.size();
    for (int i = n - 2; i >= 0; i--) {
        while (hull.size() > lower_size) {
            Point2D a = hull[hull.size() - 2];
            Point2D b = hull[hull.size() - 1];
            if ((b - a).cross(points[i] - a) <= 0) hull.pop_back();
            else break;
        }
        hull.push_back(points[i]);
    }
    
    hull.pop_back();
    return hull;
}

// =============================================================================
// Quad Fitting from Hull
// =============================================================================

Quad fit_quad_to_hull(const std::vector<Point2D>& hull, float total_perimeter) {
    Quad quad;
    quad.confidence = total_perimeter;
    
    if (hull.size() < 4) return quad;
    
    int n = hull.size();
    
    // Find 4 corners with maximum curvature
    std::vector<std::pair<float, int>> curvatures;
    
    for (int i = 0; i < n; i++) {
        Point2D prev = hull[(i - 1 + n) % n];
        Point2D curr = hull[i];
        Point2D next = hull[(i + 1) % n];
        
        Point2D v1 = (curr - prev).normalized();
        Point2D v2 = (next - curr).normalized();
        
        float angle = std::atan2(std::abs(v1.cross(v2)), v1.dot(v2));
        curvatures.push_back({angle, i});
    }
    
    std::sort(curvatures.begin(), curvatures.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::vector<int> corner_indices;
    for (int i = 0; i < std::min(4, (int)curvatures.size()); i++) {
        corner_indices.push_back(curvatures[i].second);
    }
    
    std::sort(corner_indices.begin(), corner_indices.end());
    
    if (corner_indices.size() < 4) {
        // Fallback to extreme points
        Point2D centroid(0, 0);
        for (const auto& p : hull) { centroid.x += p.x; centroid.y += p.y; }
        centroid.x /= n; centroid.y /= n;
        
        float best_score[4] = {-1e9f, -1e9f, -1e9f, -1e9f};
        int best_idx[4] = {0, 0, 0, 0};
        Point2D dirs[4] = {{1, -1}, {1, 1}, {-1, 1}, {-1, -1}};
        
        for (int i = 0; i < n; i++) {
            Point2D delta = hull[i] - centroid;
            for (int d = 0; d < 4; d++) {
                float score = delta.dot(dirs[d]);
                if (score > best_score[d]) {
                    best_score[d] = score;
                    best_idx[d] = i;
                }
            }
        }
        
        corner_indices.clear();
        for (int d = 0; d < 4; d++) corner_indices.push_back(best_idx[d]);
        std::sort(corner_indices.begin(), corner_indices.end());
    }
    
    for (int i = 0; i < 4; i++) {
        quad.corners[i] = hull[corner_indices[i]];
    }
    
    return quad;
}

// =============================================================================
// Find Quads from Binary Image (like original AprilTag)
// =============================================================================

// Check if pixel is on boundary of a black region
inline bool is_boundary_pixel(const Buffer<uint8_t>& binary, int x, int y) {
    if (binary(x, y) == 0) return false;  // Not black
    
    // Boundary if any 4-connected neighbor is white (0)
    int w = binary.width(), h = binary.height();
    if (x > 0 && binary(x-1, y) == 0) return true;
    if (x < w-1 && binary(x+1, y) == 0) return true;
    if (y > 0 && binary(x, y-1) == 0) return true;
    if (y < h-1 && binary(x, y+1) == 0) return true;
    return false;
}

std::vector<Quad> find_quads_from_binary(const Buffer<uint8_t>& binary, int min_area, int max_area) {
    int width = binary.width();
    int height = binary.height();
    
    // Find connected components of BLACK pixels using Union-Find
    UnionFind uf(width * height);
    
    // 4-connectivity for black pixels
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (binary(x, y) == 0) continue;  // Skip white
            
            int idx = y * width + x;
            
            if (x + 1 < width && binary(x + 1, y) > 0) {
                uf.unite(idx, idx + 1);
            }
            if (y + 1 < height && binary(x, y + 1) > 0) {
                uf.unite(idx, idx + width);
            }
        }
    }
    
    // Gather BOUNDARY points for each component
    std::unordered_map<int, std::vector<Point2D>> component_boundary;
    std::unordered_map<int, int> component_area;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (binary(x, y) == 0) continue;
            
            int idx = y * width + x;
            int root = uf.find(idx);
            
            component_area[root]++;
            
            // Only add boundary pixels
            if (is_boundary_pixel(binary, x, y)) {
                component_boundary[root].push_back({(float)x, (float)y});
            }
        }
    }
    
    std::vector<Quad> quads;
    
    for (auto& [root, boundary_points] : component_boundary) {
        int area = component_area[root];
        
        // Filter by area
        if (area < min_area || area > max_area) continue;
        
        // Need enough boundary points
        if (boundary_points.size() < 20) continue;
        
        // Compute bounding box
        float min_x = 1e9f, max_x = -1e9f, min_y = 1e9f, max_y = -1e9f;
        for (const auto& p : boundary_points) {
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y);
            max_y = std::max(max_y, p.y);
        }
        
        float box_width = max_x - min_x;
        float box_height = max_y - min_y;
        
        // Must be reasonably sized
        if (box_width < 15 || box_height < 15) continue;
        if (box_width > width * 0.8f || box_height > height * 0.8f) continue;
        
        // Aspect ratio check (AprilTags are roughly square)
        float aspect = std::max(box_width, box_height) / std::max(1.0f, std::min(box_width, box_height));
        if (aspect > 3.0f) continue;
        
        // Check "rectangularity" - boundary should have points distributed around all 4 sides
        bool has_top = false, has_bottom = false, has_left = false, has_right = false;
        float margin = 0.25f;
        
        for (const auto& p : boundary_points) {
            float rel_x = (p.x - min_x) / box_width;
            float rel_y = (p.y - min_y) / box_height;
            
            if (rel_y < margin) has_top = true;
            if (rel_y > 1.0f - margin) has_bottom = true;
            if (rel_x < margin) has_left = true;
            if (rel_x > 1.0f - margin) has_right = true;
        }
        
        if (!has_top || !has_bottom || !has_left || !has_right) continue;
        
        // Compute convex hull of boundary
        std::vector<Point2D> hull = convex_hull(boundary_points);
        if (hull.size() < 4) continue;
        
        // Fit quad
        Quad quad = fit_quad_to_hull(hull, (float)area);
        
        if (quad.is_valid()) {
            quads.push_back(quad);
        }
    }
    
    // Sort by area (larger is better confidence for AprilTags)
    std::sort(quads.begin(), quads.end(), [](const Quad& a, const Quad& b) {
        return a.confidence > b.confidence;
    });
    
    // Non-maximum suppression
    std::vector<bool> suppressed(quads.size(), false);
    std::vector<Quad> result;
    
    for (size_t i = 0; i < quads.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(quads[i]);
        
        Point2D ci = (quads[i].corners[0] + quads[i].corners[2]) * 0.5f;
        float ri = (quads[i].corners[2] - quads[i].corners[0]).norm() / 2;
        
        for (size_t j = i + 1; j < quads.size(); j++) {
            if (suppressed[j]) continue;
            Point2D cj = (quads[j].corners[0] + quads[j].corners[2]) * 0.5f;
            if ((cj - ci).norm() < ri * 0.5f) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

// =============================================================================
// FAST Quad Detection (Decimation + Raw Pointers + Flattened Union-Find)
// =============================================================================

struct FastUF {
    int* parent;
    int n;
    
    FastUF(int size) : n(size) {
        parent = new int[size];
        for (int i = 0; i < size; i++) parent[i] = i;
    }
    ~FastUF() { delete[] parent; }
    
    inline int find(int x) {
        int root = x;
        while (root != parent[root]) root = parent[root];
        while (x != root) { int next = parent[x]; parent[x] = root; x = next; }
        return root;
    }
    
    inline void unite(int x, int y) {
        int rx = find(x), ry = find(y);
        if (rx != ry) parent[ry] = rx;
    }
    
    void flatten() {
        for (int i = 0; i < n; i++) {
            parent[i] = find(i);
        }
    }
};

std::vector<Quad> find_quads_fast(const Buffer<uint8_t>& binary, int min_area, int max_area, int decimation = 1) {
    const int w = binary.width();
    const int h = binary.height();
    const int s_w = w / decimation;
    const int s_h = h / decimation;
    
    const uint8_t* __restrict__ ptr = binary.data();
    const int stride = binary.stride(1);

    // Pass 1: Union-Find on decimated grid
    FastUF uf(s_w * s_h);
    
    for (int sy = 0; sy < s_h; sy++) {
        const int y = sy * decimation;
        const uint8_t* row = ptr + y * stride;
        
        for (int sx = 0; sx < s_w; sx++) {
            const int x = sx * decimation;
            if (row[x] == 0) continue;
            
            const int idx = sy * s_w + sx;
            
            // Connect right
            if (sx + 1 < s_w && row[x + decimation] != 0) {
                uf.unite(idx, idx + 1);
            }
            // Connect down
            if (sy + 1 < s_h && ptr[(y + decimation) * stride + x] != 0) {
                uf.unite(idx, idx + s_w);
            }
        }
    }
    
    // KEY OPTIMIZATION: Flatten all parents for O(1) lookup in pass 2
    uf.flatten();

    // Pass 2: Collect boundary points + count area
    // Use vectors indexed by root (sparse, but fast)
    std::vector<std::vector<Point2D>> clusters(s_w * s_h);
    std::vector<int> area_counts(s_w * s_h, 0);
    std::vector<int> active_roots;
    active_roots.reserve(256);

    for (int sy = 1; sy < s_h - 1; sy++) {
        const int y = sy * decimation;
        const uint8_t* row = ptr + y * stride;
        const uint8_t* row_up = ptr + (y - decimation) * stride;
        const uint8_t* row_down = ptr + (y + decimation) * stride;

        for (int sx = 1; sx < s_w - 1; sx++) {
            const int x = sx * decimation;
            if (row[x] == 0) continue;

            const int root = uf.parent[sy * s_w + sx];  // O(1) - already flattened!
            
            if (area_counts[root] == 0) {
                active_roots.push_back(root);
            }
            area_counts[root]++;

            // Boundary check
            if ((row[x - decimation] == 0) | (row[x + decimation] == 0) |
                (row_up[x] == 0) | (row_down[x] == 0)) {
                clusters[root].emplace_back((float)x, (float)y);
            }
        }
    }

    // Pass 3: Fit quads
    const int scaled_min_area = min_area / (decimation * decimation);
    const int scaled_max_area = max_area / (decimation * decimation);
    
    std::vector<Quad> quads;
    quads.reserve(active_roots.size());

    for (int root : active_roots) {
        const int area = area_counts[root];
        if (area < scaled_min_area || area > scaled_max_area) continue;

        std::vector<Point2D>& boundary = clusters[root];
        if (boundary.size() < 10) continue;

        // Bounding box check
        float min_x = 1e9f, max_x = -1e9f, min_y = 1e9f, max_y = -1e9f;
        for (const auto& p : boundary) {
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y);
            max_y = std::max(max_y, p.y);
        }
        if ((max_x - min_x) < 15 || (max_y - min_y) < 15) continue;

        std::vector<Point2D> hull = convex_hull(boundary);
        if (hull.size() < 4) continue;

        Quad quad = fit_quad_to_hull(hull, (float)area * decimation * decimation);
        if (quad.is_valid()) {
            quads.push_back(quad);
        }
    }
    
    std::sort(quads.begin(), quads.end(), [](const Quad& a, const Quad& b) {
        return a.confidence > b.confidence;
    });

    return quads;
}

// =============================================================================
// Visualization
// =============================================================================

Buffer<float> convert_to_grayscale(const Buffer<uint8_t>& input) {
    Buffer<float> gray(input.width(), input.height());
    
    if (input.channels() == 1) {
        for (int y = 0; y < input.height(); ++y) {
            for (int x = 0; x < input.width(); ++x) {
                gray(x, y) = static_cast<float>(input(x, y));
            }
        }
    } else {
        for (int y = 0; y < input.height(); ++y) {
            for (int x = 0; x < input.width(); ++x) {
                float r = input(x, y, 0);
                float g = input(x, y, 1);
                float b = input(x, y, 2);
                gray(x, y) = 0.299f * r + 0.587f * g + 0.114f * b;
            }
        }
    }
    return gray;
}

void draw_line(Buffer<uint8_t>& buf, Point2D p1, Point2D p2, uint8_t r, uint8_t g, uint8_t b) {
    int steps = std::max(1, (int)(std::abs(p2.x - p1.x) + std::abs(p2.y - p1.y)));
    for (int i = 0; i <= steps; ++i) {
        float t = (float)i / steps;
        int x = (int)(p1.x + t * (p2.x - p1.x));
        int y = (int)(p1.y + t * (p2.y - p1.y));
        if (x >= 0 && x < buf.width() && y >= 0 && y < buf.height()) {
            buf(x, y, 0) = r;
            buf(x, y, 1) = g;
            buf(x, y, 2) = b;
        }
    }
}

Buffer<uint8_t> visualize_quads(const Buffer<uint8_t>& input, const std::vector<Quad>& quads) {
    Buffer<uint8_t> vis(input.width(), input.height(), 3);
    
    for (int y = 0; y < input.height(); ++y) {
        for (int x = 0; x < input.width(); ++x) {
            uint8_t v = (input.channels() == 1) ? input(x, y) :
                        (uint8_t)(0.299f * input(x, y, 0) + 0.587f * input(x, y, 1) + 0.114f * input(x, y, 2));
            vis(x, y, 0) = vis(x, y, 1) = vis(x, y, 2) = v;
        }
    }
    
    for (const auto& q : quads) {
        for (int i = 0; i < 4; ++i) {
            draw_line(vis, q.corners[i], q.corners[(i+1)%4], 255, 0, 0);
        }
        
        for (int i = 0; i < 4; ++i) {
            int cx = (int)q.corners[i].x;
            int cy = (int)q.corners[i].y;
            for (int dy = -3; dy <= 3; ++dy) {
                for (int dx = -3; dx <= 3; ++dx) {
                    if (dx*dx + dy*dy <= 9) {
                        int px = cx + dx, py = cy + dy;
                        if (px >= 0 && px < vis.width() && py >= 0 && py < vis.height()) {
                            vis(px, py, 0) = 0;
                            vis(px, py, 1) = 255;
                            vis(px, py, 2) = 0;
                        }
                    }
                }
            }
        }
    }
    
    return vis;
}

Buffer<uint8_t> visualize_edges(const Buffer<uint8_t>& edges) {
    Buffer<uint8_t> vis(edges.width(), edges.height(), 3);
    
    for (int y = 0; y < edges.height(); ++y) {
        for (int x = 0; x < edges.width(); ++x) {
            uint8_t v = edges(x, y);
            vis(x, y, 0) = v;
            vis(x, y, 1) = v;
            vis(x, y, 2) = v;
        }
    }
    
    return vis;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    try {
        using Clock = std::chrono::steady_clock;
        auto to_ms = [](Clock::duration d) {
            return std::chrono::duration<double, std::milli>(d).count();
        };

        std::vector<std::pair<std::string, double>> timings;
        const auto program_start = Clock::now();

        const char* input_path = (argc > 1) ? argv[1] : "../apriltags.png";
        
        std::cout << "=== AprilTag Adaptive Threshold Pipeline ===" << std::endl;
        std::cout << "Loading: " << input_path << std::endl;
        std::cout << "(No manual tuning required - adapts to image automatically)" << std::endl;
        
        auto stage_start = Clock::now();
        Buffer<uint8_t> input = load_image(input_path);
        auto stage_end = Clock::now();
        timings.emplace_back("load_image", to_ms(stage_end - stage_start));
        std::cout << "Image dimensions: " << input.width() << "x" << input.height() 
                  << "x" << input.channels() << std::endl;
        
        // =================================================================
        // Stage 1: GPU - Grayscale + Adaptive Threshold (binary image)
        // Grayscale conversion is now fused into the Halide pipeline
        // =================================================================
        Buffer<uint8_t> binary(input.width(), input.height());
        
        stage_start = Clock::now();
        int result = atag_edge_detect(input, binary);
        stage_end = Clock::now();
        timings.emplace_back("grayscale+threshold (GPU)", to_ms(stage_end - stage_start));
        
        if (result != 0) {
            throw std::runtime_error("Halide pipeline failed: " + std::to_string(result));
        }
        
        std::cout << "Stage 'grayscale+threshold (GPU)' completed in "
                  << timings.back().second << " ms" << std::endl;
        
        // Copy to host
        stage_start = Clock::now();
        binary.copy_to_host();
        stage_end = Clock::now();
        timings.emplace_back("copy_to_host", to_ms(stage_end - stage_start));
        std::cout << "Copy to host: " << timings.back().second << " ms" << std::endl;
        
        // Count black pixels
        stage_start = Clock::now();
        int black_count = 0;
        for (int y = 0; y < binary.height(); y++) {
            for (int x = 0; x < binary.width(); x++) {
                if (binary(x, y) > 0) black_count++;
            }
        }
        stage_end = Clock::now();
        timings.emplace_back("black_pixel_count", to_ms(stage_end - stage_start));
        std::cout << "Black pixels: " << black_count << " (" 
                  << (100.0f * black_count / (binary.width() * binary.height())) << "%)" << std::endl;
        
        // =================================================================
        // Stage 2: CPU - Connected Components + Quad Fitting
        // =================================================================
        // Min/max area based on expected tag sizes (adaptive to image size)
        int img_area = binary.width() * binary.height();
        int min_area = img_area / 2000;   // Tags should be at least 0.05% of image
        int max_area = img_area / 4;      // Tags should be at most 25% of image
        
        stage_start = Clock::now();
        std::vector<Quad> quads = find_quads_fast(binary, min_area, max_area, 2);
        stage_end = Clock::now();
        timings.emplace_back("quad_detect (CPU)", to_ms(stage_end - stage_start));
        
        std::cout << "Stage 'quad_detect (CPU)' completed in "
                  << timings.back().second << " ms" << std::endl;
        std::cout << "Found " << quads.size() << " quads" << std::endl;
        
        // =================================================================
        // Output
        // =================================================================
        for (size_t i = 0; i < quads.size(); ++i) {
            const auto& q = quads[i];
            std::cout << "Quad " << i << ": [";
            for (int j = 0; j < 4; ++j) {
                std::cout << "(" << (int)q.corners[j].x << ", " << (int)q.corners[j].y << ")";
                if (j < 3) std::cout << ", ";
            }
            std::cout << "] perimeter=" << (int)q.confidence << std::endl;
        }
        
        // Save visualizations
        stage_start = Clock::now();
        Buffer<uint8_t> binary_vis = visualize_edges(binary);
        save_image(binary_vis, "binary_output.png");
        stage_end = Clock::now();
        timings.emplace_back("save_binary_output", to_ms(stage_end - stage_start));
        std::cout << "Saved: binary_output.png" << std::endl;
        
        stage_start = Clock::now();
        Buffer<uint8_t> quads_vis = visualize_quads(input, quads);
        save_image(quads_vis, "quads_output.png");
        stage_end = Clock::now();
        timings.emplace_back("save_quads_output", to_ms(stage_end - stage_start));
        std::cout << "Saved: quads_output.png" << std::endl;
        
        // Cleanup
        binary.device_free();

        const auto program_end = Clock::now();

        std::cout << "\n--- Timing Summary (ms) ---" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        for (const auto& entry : timings) {
            std::cout << "  " << std::left << std::setw(28) << entry.first
                      << " : " << entry.second << std::endl;
        }
        std::cout << "  " << std::left << std::setw(28) << "total_wall_time"
                  << " : " << to_ms(program_end - program_start) << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
