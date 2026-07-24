/*
 * omega_raytrace - BVH-accelerated raytracing for Source 2 / Deadlock visibility checks
 * 
 * Bounding Volume Hierarchy (BVH) for O(log n) ray queries
 * Möller-Trumbore algorithm for ray-triangle intersection
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

/* ========== Triangle/BVH Structures ========== */

typedef struct {
    float v0[3];
    float v1[3];
    float v2[3];
} Triangle;

typedef struct {
    float min[3];
    float max[3];
} AABB;

typedef struct {
    AABB bounds;
    int left;
    int right;
    int tri_count;
    int _pad;
} BVHNode;

/* ========== Global State ========== */

static Triangle* g_triangles = NULL;
static int g_tri_count = 0;
static BVHNode* g_bvh_nodes = NULL;
static int g_bvh_node_count = 0;
static int* g_tri_indices = NULL;

/* Constants */
#define MAX_LEAF_TRIS 3
#define EPSILON 1e-6f

/* ========== Math Helpers ========== */

static inline float minf(float a, float b) { return a < b ? a : b; }
static inline float maxf(float a, float b) { return a > b ? a : b; }

/* ========== AABB Helpers ========== */

static inline void aabb_init(AABB* box) {
    box->min[0] = box->min[1] = box->min[2] = FLT_MAX;
    box->max[0] = box->max[1] = box->max[2] = -FLT_MAX;
}

static inline void aabb_expand_point(AABB* box, const float* p) {
    for (int i = 0; i < 3; i++) {
        if (p[i] < box->min[i]) box->min[i] = p[i];
        if (p[i] > box->max[i]) box->max[i] = p[i];
    }
}

static inline void aabb_expand_triangle(AABB* box, const Triangle* tri) {
    aabb_expand_point(box, tri->v0);
    aabb_expand_point(box, tri->v1);
    aabb_expand_point(box, tri->v2);
}

static inline void aabb_get_centroid(const Triangle* tri, float* centroid) {
    centroid[0] = (tri->v0[0] + tri->v1[0] + tri->v2[0]) / 3.0f;
    centroid[1] = (tri->v0[1] + tri->v1[1] + tri->v2[1]) / 3.0f;
    centroid[2] = (tri->v0[2] + tri->v1[2] + tri->v2[2]) / 3.0f;
}

/* Ray-AABB intersection (slab method) - returns 1 if intersect, 0 otherwise */
static inline int ray_aabb_intersect(
    const float* origin, const float* inv_dir,
    const AABB* box, float t_max
) {
    float t1, t2, t_min = 0.0f;
    
    for (int i = 0; i < 3; i++) {
        t1 = (box->min[i] - origin[i]) * inv_dir[i];
        t2 = (box->max[i] - origin[i]) * inv_dir[i];
        
        if (t1 > t2) {
            float tmp = t1; t1 = t2; t2 = tmp;
        }
        
        if (t1 > t_min) t_min = t1;
        if (t2 < t_max) t_max = t2;
        
        if (t_min > t_max) return 0;
    }
    
    return 1;
}

/* ========== BVH Construction ========== */

static int* g_sort_axis_ptr;
static Triangle* g_sort_tris_ptr;

static int compare_centroid(const void* a, const void* b) {
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    int axis = *g_sort_axis_ptr;
    
    float ca[3], cb[3];
    aabb_get_centroid(&g_sort_tris_ptr[ia], ca);
    aabb_get_centroid(&g_sort_tris_ptr[ib], cb);
    
    if (ca[axis] < cb[axis]) return -1;
    if (ca[axis] > cb[axis]) return 1;
    return 0;
}

static int bvh_build_recursive(int start, int count) {
    if (g_bvh_node_count >= g_tri_count * 2) {
        return -1;
    }
    
    int node_idx = g_bvh_node_count++;
    BVHNode* node = &g_bvh_nodes[node_idx];
    aabb_init(&node->bounds);
    
    for (int i = 0; i < count; i++) {
        aabb_expand_triangle(&node->bounds, &g_triangles[g_tri_indices[start + i]]);
    }
    
    if (count <= MAX_LEAF_TRIS) {
        node->left = start;
        node->right = -1;
        node->tri_count = count;
        return node_idx;
    }
    
    float size[3] = {
        node->bounds.max[0] - node->bounds.min[0],
        node->bounds.max[1] - node->bounds.min[1],
        node->bounds.max[2] - node->bounds.min[2]
    };
    
    int axis = 0;
    if (size[1] > size[0]) axis = 1;
    if (size[2] > size[axis]) axis = 2;
    
    g_sort_axis_ptr = &axis;
    g_sort_tris_ptr = g_triangles;
    qsort(&g_tri_indices[start], count, sizeof(int), compare_centroid);
    
    int mid = count / 2;
    
    node->tri_count = 0;
    node->left = bvh_build_recursive(start, mid);
    node->right = bvh_build_recursive(start + mid, count - mid);
    
    return node_idx;
}

static void bvh_build(void) {
    if (g_tri_count == 0 || !g_triangles) return;
    
    g_bvh_nodes = (BVHNode*)malloc(sizeof(BVHNode) * g_tri_count * 2);
    g_tri_indices = (int*)malloc(sizeof(int) * g_tri_count);
    
    if (!g_bvh_nodes || !g_tri_indices) {
        if (g_bvh_nodes) free(g_bvh_nodes);
        if (g_tri_indices) free(g_tri_indices);
        g_bvh_nodes = NULL;
        g_tri_indices = NULL;
        return;
    }
    
    for (int i = 0; i < g_tri_count; i++) {
        g_tri_indices[i] = i;
    }
    
    g_bvh_node_count = 0;
    bvh_build_recursive(0, g_tri_count);
}

/* ========== Ray Intersections ========== */

static inline int ray_triangle_intersect(
    const float* origin, const float* dir,
    const Triangle* tri, float* out_t
) {
    float edge1[3], edge2[3], h[3], s[3], q[3];
    float a, f, u, v;
    
    edge1[0] = tri->v1[0] - tri->v0[0];
    edge1[1] = tri->v1[1] - tri->v0[1];
    edge1[2] = tri->v1[2] - tri->v0[2];
    
    edge2[0] = tri->v2[0] - tri->v0[0];
    edge2[1] = tri->v2[1] - tri->v0[1];
    edge2[2] = tri->v2[2] - tri->v0[2];
    
    h[0] = dir[1] * edge2[2] - dir[2] * edge2[1];
    h[1] = dir[2] * edge2[0] - dir[0] * edge2[2];
    h[2] = dir[0] * edge2[1] - dir[1] * edge2[0];
    
    a = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2];
    
    if (a > -EPSILON && a < EPSILON) return 0;
    
    f = 1.0f / a;
    s[0] = origin[0] - tri->v0[0];
    s[1] = origin[1] - tri->v0[1];
    s[2] = origin[2] - tri->v0[2];
    
    u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
    if (u < 0.0f || u > 1.0f) return 0;
    
    q[0] = s[1] * edge1[2] - s[2] * edge1[1];
    q[1] = s[2] * edge1[0] - s[0] * edge1[2];
    q[2] = s[0] * edge1[1] - s[1] * edge1[0];
    
    v = f * (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]);
    if (v < 0.0f || u + v > 1.0f) return 0;
    
    float t = f * (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]);
    if (t > EPSILON) {
        *out_t = t;
        return 1;
    }
    
    return 0;
}

static int bvh_trace_ray(const float* origin, const float* dir, float max_dist) {
    if (!g_bvh_nodes || g_bvh_node_count == 0) return 0;
    
    float inv_dir[3] = {
        1.0f / (fabsf(dir[0]) > EPSILON ? dir[0] : (dir[0] < 0 ? -EPSILON : EPSILON)),
        1.0f / (fabsf(dir[1]) > EPSILON ? dir[1] : (dir[1] < 0 ? -EPSILON : EPSILON)),
        1.0f / (fabsf(dir[2]) > EPSILON ? dir[2] : (dir[2] < 0 ? -EPSILON : EPSILON))
    };
    
    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        const BVHNode* node = &g_bvh_nodes[node_idx];
        
        if (!ray_aabb_intersect(origin, inv_dir, &node->bounds, max_dist)) {
            continue;
        }
        
        if (node->tri_count > 0) {
            for (int i = 0; i < node->tri_count; i++) {
                int tri_idx = g_tri_indices[node->left + i];
                float t;
                if (ray_triangle_intersect(origin, dir, &g_triangles[tri_idx], &t)) {
                    if (t > EPSILON && t < max_dist - EPSILON) {
                        return 1;
                    }
                }
            }
        } else {
            if (node->right != -1 && stack_ptr < 64) {
                stack[stack_ptr++] = node->right;
            }
            if (node->left != -1 && stack_ptr < 64) {
                stack[stack_ptr++] = node->left;
            }
        }
    }
    
    return 0;
}

/* ========== Public API ========== */

EXPORT int rt_load(const char* filepath) {
    if (g_triangles) {
        free(g_triangles);
        g_triangles = NULL;
    }
    if (g_bvh_nodes) {
        free(g_bvh_nodes);
        g_bvh_nodes = NULL;
    }
    if (g_tri_indices) {
        free(g_tri_indices);
        g_tri_indices = NULL;
    }
    g_tri_count = 0;
    g_bvh_node_count = 0;
    
    FILE* f = fopen(filepath, "rb");
    if (!f) return -1;
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (size % 36 != 0) {
        fclose(f);
        return -1;
    }
    
    int count = size / 36;
    
    g_triangles = (Triangle*)malloc(sizeof(Triangle) * count);
    if (!g_triangles) {
        fclose(f);
        return -1;
    }
    
    size_t read = fread(g_triangles, sizeof(Triangle), count, f);
    fclose(f);
    
    if ((int)read != count) {
        free(g_triangles);
        g_triangles = NULL;
        return -1;
    }
    
    g_tri_count = count;
    bvh_build();
    
    return count;
}

EXPORT int rt_is_visible(
    float x1, float y1, float z1,
    float x2, float y2, float z2
) {
    if (!g_triangles || g_tri_count == 0) return 1;
    
    float origin[3] = {x1, y1, z1};
    float target[3] = {x2, y2, z2};
    
    float dir[3];
    dir[0] = target[0] - origin[0];
    dir[1] = target[1] - origin[1];
    dir[2] = target[2] - origin[2];
    
    float dist = sqrtf(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
    if (dist < EPSILON) return 1;
    
    dir[0] /= dist;
    dir[1] /= dist;
    dir[2] /= dist;
    
    if (g_bvh_nodes && g_bvh_node_count > 0) {
        return bvh_trace_ray(origin, dir, dist) ? 0 : 1;
    }
    
    for (int i = 0; i < g_tri_count; i++) {
        float t;
        if (ray_triangle_intersect(origin, dir, &g_triangles[i], &t)) {
            if (t > EPSILON && t < dist - EPSILON) {
                return 0;
            }
        }
    }
    
    return 1;
}

EXPORT void rt_unload(void) {
    if (g_triangles) {
        free(g_triangles);
        g_triangles = NULL;
    }
    if (g_bvh_nodes) {
        free(g_bvh_nodes);
        g_bvh_nodes = NULL;
    }
    if (g_tri_indices) {
        free(g_tri_indices);
        g_tri_indices = NULL;
    }
    g_tri_count = 0;
    g_bvh_node_count = 0;
}

EXPORT int rt_get_triangle_count(void) {
    return g_tri_count;
}

EXPORT int rt_get_bvh_node_count(void) {
    return g_bvh_node_count;
}
