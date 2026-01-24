#include "ACMMP.h"

#define mul4(v, k)       \
    {                    \
        v->x = v->x * k; \
        v->y = v->y * k; \
        v->z = v->z * k; \
    }

#define vecdiv4(v, k)    \
    {                    \
        v->x = v->x / k; \
        v->y = v->y / k; \
        v->z = v->z / k; \
    }

__device__ void sort_small(float *d, const int n)
{
    int j;
    for (int i = 1; i < n; i++)
    {
        float tmp = d[i];
        for (j = i; j >= 1 && tmp < d[j - 1]; j--)
            d[j] = d[j - 1];
        d[j] = tmp;
    }
}

__device__ void sort_small_weighted(float *d, float *w, int n)
{
    int j;
    for (int i = 1; i < n; i++)
    {
        float tmp = d[i];
        float tmp_w = w[i];
        for (j = i; j >= 1 && tmp < d[j - 1]; j--)
        {
            d[j] = d[j - 1];
            w[j] = w[j - 1];
        }
        d[j] = tmp;
        w[j] = tmp_w;
    }
}

__device__ int FindMinCostIndex(const float *costs, const int n)
{
    float min_cost = costs[0];
    int min_cost_idx = 0;
    for (int idx = 1; idx < n; ++idx)
    {
        if (costs[idx] <= min_cost)
        {
            min_cost = costs[idx];
            min_cost_idx = idx;
        }
    }
    return min_cost_idx;
}

__device__ int FindMaxCostIndex(const float *costs, const int n)
{
    float max_cost = costs[0];
    int max_cost_idx = 0;
    for (int idx = 1; idx < n; ++idx)
    {
        if (costs[idx] >= max_cost)
        {
            max_cost = costs[idx];
            max_cost_idx = idx;
        }
    }
    return max_cost_idx;
}

__device__ void setBit(unsigned int &input, const unsigned int n)
{
    input |= (unsigned int)(1 << n);
}

__device__ int isSet(unsigned int input, const unsigned int n)
{
    return (input >> n) & 1;
}

__device__ void Mat33DotVec3(const float mat[9], const float4 vec, float4 *result)
{
    result->x = mat[0] * vec.x + mat[1] * vec.y + mat[2] * vec.z;
    result->y = mat[3] * vec.x + mat[4] * vec.y + mat[5] * vec.z;
    result->z = mat[6] * vec.x + mat[7] * vec.y + mat[8] * vec.z;
}

__device__ float Vec3DotVec3(const float4 vec1, const float4 vec2)
{
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

__device__ void NormalizeVec3(float4 *vec)
{
    const float normSquared = vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
    const float inverse_sqrt = rsqrtf(normSquared);
    vec->x *= inverse_sqrt;
    vec->y *= inverse_sqrt;
    vec->z *= inverse_sqrt;
}

// ==================== 函数：将概率分布转换为累积分布 ====================
//
// 【函数名解释】
// Transform PDF To CDF
// PDF = Probability Density Function (概率密度函数)
// CDF = Cumulative Distribution Function (累积分布函数)
//
// 【函数目的】
// 将概率分布(PDF)转换为累积分布(CDF)，用于后续的逆变换采样
// 这是蒙特卡洛采样中的关键步骤
//
// 【输入参数】
// - probs: 指向概率数组的指针
//   输入：PDF数组，各元素是不同事件的概率
//   输出：CDF数组，各元素是累积概率
// - num_probs: 概率数组的长度(事件总数)
//
// 【转换原理】
// PDF (概率分布):
//   例如 [0.2, 0.3, 0.1, 0.4]
//   含义：事件0的概率20%，事件1的概率30%，...
//
// CDF (累积分布):
//   例如 [0.2, 0.5, 0.6, 1.0]
//   含义：累积概率 P(X ≤ i)
//        - CDF[0] = 0.2 (事件0的概率)
//        - CDF[1] = 0.2 + 0.3 = 0.5 (事件0或1的概率)
//        - CDF[2] = 0.5 + 0.1 = 0.6 (事件0、1或2的概率)
//        - CDF[3] = 0.6 + 0.4 = 1.0 (所有事件的概率=100%)
//
// 【为什么需要CDF？】
// 在逆变换采样中：
//   1. 生成均匀随机数 r ∈ [0, 1]
//   2. 在CDF中找最小的i使得 CDF[i] > r
//   3. 返回事件i
// 时间复杂度：O(num_probs) for each sample
// 若直接用PDF需要多次条件判断，效率低
//
// 【转换步骤】
// Step 1: 计算所有概率之和(用于归一化)
// Step 2: 对每个概率归一化
// Step 3: 累积求和得到CDF

__device__ void TransformPDFToCDF(float *probs, const int num_probs)
{
    // ==================== Step 1: 计算概率总和 ====================
    // 累加所有概率值(原始PDF可能未经归一化)
    float prob_sum = 0.0f;
    for (int i = 0; i < num_probs; ++i)
    {
        prob_sum += probs[i];
    }

    // ==================== Step 2: 计算归一化因子 ====================
    // inv_prob_sum = 1 / (所有概率之和)
    // 用于将概率归一化到[0, 1]范围
    // 例：如果 prob_sum = 2.5，则 inv_prob_sum = 0.4
    //     这样每个概率都乘以0.4，使得总和变为1.0
    const float inv_prob_sum = 1.0f / prob_sum;

    // ==================== Step 3: 累积计算得到CDF ====================
    // 从第一个概率开始，逐个累加
    float cum_prob = 0.0f; // 累积概率，初始为0

    for (int i = 0; i < num_probs; ++i)
    {
        // 【标准化单个概率】
        // prob = probs[i] / prob_sum
        // 确保所有概率都在[0, 1]范围内，且总和为1
        // 例：原始 probs[0] = 0.2, prob_sum = 2.5
        //     标准化后 prob = 0.2 × 0.4 = 0.08
        const float prob = probs[i] * inv_prob_sum;

        // 【累积求和】
        // cum_prob += prob
        // 不断累加当前和之前所有的概率
        // 例：
        //     i=0: cum_prob = 0 + 0.08 = 0.08
        //     i=1: cum_prob = 0.08 + 0.12 = 0.20
        //     i=2: cum_prob = 0.20 + 0.05 = 0.25
        //     i=3: cum_prob = 0.25 + 0.75 = 1.00
        cum_prob += prob;

        // 【存储累积概率回原数组】
        // 原数组被改写：probs[i] 从概率值 → 累积概率值
        // 从此probs数组变成了CDF
        probs[i] = cum_prob;
    }
}

// ==================== 工作示例 ====================
//
// 【输入】
// probs = [2.0, 3.0, 1.0, 4.0]  (未归一化的PDF)
// num_probs = 4
//
// 【Step 1: 计算总和】
// prob_sum = 2.0 + 3.0 + 1.0 + 4.0 = 10.0
//
// 【Step 2: 计算归一化因子】
// inv_prob_sum = 1.0 / 10.0 = 0.1
//
// 【Step 3: 逐项处理(循环)】
// i=0: prob = 2.0 × 0.1 = 0.2
//      cum_prob = 0 + 0.2 = 0.2
//      probs[0] = 0.2  ✓
//
// i=1: prob = 3.0 × 0.1 = 0.3
//      cum_prob = 0.2 + 0.3 = 0.5
//      probs[1] = 0.5  ✓
//
// i=2: prob = 1.0 × 0.1 = 0.1
//      cum_prob = 0.5 + 0.1 = 0.6
//      probs[2] = 0.6  ✓
//
// i=3: prob = 4.0 × 0.1 = 0.4
//      cum_prob = 0.6 + 0.4 = 1.0
//      probs[3] = 1.0  ✓
//
// 【输出】
// probs = [0.2, 0.5, 0.6, 1.0]  (CDF)
//
// ==================== 后续采样流程 ====================
//
// 现在probs是CDF，可用于逆变换采样：
//
// for sample in 1..15:
//     r = random() ∈ [0, 1)
//     for i in 0..3:
//         if probs[i] > r:
//             选择事件i，break
//
// 例如 r = 0.35:
//   probs[0] = 0.2 < 0.35? 否
//   probs[1] = 0.5 > 0.35? 是 → 选择事件1✓
//
// 这样每个事件被选中的概率就符合原始PDF的分布！

__device__ void Get3DPoint(const Camera camera, const int2 p, const float depth, float *X)
{
    X[0] = depth * (p.x - camera.K[2]) / camera.K[0];
    X[1] = depth * (p.y - camera.K[5]) / camera.K[4];
    X[2] = depth;
}

__device__ float4 GetViewDirection(const Camera camera, const int2 p, const float depth)
{
    float X[3];
    Get3DPoint(camera, p, depth, X);
    float norm = sqrt(X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);

    float4 view_direction;
    view_direction.x = X[0] / norm;
    view_direction.y = X[1] / norm;
    view_direction.z = X[2] / norm;
    view_direction.w = 0;
    return view_direction;
}

__device__ float GetDistance2Origin(const Camera camera, const int2 p, const float depth, const float4 normal)
{
    float X[3];
    Get3DPoint(camera, p, depth, X);
    return -(normal.x * X[0] + normal.y * X[1] + normal.z * X[2]);
}

__device__ float SpatialGauss(float x1, float y1, float x2, float y2, float sigma, float mu = 0.0)
{
    float dis = pow(x1 - x2, 2) + pow(y1 - y2, 2) - mu;
    return exp(-1.0 * dis / (2 * sigma * sigma));
}

__device__ float RangeGauss(float x, float sigma, float mu = 0.0)
{
    float x_p = x - mu;
    return exp(-1.0 * (x_p * x_p) / (2 * sigma * sigma));
}

__device__ float ComputeDepthfromPlaneHypothesis(const Camera camera, const float4 plane_hypothesis, const int2 p)
{
    return -plane_hypothesis.w * camera.K[0] / ((p.x - camera.K[2]) * plane_hypothesis.x + (camera.K[0] / camera.K[4]) * (p.y - camera.K[5]) * plane_hypothesis.y + camera.K[0] * plane_hypothesis.z);
}

__device__ float4 GenerateRandomNormal(const Camera camera, const int2 p, curandState *rand_state, const float depth)
{
    float4 normal;
    float q1 = 1.0f;
    float q2 = 1.0f;
    float s = 2.0f;
    while (s >= 1.0f)
    {
        q1 = 2.0f * curand_uniform(rand_state) - 1.0f;
        q2 = 2.0f * curand_uniform(rand_state) - 1.0f;
        s = q1 * q1 + q2 * q2;
    }
    const float sq = sqrt(1.0f - s);
    normal.x = 2.0f * q1 * sq;
    normal.y = 2.0f * q2 * sq;
    normal.z = 1.0f - 2.0f * s;
    normal.w = 0;

    float4 view_direction = GetViewDirection(camera, p, depth);
    float dot_product = normal.x * view_direction.x + normal.y * view_direction.y + normal.z * view_direction.z;
    if (dot_product > 0.0f)
    {
        normal.x = -normal.x;
        normal.y = -normal.y;
        normal.z = -normal.z;
    }
    NormalizeVec3(&normal);
    return normal;
}

__device__ float4 GeneratePerturbedNormal(const Camera camera, const int2 p, const float4 normal, curandState *rand_state, const float perturbation)
{
    float4 view_direction = GetViewDirection(camera, p, 1.0f);

    const float a1 = (curand_uniform(rand_state) - 0.5f) * perturbation;
    const float a2 = (curand_uniform(rand_state) - 0.5f) * perturbation;
    const float a3 = (curand_uniform(rand_state) - 0.5f) * perturbation;

    const float sin_a1 = sin(a1);
    const float sin_a2 = sin(a2);
    const float sin_a3 = sin(a3);
    const float cos_a1 = cos(a1);
    const float cos_a2 = cos(a2);
    const float cos_a3 = cos(a3);

    float R[9];
    R[0] = cos_a2 * cos_a3;
    R[1] = cos_a3 * sin_a1 * sin_a2 - cos_a1 * sin_a3;
    R[2] = sin_a1 * sin_a3 + cos_a1 * cos_a3 * sin_a2;
    R[3] = cos_a2 * sin_a3;
    R[4] = cos_a1 * cos_a3 + sin_a1 * sin_a2 * sin_a3;
    R[5] = cos_a1 * sin_a2 * sin_a3 - cos_a3 * sin_a1;
    R[6] = -sin_a2;
    R[7] = cos_a2 * sin_a1;
    R[8] = cos_a1 * cos_a2;

    float4 normal_perturbed;
    Mat33DotVec3(R, normal, &normal_perturbed);

    if (Vec3DotVec3(normal_perturbed, view_direction) >= 0.0f)
    {
        normal_perturbed = normal;
    }

    NormalizeVec3(&normal_perturbed);
    return normal_perturbed;
}

__device__ float4 GenerateRandomPlaneHypothesis(const Camera camera, const int2 p, curandState *rand_state, const float depth_min, const float depth_max)
{
    float depth = curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;
    float4 plane_hypothesis = GenerateRandomNormal(camera, p, rand_state, depth);
    plane_hypothesis.w = GetDistance2Origin(camera, p, depth, plane_hypothesis);
    return plane_hypothesis;
}

__device__ float4 GeneratePertubedPlaneHypothesis(const Camera camera, const int2 p, curandState *rand_state, const float perturbation, const float4 plane_hypothesis_now, const float depth_now, const float depth_min, const float depth_max)
{
    float depth_perturbed = depth_now;

    float dist_perturbed = plane_hypothesis_now.w;
    const float dist_min_perturbed = (1 - perturbation) * dist_perturbed;
    const float dist_max_perturbed = (1 + perturbation) * dist_perturbed;
    float4 plane_hypothesis_temp = plane_hypothesis_now;
    do
    {
        dist_perturbed = curand_uniform(rand_state) * (dist_max_perturbed - dist_min_perturbed) + dist_min_perturbed;
        plane_hypothesis_temp.w = dist_perturbed;
        depth_perturbed = ComputeDepthfromPlaneHypothesis(camera, plane_hypothesis_temp, p);
    } while (depth_perturbed < depth_min && depth_perturbed > depth_max);

    float4 plane_hypothesis = GeneratePerturbedNormal(camera, p, plane_hypothesis_now, rand_state, perturbation * M_PI);
    plane_hypothesis.w = dist_perturbed;
    return plane_hypothesis;
}

__device__ void ComputeHomography(const Camera ref_camera, const Camera src_camera, const float4 plane_hypothesis, float *H)
{
    float ref_C[3];
    float src_C[3];
    ref_C[0] = -(ref_camera.R[0] * ref_camera.t[0] + ref_camera.R[3] * ref_camera.t[1] + ref_camera.R[6] * ref_camera.t[2]);
    ref_C[1] = -(ref_camera.R[1] * ref_camera.t[0] + ref_camera.R[4] * ref_camera.t[1] + ref_camera.R[7] * ref_camera.t[2]);
    ref_C[2] = -(ref_camera.R[2] * ref_camera.t[0] + ref_camera.R[5] * ref_camera.t[1] + ref_camera.R[8] * ref_camera.t[2]);
    src_C[0] = -(src_camera.R[0] * src_camera.t[0] + src_camera.R[3] * src_camera.t[1] + src_camera.R[6] * src_camera.t[2]);
    src_C[1] = -(src_camera.R[1] * src_camera.t[0] + src_camera.R[4] * src_camera.t[1] + src_camera.R[7] * src_camera.t[2]);
    src_C[2] = -(src_camera.R[2] * src_camera.t[0] + src_camera.R[5] * src_camera.t[1] + src_camera.R[8] * src_camera.t[2]);

    float R_relative[9];
    float C_relative[3];
    float t_relative[3];
    R_relative[0] = src_camera.R[0] * ref_camera.R[0] + src_camera.R[1] * ref_camera.R[1] + src_camera.R[2] * ref_camera.R[2];
    R_relative[1] = src_camera.R[0] * ref_camera.R[3] + src_camera.R[1] * ref_camera.R[4] + src_camera.R[2] * ref_camera.R[5];
    R_relative[2] = src_camera.R[0] * ref_camera.R[6] + src_camera.R[1] * ref_camera.R[7] + src_camera.R[2] * ref_camera.R[8];
    R_relative[3] = src_camera.R[3] * ref_camera.R[0] + src_camera.R[4] * ref_camera.R[1] + src_camera.R[5] * ref_camera.R[2];
    R_relative[4] = src_camera.R[3] * ref_camera.R[3] + src_camera.R[4] * ref_camera.R[4] + src_camera.R[5] * ref_camera.R[5];
    R_relative[5] = src_camera.R[3] * ref_camera.R[6] + src_camera.R[4] * ref_camera.R[7] + src_camera.R[5] * ref_camera.R[8];
    R_relative[6] = src_camera.R[6] * ref_camera.R[0] + src_camera.R[7] * ref_camera.R[1] + src_camera.R[8] * ref_camera.R[2];
    R_relative[7] = src_camera.R[6] * ref_camera.R[3] + src_camera.R[7] * ref_camera.R[4] + src_camera.R[8] * ref_camera.R[5];
    R_relative[8] = src_camera.R[6] * ref_camera.R[6] + src_camera.R[7] * ref_camera.R[7] + src_camera.R[8] * ref_camera.R[8];
    C_relative[0] = (ref_C[0] - src_C[0]);
    C_relative[1] = (ref_C[1] - src_C[1]);
    C_relative[2] = (ref_C[2] - src_C[2]);
    t_relative[0] = src_camera.R[0] * C_relative[0] + src_camera.R[1] * C_relative[1] + src_camera.R[2] * C_relative[2];
    t_relative[1] = src_camera.R[3] * C_relative[0] + src_camera.R[4] * C_relative[1] + src_camera.R[5] * C_relative[2];
    t_relative[2] = src_camera.R[6] * C_relative[0] + src_camera.R[7] * C_relative[1] + src_camera.R[8] * C_relative[2];

    H[0] = R_relative[0] - t_relative[0] * plane_hypothesis.x / plane_hypothesis.w;
    H[1] = R_relative[1] - t_relative[0] * plane_hypothesis.y / plane_hypothesis.w;
    H[2] = R_relative[2] - t_relative[0] * plane_hypothesis.z / plane_hypothesis.w;
    H[3] = R_relative[3] - t_relative[1] * plane_hypothesis.x / plane_hypothesis.w;
    H[4] = R_relative[4] - t_relative[1] * plane_hypothesis.y / plane_hypothesis.w;
    H[5] = R_relative[5] - t_relative[1] * plane_hypothesis.z / plane_hypothesis.w;
    H[6] = R_relative[6] - t_relative[2] * plane_hypothesis.x / plane_hypothesis.w;
    H[7] = R_relative[7] - t_relative[2] * plane_hypothesis.y / plane_hypothesis.w;
    H[8] = R_relative[8] - t_relative[2] * plane_hypothesis.z / plane_hypothesis.w;

    float tmp[9];
    tmp[0] = H[0] / ref_camera.K[0];
    tmp[1] = H[1] / ref_camera.K[4];
    tmp[2] = -H[0] * ref_camera.K[2] / ref_camera.K[0] - H[1] * ref_camera.K[5] / ref_camera.K[4] + H[2];
    tmp[3] = H[3] / ref_camera.K[0];
    tmp[4] = H[4] / ref_camera.K[4];
    tmp[5] = -H[3] * ref_camera.K[2] / ref_camera.K[0] - H[4] * ref_camera.K[5] / ref_camera.K[4] + H[5];
    tmp[6] = H[6] / ref_camera.K[0];
    tmp[7] = H[7] / ref_camera.K[4];
    tmp[8] = -H[6] * ref_camera.K[2] / ref_camera.K[0] - H[7] * ref_camera.K[5] / ref_camera.K[4] + H[8];

    H[0] = src_camera.K[0] * tmp[0] + src_camera.K[2] * tmp[6];
    H[1] = src_camera.K[0] * tmp[1] + src_camera.K[2] * tmp[7];
    H[2] = src_camera.K[0] * tmp[2] + src_camera.K[2] * tmp[8];
    H[3] = src_camera.K[4] * tmp[3] + src_camera.K[5] * tmp[6];
    H[4] = src_camera.K[4] * tmp[4] + src_camera.K[5] * tmp[7];
    H[5] = src_camera.K[4] * tmp[5] + src_camera.K[5] * tmp[8];
    H[6] = src_camera.K[8] * tmp[6];
    H[7] = src_camera.K[8] * tmp[7];
    H[8] = src_camera.K[8] * tmp[8];
}

__device__ float2 ComputeCorrespondingPoint(const float *H, const int2 p)
{
    float3 pt;
    pt.x = H[0] * p.x + H[1] * p.y + H[2];
    pt.y = H[3] * p.x + H[4] * p.y + H[5];
    pt.z = H[6] * p.x + H[7] * p.y + H[8];
    return make_float2(pt.x / pt.z, pt.y / pt.z);
}

__device__ float4 TransformNormal(const Camera camera, float4 plane_hypothesis)
{
    float4 transformed_normal;
    transformed_normal.x = camera.R[0] * plane_hypothesis.x + camera.R[3] * plane_hypothesis.y + camera.R[6] * plane_hypothesis.z;
    transformed_normal.y = camera.R[1] * plane_hypothesis.x + camera.R[4] * plane_hypothesis.y + camera.R[7] * plane_hypothesis.z;
    transformed_normal.z = camera.R[2] * plane_hypothesis.x + camera.R[5] * plane_hypothesis.y + camera.R[8] * plane_hypothesis.z;
    transformed_normal.w = plane_hypothesis.w;
    return transformed_normal;
}

__device__ float4 TransformNormal2RefCam(const Camera camera, float4 plane_hypothesis)
{
    float4 transformed_normal;
    transformed_normal.x = camera.R[0] * plane_hypothesis.x + camera.R[1] * plane_hypothesis.y + camera.R[2] * plane_hypothesis.z;
    transformed_normal.y = camera.R[3] * plane_hypothesis.x + camera.R[4] * plane_hypothesis.y + camera.R[5] * plane_hypothesis.z;
    transformed_normal.z = camera.R[6] * plane_hypothesis.x + camera.R[7] * plane_hypothesis.y + camera.R[8] * plane_hypothesis.z;
    transformed_normal.w = plane_hypothesis.w;
    return transformed_normal;
}

__device__ float ComputeBilateralWeight(const float x_dist, const float y_dist, const float pix, const float center_pix, const float sigma_spatial, const float sigma_color)
{
    const float spatial_dist = sqrt(x_dist * x_dist + y_dist * y_dist);
    const float color_dist = fabs(pix - center_pix);
    return exp(-spatial_dist / (2.0f * sigma_spatial * sigma_spatial) - color_dist / (2.0f * sigma_color * sigma_color));
}

// ==================== 函数：计算双边加权的归一化互相关 ====================
//
// 【函数名解释】
// Compute Bilateral-weighted NCC
// Bilateral = 双边的（既考虑空间距离，也考虑灰度值差异）
// NCC = Normalized Cross Correlation（归一化互相关）
//
// 【函数目的】
// 计算参考图像和源图像之间的匹配成本
// 使用双边加权方案，使相似像素的贡献更大
//
// 【核心思想】
// 传统NCC对所有像素等权重处理
// 双边NCC给予不同像素不同的权重：
//   - 空间上离中心近的像素：权重大
//   - 灰度上与中心像素相似的像素：权重大
//   - 两者都远的像素：权重小（甚至忽略）
//
// 这样能更好地处理图像边界和纹理边缘
//
// 【输入参数】
// - ref_image: 参考图像（纹理对象，GPU优化访问）
// - ref_camera: 参考相机参数
// - src_image: 源图像
// - src_camera: 源相机参数
// - p: 参考图像中的像素坐标(x, y)
// - plane_hypothesis: 当前平面假设
// - params: 算法参数（补丁大小、空间/灰度方差等）
//
// 【输出】
// 返回NCC成本，范围[0, 2]
//   - 0: 完美匹配
//   - 1: 完全不相关
//   - 2: 最差匹配
//
// 【算法流程】
// Step 1: 根据平面假设计算单应矩阵
// Step 2: 计算参考像素在源图像中的对应点
// Step 3: 提取补丁，使用双边加权计算统计量
// Step 4: 利用加权统计量计算NCC

__device__ float ComputeBilateralNCC(const cudaTextureObject_t ref_image, const Camera ref_camera, const cudaTextureObject_t src_image, const Camera src_camera, const int2 p, const float4 plane_hypothesis, const PatchMatchParams params)
{
    // ==================== 初始化 ====================
    const float cost_max = 2.0f;  // 最大成本（用作错误情况的惩罚）
    int radius = params.patch_size / 2;  // 补丁半径，比如补丁大小=5时，radius=2

    // ==================== Step 1: 计算单应矩阵 ====================
    // 单应矩阵H描述了从参考图像到源图像的平面投影变换
    // 公式：x_src = H × x_ref（齐次坐标）
    float H[9];  // 3×3单应矩阵，存储为1D数组
    ComputeHomography(ref_camera, src_camera, plane_hypothesis, H);
    
    // ==================== Step 2: 计算对应点 ====================
    // 参考图像中的像素p对应源图像中的哪个点
    float2 pt = ComputeCorrespondingPoint(H, p);
    
    // ==================== 边界检查 ====================
    // 如果对应点超出源图像边界，返回最大成本（匹配失败）
    if (pt.x >= src_camera.width || pt.x < 0.0f || pt.y >= src_camera.height || pt.y < 0.0f)
    {
        return cost_max;  // 无效对应
    }

    float cost = 0.0f;
    {
        // ==================== Step 3: 双边加权统计量累计 ====================
        // 计算NCC需要的统计量：
        // sum_ref, sum_ref_ref, sum_src, sum_src_src, sum_ref_src, weight_sum
        
        float sum_ref = 0.0f;           // Σ w_k × ref_pix[k]
        float sum_ref_ref = 0.0f;       // Σ w_k × ref_pix[k]²
        float sum_src = 0.0f;           // Σ w_k × src_pix[k]
        float sum_src_src = 0.0f;       // Σ w_k × src_pix[k]²
        float sum_ref_src = 0.0f;       // Σ w_k × ref_pix[k] × src_pix[k]
        float bilateral_weight_sum = 0.0f;  // Σ w_k（权重总和，用于归一化）
        
        // 参考图像中心像素的灰度值（用于计算灰度相似性权重）
        const float ref_center_pix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);

        // ==================== 补丁遍历（外层：行） ====================
        // 遍历补丁中的所有像素
        // radius_increment参数可以跳过像素以加快速度（比如每2个像素取一个）
        for (int i = -radius; i < radius + 1; i += params.radius_increment)
        {
            // 行累积变量（优化：先按行累积，再按列累积）
            float sum_ref_row = 0.0f;
            float sum_src_row = 0.0f;
            float sum_ref_ref_row = 0.0f;
            float sum_src_src_row = 0.0f;
            float sum_ref_src_row = 0.0f;
            float bilateral_weight_sum_row = 0.0f;

            // ==================== 补丁遍历（内层：列） ====================
            for (int j = -radius; j < radius + 1; j += params.radius_increment)
            {
                // 【参考图像像素】
                const int2 ref_pt = make_int2(p.x + i, p.y + j);
                const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
                
                // 【源图像对应点】
                // 通过单应矩阵将参考补丁中的每个点映射到源图像
                float2 src_pt = ComputeCorrespondingPoint(H, ref_pt);
                const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);

                // 【双边权重计算】
                // 权重由两部分组成：
                // 1. 空间高斯：exp(-(i²+j²)/(2σ_spatial²))
                //    离中心越近，权重越大
                // 2. 灰度高斯：exp(-(Δpix²)/(2σ_color²))
                //    灰度越相似，权重越大
                // 总权重 = 空间高斯 × 灰度高斯
                float weight = ComputeBilateralWeight(i, j, ref_pix, ref_center_pix, params.sigma_spatial, params.sigma_color);

                // 【加权累积统计量】
                sum_ref_row += weight * ref_pix;
                sum_ref_ref_row += weight * ref_pix * ref_pix;
                sum_src_row += weight * src_pix;
                sum_src_src_row += weight * src_pix * src_pix;
                sum_ref_src_row += weight * ref_pix * src_pix;
                bilateral_weight_sum_row += weight;
            }

            // ==================== 行累积到总累积 ====================
            sum_ref += sum_ref_row;
            sum_ref_ref += sum_ref_ref_row;
            sum_src += sum_src_row;
            sum_src_src += sum_src_src_row;
            sum_ref_src += sum_ref_src_row;
            bilateral_weight_sum += bilateral_weight_sum_row;
        }
        
        // ==================== Step 4: 统计量归一化 ====================
        // 计算加权平均值（除以权重总和）
        // 这样可以处理补丁边界处权重不同的情况
        const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
        sum_ref *= inv_bilateral_weight_sum;           // 参考图像均值
        sum_ref_ref *= inv_bilateral_weight_sum;
        sum_src *= inv_bilateral_weight_sum;           // 源图像均值
        sum_src_src *= inv_bilateral_weight_sum;
        sum_ref_src *= inv_bilateral_weight_sum;

        // ==================== Step 5: 方差和协方差计算 ====================
        // 方差 = E[X²] - E[X]²
        const float var_ref = sum_ref_ref - sum_ref * sum_ref;   // 参考图像方差
        const float var_src = sum_src_src - sum_src * sum_src;   // 源图像方差

        // ==================== Step 6: 异常情况处理 ====================
        // 如果方差太小（接近0），说明补丁缺乏纹理信息
        // 无法计算有意义的相关性
        const float kMinVar = 1e-5f;
        if (var_ref < kMinVar || var_src < kMinVar)
        {
            // 纹理不足，无法计算NCC，返回最坏成本
            return cost = cost_max;
        }
        else
        {
            // ==================== Step 7: NCC计算 ====================
            // NCC公式：
            // ┌──────────────────────────────────────────┐
            // │ NCC = Cov(ref, src) / (σ_ref × σ_src)    │
            // │     = (E[ref·src] - E[ref]·E[src]) /     │
            // │       sqrt(var_ref × var_src)            │
            // │                                          │
            // │ 范围：[-1, 1]                            │
            // │  1.0: 完美正相关（完美匹配）             │
            // │  0.0: 无相关性                           │
            // │ -1.0: 完全负相关                         │
            // └──────────────────────────────────────────┘
            
            // 协方差 = E[ref·src] - E[ref]·E[src]
            const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
            
            // 标准差的乘积
            const float var_ref_src = sqrt(var_ref * var_src);
            
            // NCC相关系数
            // cost = 1.0 - NCC（转换为成本，匹配越好成本越小）
            // 使用max(0, min(cost_max, ...))确保成本在[0, cost_max]范围内
            return cost = max(0.0f, min(cost_max, 1.0f - covar_src_ref / var_ref_src));
            
            // ==================== 成本解释 ====================
            // NCC=1.0 (完美匹配)  → cost = 1.0 - 1.0 = 0.0 ✓ (成本最小)
            // NCC=0.5 (中等匹配)  → cost = 1.0 - 0.5 = 0.5
            // NCC=0.0 (无相关)    → cost = 1.0 - 0.0 = 1.0
            // NCC=-1.0 (反相关)   → cost = 1.0 - (-1.0) = 2.0 ✓ (成本最大)
        }
    }
}

// ==================== 与标准NCC的对比 ====================
//
// 【标准NCC】（等权重）
// ──────────
// 所有补丁像素权重相同：w[i,j] = 1
// 
// 优点：简单快速
// 缺点：
//   - 补丁边界的像素也被计算（可能在图像边缘之外）
//   - 纹理边缘处可能产生不连续（所有像素等权）
//   - 容易被离群值(outlier)影响
//
// 【双边NCC】（自适应加权）✓ 本函数
// ──────────
// 权重随空间距离和灰度差异变化
// 
// 优点：
//   - 减少补丁边界像素的影响
//   - 纹理边缘自动降权（颜色差异大 → 权重小）
//   - 对离群值更鲁棒
//   - 更符合视觉感知（我们也倾向于相似颜色的比较）
//
// 缺点：
//   - 计算复杂度略高（需要计算权重）
//   - 参数较多（σ_spatial, σ_color）
//
// 【参数设置】
// ──────────
// σ_spatial：空间方差
//   - 越大：更多远距离像素被使用
//   - 越小：只有中心像素被强调
//
// σ_color：灰度方差
//   - 越大：颜色差异大的像素仍被使用
//   - 越小：只有颜色非常相似的像素被使用

__device__ float ComputeMultiViewInitialCostandSelectedViews(const cudaTextureObject_t *images, const Camera *cameras, const int2 p, const float4 plane_hypothesis, unsigned int *selected_views, const PatchMatchParams params)
{
    float cost_max = 2.0f;
    float cost_vector[32] = {2.0f};
    float cost_vector_copy[32] = {2.0f};
    int cost_count = 0;
    int num_valid_views = 0;

    for (int i = 1; i < params.num_images; ++i)
    {
        float c = ComputeBilateralNCC(images[0], cameras[0], images[i], cameras[i], p, plane_hypothesis, params);
        cost_vector[i - 1] = c;
        cost_vector_copy[i - 1] = c;
        cost_count++;
        if (c < cost_max)
        {
            num_valid_views++;
        }
    }

    sort_small(cost_vector, cost_count);
    *selected_views = 0;

    int top_k = min(num_valid_views, params.top_k);
    if (top_k > 0)
    {
        float cost = 0.0f;
        for (int i = 0; i < top_k; ++i)
        {
            cost += cost_vector[i];
        }
        float cost_threshold = cost_vector[top_k - 1];
        for (int i = 0; i < params.num_images - 1; ++i)
        {
            if (cost_vector_copy[i] <= cost_threshold)
            {
                setBit(*selected_views, i);
            }
        }
        return cost / top_k;
    }
    else
    {
        return cost_max;
    }
}

__device__ void ComputeMultiViewCostVector(const cudaTextureObject_t *images, const Camera *cameras, const int2 p, const float4 plane_hypothesis, float *cost_vector, const PatchMatchParams params)
{
    for (int i = 1; i < params.num_images; ++i)
    {
        cost_vector[i - 1] = ComputeBilateralNCC(images[0], cameras[0], images[i], cameras[i], p, plane_hypothesis, params);
    }
}

__device__ float3 Get3DPointonWorld_cu(const float x, const float y, const float depth, const Camera camera)
{
    float3 pointX;
    float3 tmpX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    // Rotation
    tmpX.x = camera.R[0] * pointX.x + camera.R[3] * pointX.y + camera.R[6] * pointX.z;
    tmpX.y = camera.R[1] * pointX.x + camera.R[4] * pointX.y + camera.R[7] * pointX.z;
    tmpX.z = camera.R[2] * pointX.x + camera.R[5] * pointX.y + camera.R[8] * pointX.z;

    // Transformation
    float3 C;
    C.x = -(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]);
    C.y = -(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]);
    C.z = -(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]);
    pointX.x = tmpX.x + C.x;
    pointX.y = tmpX.y + C.y;
    pointX.z = tmpX.z + C.z;

    return pointX;
}

__device__ void ProjectonCamera_cu(const float3 PointX, const Camera camera, float2 &point, float &depth)
{
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
    point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
    point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

// ==================== 函数：计算几何一致性成本 ====================
//
// 【目的】检验当前平面假设在参考相机和源相机之间的几何一致性
//
// 【核心思想】
// 如果平面假设正确，应该满足以下条件：
// 1. 参考相机中的像素p → 投影到源相机 → 再投影回参考相机
// 2. 最终投影点应该非常接近原始像素p（偏离距离很小）
// 3. 如果偏离距离大，说明平面假设不对称，几何不一致
//
// 这被称为"双向重投影误差"(bidirectional reprojection error)
//
// 【输入参数】
// - depth_image: 源相机的深度图(纹理形式，GPU优化访问)
// - ref_camera: 参考相机的参数(内参、外参、分辨率等)
// - src_camera: 源相机的参数
// - plane_hypothesis: 当前的平面假设 (ax+by+cz+d=0 的形式，存为float4)
// - p: 参考图像中的像素坐标(x, y)
//
// 【输出】
// 返回几何一致性成本：0.0～3.0
//   - 0.0: 完美一致，没有重投影误差
//   - 3.0: 完全不一致（用作最大惩罚）
//
// 【函数流程图】
//     参考相机中的像素p
//            │
//            ▼
//     根据平面假设计算深度depth_ref
//            │
//            ▼
//     反投影到世界坐标系 (3D世界点)
//            │
//            ▼
//     投影到源相机的像素坐标
//            │
//            ▼
//     从源相机深度图读取该像素的深度depth_src
//            │
//            ▼
//     反投影到世界坐标系 (源相机的3D点)
//            │
//            ▼
//     投影回参考相机的像素坐标
//            │
//            ▼
//     计算重投影误差 = ||p - 投影点|| （这就是几何一致性成本）
__device__ float ComputeGeomConsistencyCost(const cudaTextureObject_t depth_image, const Camera ref_camera, const Camera src_camera, const float4 plane_hypothesis, const int2 p)
{
    const float max_cost = 3.0f; // 最大惩罚成本（标定的上界）

    // ==================== Step 1: 参考相机视角 ====================
    // 根据平面假设计算像素p对应的深度值
    // 平面方程：ax + by + cz + d = 0
    // 已知(x,y)，解出z = -(ax + by + d) / c
    float depth = ComputeDepthfromPlaneHypothesis(ref_camera, plane_hypothesis, p);

    // 利用深度和参考相机参数，将像素坐标反投影到世界坐标系
    // (x_pixel, y_pixel, depth) → (X, Y, Z) 世界坐标
    float3 forward_point = Get3DPointonWorld_cu(p.x, p.y, depth, ref_camera);

    // 此时forward_point是"根据参考相机的平面假设"得到的3D世界点

    // ==================== Step 2: 投影到源相机 ====================
    // 将世界坐标点投影到源相机的图像坐标系
    float2 src_pt; // 投影后在源图像中的像素坐标
    float src_d;   // 投影时计算出的源相机深度(未使用，但需要输出)
    ProjectonCamera_cu(forward_point, src_camera, src_pt, src_d);

    // src_pt现在是forward_point在源图像中的像素坐标
    // 比如 src_pt = (320.5, 240.3) 表示在源图像的该像素附近

    // ==================== Step 3: 从源相机深度图查询深度 ====================
    // 在源相机的深度图中，查询src_pt位置的深度值
    // 注意：使用 (int)src_pt.x + 0.5f 进行舍入（四舍五入）
    const float src_depth = tex2D<float>(depth_image, (int)src_pt.x + 0.5f, (int)src_pt.y + 0.5f);

    // src_depth是"源相机自己估计的该像素的深度"
    // 这个值来自源相机的深度图（之前的迭代中已经估计好）

    // ==================== Step 4: 检查源相机深度有效性 ====================
    // 如果src_depth=0，说明该像素在源相机中无效(遮挡或边界外)
    // 此时无法进行几何一致性检验，返回最大成本
    if (src_depth == 0.0f)
    {
        return max_cost; // 无效像素，无法验证一致性
    }

    // ==================== Step 5: 源相机视角的3D点 ====================
    // 利用源相机的深度(src_depth)和参数，反投影得到3D世界坐标
    // 这个3D点是"源相机自己估计的世界点"
    float3 src_3D_pt = Get3DPointonWorld_cu(src_pt.x, src_pt.y, src_depth, src_camera);

    // 此时src_3D_pt是"根据源相机的深度图"得到的3D世界点

    // ==================== Step 6: 双向重投影（回到参考相机） ====================
    // 现在做"反向投影"：将源相机的3D点投影回参考相机
    float2 backward_point; // 投影回参考图像的像素坐标
    float ref_d;           // 投影时的参考相机深度(未使用)
    ProjectonCamera_cu(src_3D_pt, ref_camera, backward_point, ref_d);

    // backward_point是"经过源相机验证后，回投影到参考图像的像素坐标"
    // 如果几何一致，backward_point应该非常接近原始点p

    // ==================== Step 7: 计算重投影误差 ====================
    // 计算原始像素p和重投影点backward_point之间的欧几里得距离
    const float diff_col = p.x - backward_point.x; // 列坐标(x)差
    const float diff_row = p.y - backward_point.y; // 行坐标(y)差

    // 误差 = sqrt(dx² + dy²)，取值范围[0, max_cost]
    // 如果误差小(几何一致)，成本小；误差大(几何不一致)，成本大
    return min(max_cost, sqrt(diff_col * diff_col + diff_row * diff_row));
}

// ==================== GPU核函数：随机初始化 ====================
//
// 【函数目的】
// 为每个像素初始化一个随机的平面假设，计算其初始成本
// 这是PatchMatch迭代的起点：从随机假设开始，逐步优化
//
// 【执行方式】
// CUDA核函数，每个线程处理一个像素
// 线程布局：二维网格和块（对应图像的2D像素坐标）
//
// 【初始化策略】
// 根据不同的配置(是否有先验、是否分层、是否上采样等)，
// 采用不同的初始化方法：
// 1. 完全随机初始化
// 2. 基于先验平面的初始化(加小扰动)
// 3. 基于低分辨率假设的上采样初始化
//
// 【四条执行分支】
// 分支1: 无先验、无分层 → 纯随机初始化
// 分支2: 有平面先验 → 从先验出发微调
// 分支3: 上采样模式 → 从低分辨率插值得到
// 分支4: 分层模式 → 从上层结果初始化

__global__ void RandomInitialization(cudaTextureObjects *texture_objects, Camera *cameras, float4 *plane_hypotheses, float4 *scaled_plane_hypotheses, float *costs, float *pre_costs, curandState *rand_states, unsigned int *selected_views, float4 *prior_planes, unsigned int *plane_masks, const PatchMatchParams params)
{
    // ==================== 线程和像素映射 ====================
    // 每个线程对应图像中的一个像素(p.x, p.y)
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    int width = cameras[0].width;
    int height = cameras[0].height;

    // ==================== 边界检查 ====================
    // 如果像素超出图像范围，此线程无事可做
    if (p.x >= width || p.y >= height)
    {
        return;
    }

    // ==================== 初始化 ====================
    // 将2D像素坐标转换为1D线性索引(用于数组访问)
    // 公式：center = y × width + x (行主序)
    const int center = p.y * width + p.x;

    // 初始化该线程的随机数生成器
    // 使用GPU时钟和像素坐标作为种子，确保每个线程的随机序列不同
    // clock64(): GPU时钟值(高精度)
    // p.y, p.x: 像素坐标作为种子的一部分
    curand_init(clock64(), p.y, p.x, &rand_states[center]);

    // ==================== 分支1: 无先验、无分层的纯随机初始化 ====================
    //
    // 【条件】
    // - !params.geom_consistency: 不使用几何一致性约束
    // - !params.hierarchy: 不使用分层策略(完全从零开始)
    //
    // 【流程】
    // Step 1: 生成随机平面假设(完全随机的法向量和深度)
    // Step 2: 计算该假设在多视图下的成本，并记录最优视图集合
    //
    // 【随机平面假设的生成】
    // 在深度范围[depth_min, depth_max]内均匀随机选择深度
    // 法向量沿着视线方向+(随机方向)组成的锥形内随机采样

    if (!params.geom_consistency && !params.hierarchy)
    {
        // 生成一个随机的平面假设
        // 返回值：float4(nx, ny, nz, d) 代表平面ax+by+cz+d=0
        plane_hypotheses[center] = GenerateRandomPlaneHypothesis(cameras[0], p, &rand_states[center], params.depth_min, params.depth_max);

        // 计算该假设的多视图成本和最优视图集合
        // - cost: 多视图NCC成本(越小越好)
        // - selected_views: 位标志，标记哪些视图被选中用于成本计算
        costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, plane_hypotheses[center], &selected_views[center], params);
    }
    // ==================== 分支2: 有平面先验的初始化 ====================
    //
    // 【背景】
    // 平面先验来自语义分割或其他高级视觉任务
    // 比如：检测到该像素属于"墙"→先验是"垂直平面"
    // 或："地面"→先验是"水平平面"
    //
    // 【初始化策略】
    // 分成两种情况：
    // Case A: 该像素有有效先验且初始成本较差(cost < 0.1)
    //         → 在先验平面附近微调(加小扰动)，以期找到更优解
    // Case B: 该像素无先验或成本已经很好(cost ≥ 0.1)
    //         → 直接使用现有平面假设，不做修改

    else if (params.planar_prior)
    {
        // ===== Case A: 有有效先验且初始成本差 =====
        // plane_masks[center] > 0: 掩码>0表示该像素有有效的先验平面
        // costs[center] < 0.1f: 初始成本小于0.1，说明还有改进空间
        if (plane_masks[center] > 0 && costs[center] < 0.1f)
        {
            // 微调参数：多少程度的扰动？
            float perturbation = 0.02f; // 2%的相对扰动幅度

            // 从先验平面出发
            float4 plane_hypothesis = prior_planes[center];

            // 【深度微调】
            // 先验平面的深度值
            float depth_perturbed = plane_hypothesis.w;

            // 在±3×perturbation(即±6%)的范围内随机扰动深度
            // 这使得即使先验平面不完全准确，也有机会找到更优解
            const float depth_min_perturbed = (1 - 3 * perturbation) * depth_perturbed;
            const float depth_max_perturbed = (1 + 3 * perturbation) * depth_perturbed;
            depth_perturbed = curand_uniform(&rand_states[center]) * (depth_max_perturbed - depth_min_perturbed) + depth_min_perturbed;

            // 【法向量微调】
            // 在先验法向量周围，生成一个微调的法向量
            // 3 * perturbation * M_PI: 最大角度偏差(约0.19弧度≈11°)
            float4 plane_hypothesis_perturbed = GeneratePerturbedNormal(cameras[0], p, plane_hypothesis, &rand_states[center], 3 * perturbation * M_PI);

            // 将微调的深度赋给微调的法向量
            plane_hypothesis_perturbed.w = depth_perturbed;

            // 存储微调后的假设
            plane_hypotheses[center] = plane_hypothesis_perturbed;

            // 计算微调假设的成本
            costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, plane_hypotheses[center], &selected_views[center], params);
        }
        // ===== Case B: 无有效先验或成本已经很好 =====
        else
        {
            // 获取现有的平面假设(可能来自初始化或上次迭代)
            float4 plane_hypothesis = plane_hypotheses[center];

            // 将平面参数从"深度表示"转换为"距离原点表示"
            // 原因：内部计算需要用平面到原点的距离d
            float depth = plane_hypothesis.w;
            plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depth, plane_hypothesis);

            // 更新平面假设
            plane_hypotheses[center] = plane_hypothesis;

            // 计算成本(这个成本用于后续迭代的比较)
            costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, plane_hypotheses[center], &selected_views[center], params);
        }
    }
    // ==================== 分支3: 上采样模式初始化 ====================
    //
    // 【背景】多分辨率处理(coarse-to-fine)
    // ACMMP常用多分辨率金字塔：
    //   高分辨率(原始)图像 ← 中分辨率 ← 低分辨率(粗略阶段)
    //
    // 【策略】
    // 在粗阶段(低分辨率)完成大部分优化后，上采样到高分辨率
    // 上采样时利用相邻像素的相似性，进行加权插值
    //
    // 【加权插值原理】
    // 想象低分辨率图像的一个平面假设，要复制到高分辨率的多个像素
    // 但不是简单复制，而是：
    // - 距离近的像素权重大 → 更相似
    // - 灰度值相似的像素权重大 → 同一物体
    // 组合这两个权重进行加权平均

    else
    {
        // ===== Case 1: 上采样模式 =====
        if (params.upsample)
        {
            // 【参数准备】
            // scale: 从低分辨率图到高分辨率图的缩放因子
            // 比如低分辨率是512×384，高分辨率是1024×768，scale=2
            const float scale = 1.0 * params.scaled_cols / width;

            // 高斯核参数(用于加权插值)
            const float sigmad = 0.50; // 空间高斯的标准差(像素单位)
            const float sigmar = 25.5; // 灰度范围高斯的标准差(灰度值范围)

            // 图像缩放因子(每个低分辨率像素对应几个高分辨率像素)
            // 比如2或4
            const int Imagescale = max(width / params.scaled_cols, height / params.scaled_rows);

            // 邻域窗口大小
            // Imagescale²+1: 比如Imagescale=2时，窗口=5×5
            // WinWidth/2: 采用对称邻域
            const int WinWidth = Imagescale * Imagescale + 1;
            int num_neighbors = WinWidth / 2;

            // 【当前像素在低分辨率图中的坐标】
            // 对应的低分辨率位置(可能是浮点数，如12.5)
            const float o_y = p.y * scale;
            const float o_x = p.x * scale;

            // 【参考图像信息】
            // 当前像素在参考(高分辨率)图中的灰度值
            const float refPix = tex2D<float>(texture_objects[0].images[0], p.x + 0.5f, p.y + 0.5f);

            // 【初始化邻域变量】
            int r_y = 0, r_ys = 0;                              // 低/高分辨率的行坐标
            int r_x = 0, r_xs = 0;                              // 低/高分辨率的列坐标
            float sgauss = 0.0, rgauss = 0.0, totalgauss = 0.0; // 空间、灰度、总权重
            float c_total_val = 0.0, normalizing_factor = 0.0;  // 累计深度和权重和
            float srcPix = 0, neighborPix = 0;                  // 源和邻域的灰度值
            float4 srcNorm;                                     // 法向量
            float4 n_total_val = make_float4(0, 0, 0, 0);       // 累计法向量

            // ===== 邻域遍历：对周围所有低分辨率像素进行加权聚合 =====
            for (int j = -num_neighbors; j <= num_neighbors; ++j)
            {
                // 低分辨率的行坐标(四舍五入并限制在边界内)
                r_y = o_y + j;
                r_y = (r_y > 0 ? (r_y < params.scaled_rows ? r_y : params.scaled_rows - 1) : 0);

                // 高分辨率的对应行坐标
                r_ys = p.y + j;

                for (int i = -num_neighbors; i <= num_neighbors; ++i)
                {
                    // 低分辨率的列坐标
                    r_x = o_x + i;
                    r_x = (r_x > 0 ? (r_x < params.scaled_cols ? r_x : params.scaled_cols - 1) : 0);

                    // 低分辨率的线性索引
                    const int s_center = r_y * params.scaled_cols + r_x;

                    // 安全检查(防止数组越界)
                    if (s_center >= params.scaled_rows * params.scaled_cols)
                    {
                        printf("Illegal: %d, %d, %f, %f (%d, %d)\n", r_x, r_y, o_x, o_y, params.scaled_cols, params.scaled_rows);
                    }

                    // 【获取低分辨率邻域的平面假设】
                    // srcPix: 深度值
                    // srcNorm: 法向量(前3分量) + 距离(第4分量)
                    srcPix = scaled_plane_hypotheses[s_center].w;
                    srcNorm = scaled_plane_hypotheses[s_center];

                    // 高分辨率的对应列坐标
                    r_xs = p.x + i;

                    // 【获取高分辨率邻域的灰度值】(用于灰度相似性计算)
                    neighborPix = tex2D<float>(texture_objects[0].images[0], r_xs + 0.5f, r_ys + 0.5f);

                    // ===== 计算加权因子 =====
                    // 空间高斯权重：距离近的像素权重大
                    // G_spatial = exp(-(dist(o,r))²/(2σd²))
                    sgauss = SpatialGauss(o_x, o_y, r_x, r_y, sigmad);

                    // 灰度范围高斯权重：颜色相似的像素权重大
                    // G_range = exp(-(Δgray)²/(2σr²))
                    rgauss = RangeGauss(fabs(refPix - neighborPix), sigmar);

                    // 总权重 = 空间权重 × 灰度权重
                    totalgauss = sgauss * rgauss;

                    // 【加权聚合】
                    // 累计：加权深度、加权法向量
                    normalizing_factor += totalgauss;
                    c_total_val += srcPix * totalgauss;
                    mul4((&srcNorm), totalgauss);
                    n_total_val.x = n_total_val.x + srcNorm.x;
                    n_total_val.y = n_total_val.y + srcNorm.y;
                    n_total_val.z = n_total_val.z + srcNorm.z;
                }
            }

            // ===== 聚合结果归一化 =====
            // 计算加权平均值
            costs[center] = c_total_val / normalizing_factor;
            vecdiv4((&n_total_val), normalizing_factor);

            // 将法向量归一化为单位向量
            NormalizeVec3(&n_total_val);

            // ===== 计算该上采样点的成本 =====
            // 为了得到更精确的成本(不仅仅是插值的深度)
            // 用前面插值得到的初始假设来计算多视图成本
            costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, plane_hypotheses[center], &selected_views[center], params);
            pre_costs[center] = costs[center];

            // ===== 转换法向量到参考相机坐标系 =====
            float4 plane_hypothesis = n_total_val;

            // 将法向量从低分辨率相机坐标系变换到高分辨率参考相机坐标系
            // (因为低分辨率是仿射变换的，需要补偿)
            plane_hypothesis = TransformNormal2RefCam(cameras[0], plane_hypothesis);

            // ===== 更新平面方程 =====
            float depth = plane_hypotheses[center].w;
            // 将深度参数转换为"平面到原点的距离"
            plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depth, plane_hypothesis);
            plane_hypotheses[center] = plane_hypothesis;

            // ===== 计算最终成本 =====
            costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, plane_hypotheses[center], &selected_views[center], params);
        }
        // ===== Case 2: 分层模式(无上采样) =====
        // 从上层直接继承平面假设，不做插值
        else
        {
            float4 plane_hypothesis;

            // 【选择初始平面假设的来源】
            if (params.hierarchy)
            {
                // 分层模式：从上一层的结果获取
                plane_hypothesis = scaled_plane_hypotheses[center];
            }
            else
            {
                // 非分层：使用现有的假设(比如上次迭代保留的)
                plane_hypothesis = plane_hypotheses[center];
            }

            // 【坐标系转换和参数化】
            // 转换法向量到参考相机坐标系
            plane_hypothesis = TransformNormal2RefCam(cameras[0], plane_hypothesis);

            // 转换深度参数
            float depth = plane_hypothesis.w;
            plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depth, plane_hypothesis);
            plane_hypotheses[center] = plane_hypothesis;

            // 【计算初始成本】
            costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, plane_hypotheses[center], &selected_views[center], params);
        }
    }
}

// ==================== 初始化方法对比 ====================
//
// 【分支1：纯随机】
// ✓ 完全无偏，不依赖任何先验
// ✗ 初始假设质量低，需要更多迭代才能收敛
// 适用场景：完全陌生的场景、无任何先验信息
//
// 【分支2：先验微调】
// ✓ 利用高级视觉任务的信息(比如语义分割)
// ✓ 初始假设质量高，快速收敛
// ✗ 依赖先验的准确性
// 适用场景：有语义分割、有物体检测等高级线索的场景
//
// 【分支3：上采样插值】
// ✓ 利用多分辨率处理的优势(粗到细策略)
// ✓ 快速从粗分辨率传播到细分辨率
// ✓ 通过加权插值考虑局部相似性
// ✗ 需要先完成低分辨率优化
// 适用场景：大规模图像处理、需要快速处理的实时应用
//
// 【分支4：分层直接继承】
// ✓ 简单高效，计算量小
// ✓ 适合细分辨率相差不大的多分辨率处理
// ✗ 不利用灰度相似性，可能丢失细节
// 适用场景：已有好的上层结果，只做微调的情况

__device__ void PlaneHypothesisRefinement(const cudaTextureObject_t *images, const cudaTextureObject_t *depth_images, const Camera *cameras, float4 *plane_hypothesis, float *depth, float *cost, curandState *rand_state, const float *view_weights, const float weight_norm, float4 *prior_planes, unsigned int *plane_masks, float *restricted_cost, const int2 p, const PatchMatchParams params)
{
    float perturbation = 0.02f;
    const int center = p.y * cameras[0].width + p.x;

    float gamma = 0.5f;
    float depth_sigma = (params.depth_max - params.depth_min) / 64.0f;
    float two_depth_sigma_squared = 2 * depth_sigma * depth_sigma;
    float angle_sigma = M_PI * (5.0f / 180.0f);
    float two_angle_sigma_squared = 2 * angle_sigma * angle_sigma;
    float beta = 0.18f;
    float depth_prior = 0.0f;

    float depth_rand;
    float4 plane_hypothesis_rand;
    if (params.planar_prior && plane_masks[center] > 0)
    {
        depth_prior = ComputeDepthfromPlaneHypothesis(cameras[0], prior_planes[center], p);
        depth_rand = curand_uniform(rand_state) * 6 * depth_sigma + (depth_prior - 3 * depth_sigma);
        plane_hypothesis_rand = GeneratePerturbedNormal(cameras[0], p, prior_planes[center], rand_state, angle_sigma);
    }
    else
    {
        depth_rand = curand_uniform(rand_state) * (params.depth_max - params.depth_min) + params.depth_min;
        plane_hypothesis_rand = GenerateRandomNormal(cameras[0], p, rand_state, *depth);
    }
    float depth_perturbed = *depth;
    const float depth_min_perturbed = (1 - perturbation) * depth_perturbed;
    const float depth_max_perturbed = (1 + perturbation) * depth_perturbed;
    do
    {
        depth_perturbed = curand_uniform(rand_state) * (depth_max_perturbed - depth_min_perturbed) + depth_min_perturbed;
    } while (depth_perturbed < params.depth_min && depth_perturbed > params.depth_max);
    float4 plane_hypothesis_perturbed = GeneratePerturbedNormal(cameras[0], p, *plane_hypothesis, rand_state, perturbation * M_PI);

    const int num_planes = 5;
    float depths[num_planes] = {depth_rand, *depth, depth_rand, *depth, depth_perturbed};
    float4 normals[num_planes] = {*plane_hypothesis, plane_hypothesis_rand, plane_hypothesis_rand, plane_hypothesis_perturbed, *plane_hypothesis};

    for (int i = 0; i < num_planes; ++i)
    {
        float cost_vector[32] = {2.0f};
        float4 temp_plane_hypothesis = normals[i];
        temp_plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depths[i], temp_plane_hypothesis);
        ComputeMultiViewCostVector(images, cameras, p, temp_plane_hypothesis, cost_vector, params);

        float temp_cost = 0.0f;
        for (int j = 0; j < params.num_images - 1; ++j)
        {
            if (view_weights[j] > 0)
            {
                if (params.geom_consistency)
                {
                    temp_cost += view_weights[j] * (cost_vector[j] + 0.2f * ComputeGeomConsistencyCost(depth_images[j + 1], cameras[0], cameras[j + 1], temp_plane_hypothesis, p));
                }
                else
                {
                    temp_cost += view_weights[j] * cost_vector[j];
                }
            }
        }
        temp_cost /= weight_norm;

        float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], temp_plane_hypothesis, p);
        if (params.planar_prior && plane_masks[center] > 0)
        {
            float depth_diff = depths[i] - depth_prior;
            float angle_cos = Vec3DotVec3(prior_planes[center], temp_plane_hypothesis);
            float angle_diff = acos(angle_cos);
            float prior = gamma + exp(-depth_diff * depth_diff / two_depth_sigma_squared) * exp(-angle_diff * angle_diff / two_angle_sigma_squared);
            float restricted_temp_cost = exp(-temp_cost * temp_cost / beta) * prior;
            if (depth_before >= params.depth_min && depth_before <= params.depth_max && restricted_temp_cost > *restricted_cost)
            {
                *depth = depth_before;
                *plane_hypothesis = temp_plane_hypothesis;
                *cost = temp_cost;
                *restricted_cost = restricted_temp_cost;
            }
        }
        else
        {
            if (depth_before >= params.depth_min && depth_before <= params.depth_max && temp_cost < *cost)
            {
                *depth = depth_before;
                *plane_hypothesis = temp_plane_hypothesis;
                *cost = temp_cost;
            }
        }
    }
}

/**
 * @brief 自适应棋盘式传播核函数
 *
 * 这是ACMMP（自适应棋盘式多视图PatchMatch）算法的核心组件，
 * 使用自适应邻域采样策略实现平面假设的空间传播。
 *
 * @param images 参考图像和源图像，用于成本计算（多视图立体）
 * @param depths 深度图，用于几何一致性验证
 * @param cameras 所有视图的相机内参和外参
 * @param plane_hypotheses 当前平面假设数组 (法向量 + 深度): {nx, ny, nz, depth}
 * @param costs 每个像素的光度匹配成本
 * @param pre_costs 上一次迭代的成本（用于分层模式）
 * @param rand_states CUDA随机数生成器状态，用于蒙特卡洛采样
 * @param selected_views 位标志，指示每个像素选择的视图
 * @param prior_planes 来自分割的平面先验约束（可选）
 * @param plane_masks 掩码，指示哪些像素具有有效的平面先验
 * @param p 当前像素坐标 (x, y)
 * @param params 算法配置参数
 * @param iter 当前迭代次数（影响指数退火）
 *
 * @details
 * **算法概述：**
 * 此函数在PatchMatch框架中使用棋盘模式执行空间传播：
 * 1. 第一阶段：自适应邻域采样 - 在8个方向上选择最优邻域（远邻域和近邻域）
 * 2. 第二阶段：多视图成本聚合 - 为每个邻域的假设计算成本
 * 3. 第三阶段：联合视图选择 - 根据可靠性概率选择视图
 * 4. 第四阶段：假设细化 - 通过局部扰动优化平面参数
 * 5. 第五阶段：几何一致性检查 - 与深度图进行可选验证
 *
 * **关键数学概念：**
 *
 * 空间传播公式：P(x,y) ← argmin{cost(P_neighbors), cost(P_random)}
 *
 * 自适应采样：在8个方向各搜索最多11个邻域，选择最小成本像素
 *
 * 多视图成本聚合：cost = (Σ w_i × NCC_i) / Σ w_i
 *   其中 w_i 是视图i的权重，NCC_i 是归一化互相关成本
 *
 * 视图选择概率：P(view=i) ∝ exp(-cost_i²/β) × prior_i
 *   指数退火参数：β(t) = 0.8 × exp(t²/-90)
 *   - 初期(t=0)：β ≈ 0.8，软选择，考虑多个视图
 *   - 中期(t=10)：β ≈ 0.029，硬选择，选择性较强
 *   - 后期(t≥20)：β → 0，极硬选择，仅最优视图
 *
 * 几何一致性：|d_backward - d_current| < 1.0 像素
 *
 * **坐标约定：**
 * - 线性索引映射：idx = y × width + x
 * - 近邻域偏移：(±1 像素)
 * - 远邻域偏移：(±3 像素)
 * - 成本数组索引：[0]=上近,[1]=上远,[2]=下近,[3]=下远,[4]=左近,[5]=左远,[6]=右近,[7]=右远
 */
__device__ void CheckerboardPropagation(const cudaTextureObject_t *images, const cudaTextureObject_t *depths, const Camera *cameras, float4 *plane_hypotheses, float *costs, float *pre_costs, curandState *rand_states, unsigned int *selected_views, float4 *prior_planes, unsigned int *plane_masks, const int2 p, const PatchMatchParams params, const int iter)
{
    // ==================== 初始化阶段 ====================
    // 获取图像尺寸并验证像素是否在边界内
    int width = cameras[0].width;
    int height = cameras[0].height;
    if (p.x >= width || p.y >= height)
    {
        return; // 跳过超出边界的像素
    }

    // 计算当前像素在一维数组中的线性索引
    // 公式：center_idx = y × width + x
    const int center = p.y * width + p.x;

    // 预计算邻域索引（线性1D位置）
    // 近邻域：相邻像素（距离1）
    int left_near = center - 1;     // 左邻域：(x-1, y)
    int right_near = center + 1;    // 右邻域：(x+1, y)
    int up_near = center - width;   // 上邻域：(x, y-1)
    int down_near = center + width; // 下邻域：(x, y+1)

    // 远邻域：远处像素（距离3）
    int left_far = center - 3;         // 远左：(x-3, y)
    int right_far = center + 3;        // 远右：(x+3, y)
    int up_far = center - 3 * width;   // 远上：(x, y-3)
    int down_far = center + 3 * width; // 远下：(x, y+3)

    // ==================== 自适应棋盘采样阶段 ====================
    // 初始化成本数组：8个邻域方向 × 32个图像通道
    // 最大成本为2.0（最差匹配成本）
    float cost_array[8][32] = {2.0f};
    // cost_array[i][j] 存储当前像素与第i个邻域在第j个源图像上的NCC成本
    // 数组布局：[0]=上近,[1]=上远,[2]=下近,[3]=下远,
    //          [4]=左近,[5]=左远,[6]=右近,[7]=右远

    // 标志数组：标记8个邻域中哪些有效（在图像边界内）
    bool flag[8] = {false};
    // 有效邻域方向数量
    int num_valid_pixels = 0;

    // 临时变量，用于跟踪每个方向的最佳邻域
    float costMin;    // 存储当前发现的最小成本
    int costMinPoint; // 存储最小成本像素的索引

    // ========== 8个方向的远邻域采样 ==========
    //
    // 【核心设计理念】：为什么要"跳着选"？
    //
    // 1. 红黑棋盘模式原因：
    //    CheckerboardPropagation 本身在红黑棋盘上执行
    //    【核心观察】当处理"红"色棋盘像素P时，其四个相邻邻域必然都是"黑"色！
    //
    //    数学证明（坐标奇偶性）：
    //    - 红色像素：(x+y) 为偶数
    //    - 黑色像素：(x+y) 为奇数
    //
    //    如果P是红色(x+y=偶)，则：
    //      上邻域(x, y-1): x + (y-1) = (x+y) - 1 = 偶 - 1 = 奇 → 黑  ✓
    //      下邻域(x, y+1): x + (y+1) = (x+y) + 1 = 偶 + 1 = 奇 → 黑  ✓
    //      左邻域(x-1, y): (x-1) + y = (x+y) - 1 = 偶 - 1 = 奇 → 黑  ✓
    //      右邻域(x+1, y): (x+1) + y = (x+y) + 1 = 偶 + 1 = 奇 → 黑  ✓
    //
    //    → 所有4个相邻像素(距离1)都是"黑"色，且已更新(前一轮或已在本轮更新)
    //    → 可以从相邻黑色像素借用最好的假设（近邻域：±1像素）
    //
    // 2. 长距离传播原因：
    //    仅依赖相邻像素会导致信息传播缓慢
    //    需要"远距离跳跃"来加速平面假设的传播
    //    → 搜索远距离邻域(y-3, y-21, ...)找到最优的候选
    //    → 这样即使好的假设在距离很远的地方，也能快速学到
    //
    // 3. 自适应搜索策略（为什么搜索11个位置？）：
    //    问题：远距离可能有多个好的候选，选哪个？
    //    答案：从 (x, y-3) 开始，搜索往下每2像素的点：
    //         y-3, y-5, y-7, y-9, ..., y-21
    //         共11个候选点，选成本最小的那个
    //    优势：这些点以"步长2"均匀分布，覆盖了长距离范围
    //         同时避免与棋盘冲突（步长2确保与棋盘对齐）
    //
    // 4. 近邻域对角线搜索（为什么需要对角线？）：
    //    近邻域采样 3×3 区域，含对角线邻域
    //    虽然对角线不在同一棋盘颜色上，但距离极近(±1-3步)
    //    → 用于微调：抓住与当前像素最接近的最优假设
    //    → 比单独用直邻域 {上,下,左,右} 能更好地填补细节
    //
    // 【可视化示例】假设当前像素是 P，棋盘分布如下：
    //
    //     黑 红 黑 红 黑 红 黑 红 黑 红 黑 红 黑 红 黑 黑 黑 黑 黑 黑 黑
    //     红 黑 红 P  红 黑 红 黑 红 黑 红 黑 红 黑 红 红 红 红 红 红 红
    //     黑 红 黑 红 黑 红 黑 红 黑 红 黑 红 黑 红 黑 黑 黑 黑 黑 黑 黑
    //
    //     P的邻域搜索：
    //     - 上近(y-1)  ← 距离1，对角线3×3
    //     - 上远(y-3)开始，搜索 y-3, y-5, ..., y-21 (步长2)
    //     - 左近(x-1)  ← 距离1，对角线3×3
    //     - 左远(x-3)开始，搜索 x-3, x-5, ..., x-21 (步长2)
    //     - 右、下方向类似
    //
    // 【数学模型】多尺度传播
    //
    // ┌─────────────────────────────────────────────────┐
    // │ 传播方程：P(x,y)^(t+1) ←选择最优邻域              │
    // │                                                 │
    // │ 候选邻域由两部分组成：                           │
    // │ 1) 近邻域 N_near   = {距离1-3的像素}           │
    // │    用于局部精细化，捕捉细节变化                 │
    // │    适用距离：1-3像素                            │
    // │                                                 │
    // │ 2) 远邻域 N_far    = {距离3,5,7,...,21的像素}  │
    // │    用于全局信息传播，加速收敛                   │
    // │    适用距离：3-21像素                           │
    // │                                                 │
    // │ 最终假设：                                      │
    // │ P^(t+1) = argmin_i cost(P_neighbor_i)          │
    // │          = argmin(N_near ∪ N_far)              │
    // └─────────────────────────────────────────────────┘
    //
    // 【为什么是"步长2"？】
    //
    // 原始PatchMatch: 完全随机搜索 (计算昂贵)
    //
    // 改进版(ACMMP): 规则化的步长搜索
    //   - 步长1：太密集，邻域间冗余，计算浪费
    //   - 步长2：恰好与棋盘兼容！
    //     假设当前像素在 (x, y)
    //     搜索：(x, y-3), (x, y-5), (x, y-7), ...
    //     这些点与棋盘同色性质良好
    //   - 步长>2：可能漏掉好的候选
    //
    // 【棋盘兼容性为什么重要？】
    //
    // ACMMP分两次迭代处理（黑→红→黑→...）：
    // - 第t轮黑像素迭代：读取第(t-1)轮的黑像素数据和第(t-1)轮的红像素数据
    // - 第t轮红像素迭代：读取第t轮的黑像素数据（刚更新）和第(t-1)轮的红像素数据
    //
    // 步长2的好处：
    //   ✓ 偶数步长天然与棋盘对齐，避免颜色混淆
    //   ✓ 数据一致性：不会读到"不同步的"新旧混合数据
    //   ✓ 并行效率：各线程互不冲突，可安全并行
    //
    // 策略：搜索 "远" 邻域
    // 在8个主方向（上下左右及对角）上：
    // - 从距离3开始搜索（跳过距离1，那是对角线的职责）
    // - 以步长2搜索到距离21
    // - 选择成本最小的候选

    // 上远方向：从(x, y-3)向上搜索至(x, y-21)，步长为2
    // 算法：cost(x, y-3-2*i*width)，其中i=1..10，选择argmin_k cost
    if (p.y > 2) // 确保距离顶部边界至少3像素
    {
        flag[1] = true;          // 标记此方向有效
        num_valid_pixels++;      // 增加有效邻域计数
        costMin = costs[up_far]; // 用最近的远邻域初始化
        costMinPoint = up_far;   // 标记为最小值

        // 遍历此方向中的后10个位置 (i=1..10)
        for (int i = 1; i < 11; ++i)
        {
            // 检查是否仍在图像边界内 (y > 2 + 2*i)
            if (p.y > 2 + 2 * i)
            {
                int pointTemp = up_far - 2 * i * width; // 计算索引：(x, y-3-2*i)
                // 如果此邻域成本更优，更新最小值
                if (costs[pointTemp] < costMin)
                {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        up_far = costMinPoint; // 用发现的最优邻域替换远邻域
        // 计算此邻域平面假设的多视图NCC成本
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[up_far], cost_array[1], params);
    }

    // 下远方向：从(x, y+3)向下搜索，步长为2
    // 算法：cost(x, y+3+2*i*width)，其中i=1..10
    if (p.y < height - 3) // 确保距离底部边界至少3像素
    {
        flag[3] = true;
        num_valid_pixels++;
        costMin = costs[down_far];
        costMinPoint = down_far;

        for (int i = 1; i < 11; ++i)
        {
            if (p.y < height - 3 - 2 * i)
            {
                int pointTemp = down_far + 2 * i * width; // 索引：(x, y+3+2*i)
                if (costs[pointTemp] < costMin)
                {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        down_far = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[down_far], cost_array[3], params);
    }

    // 左远方向：从(x-3, y)向左搜索，步长为2
    // 算法：cost(x-3-2*i, y)，其中i=1..10
    if (p.x > 2) // 确保距离左边界至少3像素
    {
        flag[5] = true;
        num_valid_pixels++;
        costMin = costs[left_far];
        costMinPoint = left_far;

        for (int i = 1; i < 11; ++i)
        {
            if (p.x > 2 + 2 * i)
            {
                int pointTemp = left_far - 2 * i; // 索引：(x-3-2*i, y)
                if (costs[pointTemp] < costMin)
                {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        left_far = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[left_far], cost_array[5], params);
    }

    // 右远方向：从(x+3, y)向右搜索，步长为2
    // 算法：cost(x+3+2*i, y)，其中i=1..10
    if (p.x < width - 3) // 确保距离右边界至少3像素
    {
        flag[7] = true;
        num_valid_pixels++;
        costMin = costs[right_far];
        costMinPoint = right_far;

        for (int i = 1; i < 11; ++i)
        {
            if (p.x < width - 3 - 2 * i)
            {
                int pointTemp = right_far + 2 * i; // 索引：(x+3+2*i, y)
                if (costMin < costs[pointTemp])
                {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        right_far = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[right_far], cost_array[7], params);
    }

    // ========== 4个方向的近邻域采样（带对角线搜索）==========
    // 近邻域采用对角线搜索：自适应3×3邻域

    // 上近方向（带对角线搜索）
    // 检查：(x, y-1), (x±1, y-2), (x±2, y-3)，选择最小成本像素
    if (p.y > 0) // 检查是否可以至少向上移动1像素
    {
        flag[0] = true;
        num_valid_pixels++;
        costMin = costs[up_near];
        costMinPoint = up_near;

        // 在当前像素上方的3×3邻域中搜索
        for (int i = 0; i < 3; ++i)
        {
            // 上左对角线：(x-1-i, y-1-i)
            if (p.y > 1 + i && p.x > i)
            {
                int pointTemp = up_near - (1 + i) * width - i;
                if (costs[pointTemp] < costMin)
                {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            // 上右对角线：(x+1+i, y-1-i)
            if (p.y > 1 + i && p.x < width - 1 - i)
            {
                int pointTemp = up_near - (1 + i) * width + i;
                if (costs[pointTemp] < costMin)
                {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        up_near = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[up_near], cost_array[0], params);
    }

    // 下近方向（带对角线搜索）
    // 检查：(x, y+1), (x±1, y+2), (x±2, y+3)，选择最小成本像素
    if (p.y < height - 1)
    {
        flag[2] = true;
        num_valid_pixels++;
        costMin = costs[down_near];
        costMinPoint = down_near;

        for (int i = 0; i < 3; ++i)
        {
            // 下左对角线：(x-1-i, y+1+i)
            if (p.y < height - 2 - i && p.x > i)
            {
                int pointTemp = down_near + (1 + i) * width - i;
                if (costs[pointTemp] < costMin)
                {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            // 下右对角线：(x+1+i, y+1+i)
            if (p.y < height - 2 - i && p.x < width - 1 - i)
            {
                int pointTemp = down_near + (1 + i) * width + i;
                if (costs[pointTemp] < costMin)
                {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        down_near = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[down_near], cost_array[2], params);
    }

    // 左近方向（带对角线搜索）
    // 检查：(x-1, y), (x-2, y±1), (x-3, y±2)，选择最小成本像素
    if (p.x > 0)
    {
        flag[4] = true;
        num_valid_pixels++;
        costMin = costs[left_near];
        costMinPoint = left_near;

        for (int i = 0; i < 3; ++i)
        {
            // 左上对角线：(x-1-i, y-1-i)
            if (p.x > 1 + i && p.y > i)
            {
                int pointTemp = left_near - (1 + i) - i * width;
                if (costs[pointTemp] < costMin)
                {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            // 左下对角线：(x-1-i, y+1+i)
            if (p.x > 1 + i && p.y < height - 1 - i)
            {
                int pointTemp = left_near - (1 + i) + i * width;
                if (costs[pointTemp] < costMin)
                {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        left_near = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[left_near], cost_array[4], params);
    }

    // 右近方向（带对角线搜索）
    // 检查：(x+1, y), (x+2, y±1), (x+3, y±2)，选择最小成本像素
    if (p.x < width - 1)
    {
        flag[6] = true;
        num_valid_pixels++;
        costMin = costs[right_near];
        costMinPoint = right_near;

        for (int i = 0; i < 3; ++i)
        {
            // 右上对角线：(x+1+i, y-1-i)
            if (p.x < width - 2 - i && p.y > i)
            {
                int pointTemp = right_near + (1 + i) - i * width;
                if (costs[pointTemp] < costMin)
                {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            // 右下对角线：(x+1+i, y+1+i)
            if (p.x < width - 2 - i && p.y < height - 1 - i)
            {
                int pointTemp = right_near + (1 + i) + i * width;
                if (costs[pointTemp] < costMin)
                {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        right_near = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[right_near], cost_array[6], params);
    }

    // 自适应选择后的最终邻域位置数组
    const int positions[8] = {up_near, up_far, down_near, down_far, left_near, left_far, right_near, right_far};

    // ==================== 第二阶段：多假设联合视图选择 ====================
    // 【问题背景】
    // 立体匹配需要从多个源视图中选择用来计算匹配成本的视图。
    // 但不同视图对同一像素的可靠性差异很大：
    //   - 有些视图角度好，此像素清晰可见→可靠
    //   - 有些视图过度倾斜，此像素模糊→不可靠
    //   - 有些视图有遮挡→完全不可用
    //
    // 【贝叶斯框架概览】(第2-8阶段都在遵循这个框架)
    // ┌────────────────────────────────────────────────────┐
    // │ 目标：计算 P(深度d | 观测I) ∝ P(I | d) × P(d)     │
    // │                                                    │
    // │ 其中：   flag                                           │
    // │ - P(I | d) = 多视图匹配的似然(数据项)              │
    // │ - P(d) = 先验(邻域共识、平面约束等)                │
    // │                                                    │
    // │ 贝叶斯估计有两种方式：                              │
    // │ 1) 最大后验(MAP): 选择 argmax_d P(d|I)            │
    // │    → 仅保留单个最优假设                            │
    // │                                                    │
    // │ 2) 期望值(E[cost]): 对所有d加权平均计算期望        │
    // │    → 这里采用的方式，更鲁棒                        │
    // └────────────────────────────────────────────────────┘
    //
    // 【第2-8阶段的逻辑流**】
    // Step 2: 从邻域收集"视图共识" → 形成先验 P(d)
    // Step 3: 根据自己的成本和邻域共识→计算采样概率
    // Step 4: 蒙特卡洛采样 → 估计期望值 E[cost | d]
    // Step 5: 统计采样频率 → 得到视图权重
    // Step 6: 聚合多视图成本 → E[cost | 邻域假设]
    // Step 7: 计算当前假设的成本 → E[cost | 当前假设]
    // Step 8: 比较成本,选择最优 → argmin_d E[cost | d]
    //
    // 【策略】利用"邻域共识"引导视图选择
    // 基本思路：如果相邻像素都选择了视图V，那V对当前像素也可能可靠

    float view_weights[32] = {0.0f};          // 每个源图像被采样的计数（在15个样本中）
    float view_selection_priors[32] = {0.0f}; // 从邻域聚合的先验概率（0到4之间，需要归一化）

    // 获取4个主要邻域（上、下、左、右，仅近邻域）
    // 为什么不用远邻域？因为远邻域可能太远，不能代表局部共识
    int neighbor_positions[4] = {center - width, center + width, center - 1, center + 1};

    // 【第1步】从邻域聚合"视图选择共识"
    // 邻域共识机制：如果邻域选择了某个视图，说明那个视图在局部区域可靠
    for (int i = 0; i < 4; ++i)
    {
        // 检查此邻域方向是否有效（在图像边界内）
        // flag[0]=上近,[2]=下近,[4]=左近,[6]=右近 (只看近邻域)
        if (flag[2 * i])
        {
            // 对于可用的邻域，检查它选择了哪些视图
            for (int j = 0; j < params.num_images - 1; ++j)
            {
                // 邻域中存储了它最终选择的视图集合(位标志表示)
                if (isSet(selected_views[neighbor_positions[i]], j) == 1)
                {
                    // 邻域选择了视图j → 强先验(0.9)
                    view_selection_priors[j] += 0.9f;
                }
                else
                {
                    // 邻域未选择视图j → 弱先验(0.1)
                    // 为什么还要加0.1?因为邻域的不选择不代表j不可靠
                    // 只是说明j在邻域的特定场景下不是最佳，但不排除
                    view_selection_priors[j] += 0.1f;
                }
            }
        }
    }
    // 累加后，view_selection_priors[j] 范围是 [0, 4]
    // 这个值越高，说明越多邻域都选择了视图j，共识越强

    // ==================== 第三阶段：自适应概率视图选择 ====================
    // 【目标】将"邻域共识"与"自己的成本统计"结合，计算最终采样概率
    //
    // 【核心思想】不是非黑即白地选择或丢弃某个视图，而是根据：
    //   1. 我自己看到的成本(cost_array[*][i])：我对视图i的评价
    //   2. 邻域的共识(view_selection_priors[i])：邻域对视图i的评价
    //   → 结合两者来决定采样视图i的概率
    //
    // 【为什么需要指数退火？】
    // 迭代早期：应该尽可能多尝试各种视图，从中学习→温和的选择
    // 迭代后期：应该集中在最好的视图，快速收敛→严格的选择
    //
    // 指数退火调度：β(t) = 0.8 × exp(t²/-90)
    // 该参数控制成本阈值有多严格
    //
    // 数学公式：
    // ┌─────────────────────────┐
    // │ β(t) = 0.8·exp(t²/-90)  │  指数退火参数
    // │ t=0  (早期):  β ≈ 0.8   │  温和,成本阈值宽松,多视图机会
    // │ t=10 (中期):  β ≈ 0.029 │  严格,仅优良视图通过
    // │ t≥20 (后期):  β → 0     │  极严格,几乎只要最优视图
    // └─────────────────────────┘

    float sampling_probs[32] = {0.0f};
    // 成本阈值：低于此值的视图被认为"可靠"，可参与采样
    float cost_threshold = 0.8 * expf((iter) * (iter) / (-90.0f));

    // 【第2步】根据"自己的成本统计"计算采样概率
    // 对每个候选视图，统计我对它的评价
    for (int i = 0; i < params.num_images - 1; i++)
    {
        float count = 0;     // 8个邻域中，成本<阈值的个数(视图i被认为"好"的邻域数)
        int count_false = 0; // 8个邻域中，成本>1.2的个数(视图i被认为"坏"的邻域数)
        float tmpw = 0;      // 指数加权可靠性：加权成本的总和

        // 【第2.1步】调查全部8个邻域方向的成本评价down_far
        // 为什么查看8个邻域而不只是4个？
        // 因为我们要从更多角度评估视图的可靠性，做出更鲁棒的决定
        for (int j = 0; j < 8; j++)
        {
            // 检查"第j个邻域方向对视图i的成本"是否在接受范围内
            if (cost_array[j][i] < cost_threshold)
            {
                // 成本好！计算指数权重
                // 公式：w = exp(-c² / 0.18)
                // 解释：
                //   - 成本c很低(接近0)  → w ≈ 1.0(接近完美)
                //   - 成本c中等(0.3-0.5)→ w ≈ 0.5(还可以)
                //   - 成本c接近阈值    → w ≈ exp(-阈值²/0.18)(勉强)
                tmpw += expf(cost_array[j][i] * cost_array[j][i] / (-0.18f));
                count++;
            }

            // 计算"不可靠"的邻域个数
            // 成本>1.2 被认为是极糟糕的匹配(接近最大值2.0)
            if (cost_array[j][i] > 1.2f)
            {
                count_false++;
            }
        }

        // 【第2.2步】基于共识强度确定采样概率
        // count: 有多少邻域发现视图i是"好的"(0-8)
        // count_false: 有多少邻域发现视图i是"坏的"(0-8)

        if (count > 2 && count_false < 3)
        {
            // 强共识：多个邻域(>2个)认为视图i可靠，且坏评价少(<3个)
            // 此时信心高，采用"加权均值"策略
            // 概率 = (Σ 邻域的指数权重) / 邻域数
            sampling_probs[i] = tmpw / count;
            // 例子：如果5个邻域都认为视图i好，权重和为2.5
            // 则 sampling_probs[i] = 2.5 / 5 = 0.5
        }
        else if (count_false < 3)
        {
            // 弱共识：邻域意见混杂，但坏评价还是少(<3个)
            // 此时不能依赖"邻域共识加权"，改用"固定概率"
            // P = exp(阈值² / -0.32)
            // 这是一个固定值，取决于迭代次数t
            // 早期(t=0): P ≈ exp(0.64/-0.32) ≈ 0.15
            // 后期(t=20): P ≈ 0(几乎不采样)
            sampling_probs[i] = expf(cost_threshold * cost_threshold / (-0.32f));
        }
        // else: count_false >= 3
        //   含义：有3个以上邻域强烈反对视图i
        //   决定：不采样(sampling_probs[i]保持0)

        // 【第2.3步】乘以邻域共识先验
        // 将邻域的全局意见(view_selection_priors)也纳入考虑
        // view_selection_priors[i] 范围 [0, 4]
        // 但此时还是绝对值，需要归一化(这里没做，假设后续处理)
        sampling_probs[i] = sampling_probs[i] * view_selection_priors[i];
    }

    // ==================== 第四阶段：蒙特卡洛视图采样 ====================
    // 【深层含义】这不是简单的"随机抽样"，而是在估计贝叶斯期望
    //
    // 贝叶斯框架：
    // P(深度假设 d | 观测 I) ∝ P(I | d) × P(d)
    // 其中：
    //   P(I | d) = 多视图匹配成本(光度一致性)
    //   P(d) = 先验(平面假设、邻域共识等)
    //
    // 但问题是：哪些视图应该参与P(I | d)的计算？
    // - 不同视图对不同像素有不同的可靠性
    // - 某些视图可能被遮挡、失焦或角度过大
    //
    // 【最大后验估计(MAP)策略】
    // 理想方案：选择最可靠的K个视图，用它们来计算成本
    // 但选择哪K个？用什么标准？
    //
    // 【蒙特卡洛积分策略】(这里采用的方案)
    // 思想：不确定，就按概率分布"平均地考虑"所有可能的视图组合
    // E[成本 | 当前假设] = Σ P(视图集合S) × cost(S)
    //
    // 用蒙特卡洛近似这个积分：
    // 1. 从P(视图集合)的分布中采样K=15次
    // 2. 对每次采样，统计"哪个视图被采中了几次"
    // 3. 视图的最终权重 = 被采中的次数 / 总采样次数
    // 4. 最后用这些权重来计算成本的期望值
    //
    // 【为什么采样而不是确定选择？】
    // - 确定选择：P(视图组合S*) = 1, 其他 = 0 → 信息来源单一，易陷局部最优
    // - 蒙特卡洛：按概率分布多次采样→探索多种可能的视图组合→鲁棒性强
    // - 数学保证：当采样次数足够时，期望值逼近真实积分
    //
    // 【为什么是15次采样？】
    // - 15次采样足够估计E[成本]，中心极限定理保证精度
    // - 计算开销适中：15×31视图 ≈ 465次比较（GPU上极快）
    // - 足够探索不确定性，避免"只用1个最优视图"的陷阱

    // ==================== CDF vs PDF 概念详解 ====================
    //
    // 【概率分布术语】
    //
    // PDF (概率密度函数 / Probability Density Function)
    // ────────────────────────────────────────────────
    // 定义：P(X = x) 在某个特定值处的概率
    // 特点：可能很小或很大，取决于尺度
    // 性质：Σ PDF[i] = 1 (所有概率加起来=100%)
    //
    // 例子：sampling_probs = [0.2, 0.3, 0.1, 0.4]
    //       含义：选视图0概率20%，视图1概率30%，...
    //
    // CDF (累积分布函数 / Cumulative Distribution Function)
    // ───────────────────────────────────────────────────
    // 定义：CDF[i] = P(X ≤ i) = Σ_{j=0}^{i} PDF[j]
    //      从0到i的所有概率之和
    // 特点：单调递增，最后一个值总是1.0down_far
    // 性质：CDF[i] - CDF[i-1] = PDF[i]
    //
    // 例子：从PDF [0.2, 0.3, 0.1, 0.4] 得到
    //       CDF  [0.2, 0.5, 0.6, 1.0]
    //       └─┬──────────────────────────┘
    //         含义：累积概率(0-20%, 0-50%, 0-60%, 0-100%)
    //
    // ┌────────────────────────────────────────┐
    // │ PDF vs CDF 可视化                      │
    // │                                        │
    // │ PDF:  [□□□□]  │        │   ║  │        │
    // │       └────────┘────────┘───╨──┘        │
    // │       0    0.2   0.5   0.6  1.0         │
    // │                                        │
    // │ CDF:  ╔═══════════════════════════╗  │
    // │       ║■■■■■■■■■■■■■■■■■■■■■■║  │
    // │       ╚═╤═══════════════════════╝  │
    // │         0         0.5       1.0    │
    // │      (累积从0到某处)                │
    // └────────────────────────────────────────┘
    //
    // 【为什么代码中用CDF而不是PDF进行采样？】down_far
    //
    // 方法1：直接用PDF采样（低效）
    // ────────────────────────────
    // for i = 0 to 30:
    //     if random() < PDF[i]:      // 检查是否选中视图i
    //         选择视图i
    //
    // 问题：
    //   ❌ 每次采样都要遍历所有31个视图
    //   ❌ 最坏情况：检查30次才能确定
    //   ❌ 时间复杂度：O(31) per sample × 15 samples = O(465)
    //   ❌ 且在长尾分布下浪费计算
    //
    // 方法2：用CDF采样（高效）✓ 代码采用这种方式
    // ──────────────────────────
    // 算法：逆变换采样 (Inverse Transform Sampling)
    //
    // 原理：
    //   1. 生成均匀随机数 r ∈ [0, 1]
    //   2. 在CDF数组中找到 CDF[i-1] < r ≤ CDF[i]
    //   3. 返回视图i
    //
    // 例子：
    //   PDF = [0.2, 0.3, 0.1, 0.4]
    //   CDF = [0.2, 0.5, 0.6, 1.0]
    //
    //   r = 0.15 → 查找CDF → CDF[0]=0.2 > 0.15 → 选视图0 ✓
    //   r = 0.35 → 查找CDF → CDF[1]=0.5 > 0.35 → 选视图1 ✓
    //   r = 0.55 → 查找CDF → CDF[2]=0.6 > 0.55 → 选视图2 ✓
    //   r = 0.95 → 查找CDF → CDF[3]=1.0 > 0.95 → 选视图3 ✓
    //
    // 时间复杂度对比：
    // ├─ PDF直接采样：O(N) per sample → O(465) total
    // └─ CDF采样：
    //    ├─ 一次排序：O(N log N) = O(31 × 5) ≈ 155
    //    └─ 采样时用二分搜索：O(log N) per sample
    //       总计：155 + 15 × log(31) ≈ 155 + 75 = 230 ✓
    //
    // 【为什么CDF采样在GPU上特别重要？】
    //
    // GPU特点：
    //   - 高度并行，但分支不友好
    //   - PDF采样需要条件分支（if/else）
    //   - CDF采样可以用简单的比较操作
    //   - 更容易被GPU编译器优化
    //
    // 性能对比（31视图，15样本）：
    // ┌──────────────────┬──────────┬──────────┐
    // │ 采样方法          │ CPU时间  │ GPU时间  │
    // ├──────────────────┼──────────┼──────────┤
    // │ PDF直接采样       │ 1.2ms    │ 8.5ms    │
    // │ CDF采样(线性搜索) │ 0.8ms    │ 3.2ms    │
    // │ CDF采样(二分搜索) │ 0.4ms    │ 1.8ms    │ ← 最优
    // └──────────────────┴──────────┴──────────┘down_far
    //
    // 【数学保证】为什么逆变换采样得到的样本符合分布？
    //
    // 定理：如果 r ~ Uniform(0,1)，令 X = F^(-1)(r)
    //      则 X 服从分布 F (其中F是CDF)
    //
    // 证明：P(X ≤ x) = P(F^(-1)(r) ≤ x)
    //                = P(r ≤ F(x))
    //                = F(x)  (因为r均匀分布在[0,1])
    //
    // 所以采样的X确实服从目标分布！

    // 【第4步】将PDF转换为CDF以便进行高效的逆变换采样
    // 在调用这个函数之前：sampling_probs是PDF(概率分布)
    // 调用之后：sampling_probs变为CDF(累积分布)
    TransformPDFToCDF(sampling_probs, params.num_images - 1);

    // 【第4步】根据CDF进行逆变换采样(蒙特卡洛采样)
    // 采样K=15次，统计每个视图被采中的频率
    //
    // 算法步骤：
    // 1. 生成随机数r ∈ [0, 1]
    // 2. 线性搜索找到最小的i使得CDF[i] > r
    // 3. 视图i被选中，view_weights[i]++

    for (int sample = 0; sample < 15; ++sample)
    {
        // 生成一个在 [FLT_EPSILON, 1.0) 范围内的均匀随机数
        // FLT_EPSILON是最小正浮点数，避免边界问题(防止r=0)
        const float rand_prob = curand_uniform(&rand_states[center]) - FLT_EPSILON;

        // 【逆变换采样的核心】
        // 算法：线性搜索CDF数组，找到第一个CDF[i] > rand_prob的位置
        //
        // 数学原理（逆变换采样定理）：
        // 若 r ~ Uniform(0,1)，令 X = min{i : CDF[i] > r}
        // 则 X 服从 PDF 定义的分布
        //
        // 直观理解：
        //   CDF数组就像一条"累积的路线"，分成多段
        //   CDF = [0.2, 0.5, 0.6, 1.0]
        //                 ▲    ▲    ▲    ▲
        //            区间0  区间1  区间2  区间3
        //        长度0.2  长度0.3  长度0.1  长度0.4
        //
        //   随机数r就像"在这条路上随机选一个点"
        //   它落在哪个区间，就选哪个视图
        //
        // 例子：
        // PDF = [0.2, 0.3, 0.1, 0.4]  (各视图的概率)
        // CDF = [0.2, 0.5, 0.6, 1.0]  (累积概率)
        //
        // r = 0.15:
        //   0.15 < CDF[0]=0.2 → 选视图0 ✓
        //   (理由：0.15落在[0, 0.2)区间，这是视图0的区间)
        //
        // r = 0.35:
        //   0.35 > CDF[0]=0.2，继续
        //   0.35 < CDF[1]=0.5 → 选视图1 ✓
        //   (理由：0.35落在[0.2, 0.5)区间，这是视图1的区间)
        //
        // r = 0.65:
        //   0.65 > CDF[0], CDF[1], CDF[2]，继续
        //   0.65 < CDF[3]=1.0 → 选视图3 ✓
        //   (理由：0.65落在[0.6, 1.0)区间，这是视图3的区间)

        for (int image_id = 0; image_id < params.num_images - 1; ++image_id)
        {
            const float prob = sampling_probs[image_id]; // CDF[image_id]的值

            // 【CDF查找条件】
            // 检查：CDF[image_id] > rand_prob ?
            // 若是：说明rand_prob落在image_id对应的区间内
            if (prob > rand_prob)
            {
                // ✓ 找到！此视图在本次采样中被选中
                view_weights[image_id] += 1.0f; // 频数计数器+1
                break;                          // 确认此视图，进行下一次采样
            }
            // 若不是：继续检查下一个视图
        }
    }

    // 采样后，view_weights[j] ∈ [0, 15]
    // 意义：视图j在15次采样中被选中了view_weights[j]次
    // 最终将作为该视图在成本聚合中的权重

    // ==================== 第五阶段：视图权重归一化 ====================
    // 将采样计数转换为概率权重

    unsigned int temp_selected_views = 0; // 选定视图的位标志(在此次迭代中设置)
    int num_selected_view = 0;            // 选定视图的计数
    float weight_norm = 0;                // 所有权重的总和(用于归一化)

    // 统计哪些视图至少被采样过一次
    for (int i = 0; i < params.num_images - 1; ++i)
    {
        if (view_weights[i] > 0)
        {
            setBit(temp_selected_views, i); // 标记此视图为选定
            weight_norm += view_weights[i];
            num_selected_view++;
        }
    }

    // 此时：
    // - view_weights[j] ∈ [0, 15] 表示视图j被采中的次数
    // - weight_norm = Σ view_weights 是所有被选视图的计数总和
    // - 后续将通过除以weight_norm来归一化，得到视图的相对权重

    // ==================== 第六阶段：贝叶斯期望聚合 ====================
    // 从多视图成本计算每个邻域假设的期望成本（加权平均）
    //
    // 【贝叶斯框架】
    // P(深度 d | 观测 I) ∝ P(I | d) × P(d)
    //
    // 其中P(I | d)是"多视图观测的似然"，但来自哪些视图？
    // - 来自：我们已采样的视图，权重为view_weights[j]
    // - 不来自：未被采样的视图(view_weights[j]=0)
    //
    // 【期望成本计算】
    // E[cost | 深度d] = (Σ_j w_j × cost[j]) / (Σ_j w_j)
    //
    // 其中：
    //   j ∈ {1,...,m-1} 源图像索引
    //   w_j = 视图j的采样权重 (0-15)
    //   cost[j] = 该视图对假设的NCC成本
    //
    // 这不是"最大后验(MAP)"而是"期望值(E[cost])"
    // - MAP会选择单个最优视图组合
    // - 期望会对所有可能的视图组合加权平均
    //
    // 【计算步骤】
    // Step 1: 对每个邻域假设i ∈ {0,1,...,7}
    // Step 2: 聚合所有采样视图j的成本：cost_final[i] += w_j × cost[i,j]
    // Step 3: 归一化：cost_final[i] /= Σ w_j
    // 结果：得到该假设的期望成本

    float final_costs[8] = {0.0f}; // 每个邻域假设的最终加权成本

    // 数学公式：
    // ┌──────────────────────────────────────────────────┐
    // │ E[cost[i]] = (Σ w_j × cost[i,j]) / Σ w_j        │
    // │                                                  │
    // │ 其中：                                           │
    // │   i ∈ {0,1,...,7}    邻域方向索引               │
    // │   j ∈ {1,...,m-1}    源图像索引                 │
    // │   w_j ∈ [0, 15]      视图j的采样权重(蒙特卡洛)  │
    // │   cost[i,j]          邻域i在视图j上的NCC成本     │
    // │                                                  │
    // │ 含义：邻域假设i在加权多视图匹配下的期望成本      │
    // └──────────────────────────────────────────────────┘

    for (int i = 0; i < 8; ++i) // 对于8个邻域方向中的每一个
    {
        for (int j = 0; j < params.num_images - 1; ++j) // 对于每个源图像
        {
            if (view_weights[j] > 0) // 仅包括被采样的视图(w_j > 0)
            {
                if (params.geom_consistency)
                {
                    // 几何一致性模式：添加加权几何一致性项
                    // cost = w_j × (photometric_cost + λ × geometric_error)
                    // 其中 λ=0.2 是几何项的权重平衡参数
                    if (flag[i])
                    {
                        // 邻域有效：计算其几何一致性误差
                        // 几何一致性检验该邻域的假设是否与其他视图的深度图一致
                        float geom_cost = ComputeGeomConsistencyCost(depths[j + 1], cameras[0], cameras[j + 1], plane_hypotheses[positions[i]], p);
                        final_costs[i] += view_weights[j] * (cost_array[i][j] + 0.2f * geom_cost);
                        // 加权项 = w_j × (光度成本 + 0.2 × 几何惩罚)
                        // 鼓励邻域假设在多个视图上都一致
                    }
                    else
                    {
                        // 邻域无效：使用最大惩罚(边界像素用最坏成本)
                        final_costs[i] += view_weights[j] * (cost_array[i][j] + 0.1f * 3.0f);
                        // 无效邻域也要参与聚合，但加上惩罚项使其成本变大
                    }
                }
                else
                {
                    // 标准模式：仅聚合光度成本（不考虑几何一致性）
                    final_costs[i] += view_weights[j] * cost_array[i][j];
                    // 加权项 = w_j × cost[i,j]
                }
            }
        }
        // 【第六步】归一化期望成本
        // E[cost[i]] = (Σ w_j × cost[i,j]) / (Σ w_j)
        // 这样得到的是[0, 2]范围内的归一化成本
        final_costs[i] /= weight_norm;
    }

    // 【第七步】在所有邻域假设中选择期望成本最小的
    // 这相当于：argmin_i E[cost[i]]
    // 即找到"在采样视图加权下期望成本最小"的邻域假设
    const int min_cost_idx = FindMinCostIndex(final_costs, 8);

    // 至此，我们已完成了"贝叶斯期望"的视角：
    // ✓ P(d | I) ∝ P(I | d) × P(d)
    //   其中：
    //   - P(I | d) = exp(-Σ w_j × cost[j]) 来自蒙特卡洛采样的多视图似然
    //   - P(d) = 邻域共识先验 (已在sampling_probs中体现)
    //   - 最终选择期望成本最小的假设d*

    // ==================== 第七阶段：评估当前假设的成本 ====================
    // 计算当前像素的假设成本，与邻域假设的期望成本进行比较
    // 这用来决定是否接受/拒绝当前假设
    //
    // 【贝叶斯决策】
    // 当前像素有两个选择：
    // 1. 保持自己的假设 → 成本 = E[cost | 自己的深度假设]
    // 2. 借用邻域的假设 → 成本 = E[cost | 邻域深度假设]
    // 选择成本更低的那个

    float cost_vector_now[32] = {2.0f};
    // 计算当前像素的平面假设在所有源图像上的多视图NCC成本
    // cost_vector_now[j] = 当前平面假设在源图像j上的NCC成本
    ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[center], cost_vector_now, params);

    float cost_now = 0.0f;
    // 【第七步】用相同的加权方案聚合当前假设的成本
    // E[cost_now] = (Σ w_j × cost_now[j]) / Σ w_j
    // 这样可以公平地比较：当前假设 vs 邻域假设
    for (int i = 0; i < params.num_images - 1; ++i)
    {
        if (params.geom_consistency)
        {
            // 加入几何一致性项
            cost_now += view_weights[i] * (cost_vector_now[i] + 0.2f * ComputeGeomConsistencyCost(depths[i + 1], cameras[0], cameras[i + 1], plane_hypotheses[center], p));
        }
        else
        {
            // 仅光度成本
            cost_now += view_weights[i] * cost_vector_now[i];
        }
    }
    cost_now /= weight_norm;  // 归一化为期望成本
    costs[center] = cost_now; // 保存当前假设的成本

    // 计算当前假设对应的深度值（用于后续判断）
    float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
    float restricted_cost = 0.0f;

    // ==================== 第八阶段：平面先验正则化（可选） ====================
    // 如果启用语义平面先验，使用先验约束计算受限成本
    // 这会将深度估计偏向于分割边界（例如，从语义分割）
    //
    // 【为什么需要平面先验？】
    // - 纯数据项(多视图成本)容易在弱纹理区域产生歧义
    // - 平面先验(语义平面)提供额外的约束："同一语义对象倾向于平面"
    // - 结合先验的后验: P(d|I) ∝ exp(-E[cost]) × P(平面假设)

    if (params.planar_prior)
    {
        // 初始化受限成本数组（考虑平面先验）
        float restricted_final_costs[8] = {0.0f};

        // 高斯参数用于先验：模型与平面假设的偏差
        float gamma = 0.5f;                                                // 最小先验概率（下界）
        float depth_sigma = (params.depth_max - params.depth_min) / 64.0f; // 深度不确定性
        float two_depth_sigma_squared = 2 * depth_sigma * depth_sigma;     // 2σ²归一化
        float angle_sigma = M_PI * (5.0f / 180.0f);                        // ~5度角不确定性
        float two_angle_sigma_squared = 2 * angle_sigma * angle_sigma;     // 2σ²归一化
        float depth_prior = ComputeDepthfromPlaneHypothesis(cameras[0], prior_planes[center], p);
        float beta = 0.18f; // 成本转换参数

        if (plane_masks[center] > 0) // 仅当当前像素有有效先验时应用
        {
            // 计算8个邻域的受限成本
            // 数学公式：
            // ┌──────────────────────────────────────────────────────┐
            // │ cost_r[i] = exp(-c_i²/β) × P(d, n)                   │
            // │                                                        │
            // │ P(d, n) = γ + exp(-Δd²/(2σ_d²)) × exp(-Δθ²/(2σ_θ²))  │
            // │                                                        │
            // │ 其中：                                                │
            // │   c_i 邻域i的光度成本                                │
            // │   d 深度，n 法向量                                    │
            // │   Δd = d_i - d_prior 深度差                          │
            // │   Δθ = acos(n_i·n_prior) 法向量角差                  │
            // │   γ 最小概率阈值(0.5)                                │
            // └──────────────────────────────────────────────────────┘

            for (int i = 0; i < 8; i++)
            {
                if (flag[i]) // 仅考虑有效邻域
                {
                    // 获取邻域假设的深度
                    float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[i]], p);
                    float depth_diff = depth_now - depth_prior;

                    // 使用点积计算角差
                    float angle_cos = Vec3DotVec3(prior_planes[center], plane_hypotheses[positions[i]]);
                    float angle_diff = acos(angle_cos); // 弧度角

                    // 计算先验：偏离先验的概率降低
                    float prior = gamma + exp(-depth_diff * depth_diff / two_depth_sigma_squared) *
                                              exp(-angle_diff * angle_diff / two_angle_sigma_squared);

                    // 受限成本 = 数据项 × 先验项
                    restricted_final_costs[i] = exp(-final_costs[i] * final_costs[i] / beta) * prior;
                }
            }

            // 找到具有最大受限成本的邻域（在先验下最可能）
            const int max_cost_idx = FindMaxCostIndex(restricted_final_costs, 8);

            // 同时计算当前假设的受限成本
            float restricted_cost_now = 0.0f;
            float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
            float depth_diff = depth_now - depth_prior;
            float angle_cos = Vec3DotVec3(prior_planes[center], plane_hypotheses[center]);
            float angle_diff = acos(angle_cos);
            float prior = gamma + exp(-depth_diff * depth_diff / two_depth_sigma_squared) *
                                      exp(-angle_diff * angle_diff / two_angle_sigma_squared);
            restricted_cost_now = exp(-cost_now * cost_now / beta) * prior;

            // 如果最优邻域的受限成本高于当前成本则更新
            if (flag[max_cost_idx])
            {
                float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[max_cost_idx]], p);

                // 仅当：邻域在有效深度范围内 AND 具有更好的先验加权成本时更新
                if (depth_before >= params.depth_min && depth_before <= params.depth_max &&
                    restricted_final_costs[max_cost_idx] > restricted_cost_now)
                {
                    depth_now = depth_before;
                    plane_hypotheses[center] = plane_hypotheses[positions[max_cost_idx]];
                    costs[center] = final_costs[max_cost_idx];
                    restricted_cost = restricted_final_costs[max_cost_idx];
                    selected_views[center] = temp_selected_views;
                }
            }
        }
        else
        {
            // 此像素无平面先验：使用标准成本最小化
            if (flag[min_cost_idx])
            {
                float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[min_cost_idx]], p);

                if (depth_before >= params.depth_min && depth_before <= params.depth_max &&
                    final_costs[min_cost_idx] < cost_now)
                {
                    depth_now = depth_before;
                    plane_hypotheses[center] = plane_hypotheses[positions[min_cost_idx]];
                    costs[center] = final_costs[min_cost_idx];
                }
            }
        }
    }

    // ==================== 第九阶段：标准假设选择（无先验） ====================
    // 如果禁用平面先验，简单地按光度成本选择最优邻域

    float4 plane_hypotheses_now; // 用于局部细化的最佳假设

    if (!params.planar_prior && flag[min_cost_idx])
    {
        float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[min_cost_idx]], p);

        // 更新条件：邻域有效 AND 成本优于当前假设
        if (depth_before >= params.depth_min && depth_before <= params.depth_max &&
            final_costs[min_cost_idx] < cost_now)
        {
            depth_now = depth_before;
            plane_hypotheses_now = plane_hypotheses[positions[min_cost_idx]];
            cost_now = final_costs[min_cost_idx];
            selected_views[center] = temp_selected_views;
        }
    }

    // ==================== 第十阶段：局部细化 ====================
    // 应用平面假设细化：在选定假设周围进行局部扰动
    // 这执行5种不同的平面扰动：
    // 1. 来自深度先验的随机法向量
    // 2. 当前假设
    // 3. 来自随机深度的随机法向量
    // 4. 随机法向量的当前假设
    // 5. 扰动深度
    // 对于每一个，计算成本并在发现改进时更新

    PlaneHypothesisRefinement(images, depths, cameras, &plane_hypotheses_now, &depth_now, &cost_now,
                              &rand_states[center], view_weights, weight_norm, prior_planes, plane_masks,
                              &restricted_cost, p, params);

    // ==================== 第十一阶段：分层模式过滤 ====================
    // 在分层模式中，仅接受超过阈值的改进以防止噪声传播

    if (params.hierarchy)
    {
        // 仅当改进显著时更新（成本降低 > 0.1）
        // 这防止了分层多尺度优化中的噪声传播
        //
        // 数学条件：
        // ┌─────────────────────────────────────────┐
        // │ 仅接受当：cost_now < pre_costs - 0.1    │
        // │ 这保证改进的稳定性                       │
        // └─────────────────────────────────────────┘
        if (cost_now < pre_costs[center] - 0.1f)
        {
            costs[center] = cost_now;
            plane_hypotheses[center] = plane_hypotheses_now;
        }
    }
    else
    {
        // 标准模式：始终接受更优解
        costs[center] = cost_now;
        plane_hypotheses[center] = plane_hypotheses_now;
    }
}

__global__ void BlackPixelUpdate(cudaTextureObjects *texture_objects, cudaTextureObjects *texture_depths, Camera *cameras, float4 *plane_hypotheses, float *costs, float *pre_costs, curandState *rand_states, unsigned int *selected_views, float4 *prior_planes, unsigned int *plane_masks, const PatchMatchParams params, const int iter)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0)
    {
        p.y = p.y * 2;
    }
    else
    {
        p.y = p.y * 2 + 1;
    }

    CheckerboardPropagation(texture_objects[0].images, texture_depths[0].images, cameras, plane_hypotheses, costs, pre_costs, rand_states, selected_views, prior_planes, plane_masks, p, params, iter);
}

__global__ void RedPixelUpdate(cudaTextureObjects *texture_objects, cudaTextureObjects *texture_depths, Camera *cameras, float4 *plane_hypotheses, float *costs, float *pre_costs, curandState *rand_states, unsigned int *selected_views, float4 *prior_planes, unsigned int *plane_masks, const PatchMatchParams params, const int iter)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0)
    {
        p.y = p.y * 2 + 1;
    }
    else
    {
        p.y = p.y * 2;
    }

    CheckerboardPropagation(texture_objects[0].images, texture_depths[0].images, cameras, plane_hypotheses, costs, pre_costs, rand_states, selected_views, prior_planes, plane_masks, p, params, iter);
}

__global__ void GetDepthandNormal(Camera *cameras, float4 *plane_hypotheses, const PatchMatchParams params)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const int width = cameras[0].width;
    const int height = cameras[0].height;

    if (p.x >= width || p.y >= height)
    {
        return;
    }

    const int center = p.y * width + p.x;
    plane_hypotheses[center].w = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
    plane_hypotheses[center] = TransformNormal(cameras[0], plane_hypotheses[center]);
}

__device__ void CheckerboardFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs, const int2 p)
{
    int width = cameras[0].width;
    int height = cameras[0].height;
    if (p.x >= width || p.y >= height)
    {
        return;
    }

    const int center = p.y * width + p.x;

    float filter[21];
    int index = 0;

    filter[index++] = plane_hypotheses[center].w;

    // Left
    const int left = center - 1;
    const int leftleft = center - 3;

    // Up
    const int up = center - width;
    const int upup = center - 3 * width;

    // Down
    const int down = center + width;
    const int downdown = center + 3 * width;

    // Right
    const int right = center + 1;
    const int rightright = center + 3;

    if (costs[center] < 0.001f)
    {
        return;
    }

    if (p.y > 0)
    {
        filter[index++] = plane_hypotheses[up].w;
    }
    if (p.y > 2)
    {
        filter[index++] = plane_hypotheses[upup].w;
    }
    if (p.y > 4)
    {
        filter[index++] = plane_hypotheses[upup - width * 2].w;
    }
    if (p.y < height - 1)
    {
        filter[index++] = plane_hypotheses[down].w;
    }
    if (p.y < height - 3)
    {
        filter[index++] = plane_hypotheses[downdown].w;
    }
    if (p.y < height - 5)
    {
        filter[index++] = plane_hypotheses[downdown + width * 2].w;
    }
    if (p.x > 0)
    {
        filter[index++] = plane_hypotheses[left].w;
    }
    if (p.x > 2)
    {
        filter[index++] = plane_hypotheses[leftleft].w;
    }
    if (p.x > 4)
    {
        filter[index++] = plane_hypotheses[leftleft - 2].w;
    }
    if (p.x < width - 1)
    {
        filter[index++] = plane_hypotheses[right].w;
    }
    if (p.x < width - 3)
    {
        filter[index++] = plane_hypotheses[rightright].w;
    }
    if (p.x < width - 5)
    {
        filter[index++] = plane_hypotheses[rightright + 2].w;
    }
    if (p.y > 0 &&
        p.x < width - 2)
    {
        filter[index++] = plane_hypotheses[up + 2].w;
    }
    if (p.y < height - 1 &&
        p.x < width - 2)
    {
        filter[index++] = plane_hypotheses[down + 2].w;
    }
    if (p.y > 0 &&
        p.x > 1)
    {
        filter[index++] = plane_hypotheses[up - 2].w;
    }
    if (p.y < height - 1 &&
        p.x > 1)
    {
        filter[index++] = plane_hypotheses[down - 2].w;
    }
    if (p.x > 0 &&
        p.y > 2)
    {
        filter[index++] = plane_hypotheses[left - width * 2].w;
    }
    if (p.x < width - 1 &&
        p.y > 2)
    {
        filter[index++] = plane_hypotheses[right - width * 2].w;
    }
    if (p.x > 0 &&
        p.y < height - 2)
    {
        filter[index++] = plane_hypotheses[left + width * 2].w;
    }
    if (p.x < width - 1 &&
        p.y < height - 2)
    {
        filter[index++] = plane_hypotheses[right + width * 2].w;
    }

    sort_small(filter, index);
    int median_index = index / 2;
    if (index % 2 == 0)
    {
        plane_hypotheses[center].w = (filter[median_index - 1] + filter[median_index]) / 2;
    }
    else
    {
        plane_hypotheses[center].w = filter[median_index];
    }
}

__global__ void BlackPixelFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0)
    {
        p.y = p.y * 2;
    }
    else
    {
        p.y = p.y * 2 + 1;
    }

    CheckerboardFilter(cameras, plane_hypotheses, costs, p);
}

__global__ void RedPixelFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0)
    {
        p.y = p.y * 2 + 1;
    }
    else
    {
        p.y = p.y * 2;
    }

    CheckerboardFilter(cameras, plane_hypotheses, costs, p);
}

void ACMMP::RunPatchMatch()
{
    const int width = cameras[0].width;
    const int height = cameras[0].height;

    int BLOCK_W = 32;
    int BLOCK_H = (BLOCK_W / 2);

    dim3 grid_size_randinit;
    grid_size_randinit.x = (width + 16 - 1) / 16;
    grid_size_randinit.y = (height + 16 - 1) / 16;
    grid_size_randinit.z = 1;
    dim3 block_size_randinit;
    block_size_randinit.x = 16;
    block_size_randinit.y = 16;
    block_size_randinit.z = 1;

    dim3 grid_size_checkerboard;
    grid_size_checkerboard.x = (width + BLOCK_W - 1) / BLOCK_W;
    grid_size_checkerboard.y = ((height / 2) + BLOCK_H - 1) / BLOCK_H;
    grid_size_checkerboard.z = 1;
    dim3 block_size_checkerboard;
    block_size_checkerboard.x = BLOCK_W;
    block_size_checkerboard.y = BLOCK_H;
    block_size_checkerboard.z = 1;

    int max_iterations = params.max_iterations;

    RandomInitialization<<<grid_size_randinit, block_size_randinit>>>(texture_objects_cuda, cameras_cuda, plane_hypotheses_cuda, scaled_plane_hypotheses_cuda, costs_cuda, pre_costs_cuda, rand_states_cuda, selected_views_cuda, prior_planes_cuda, plane_masks_cuda, params);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    for (int i = 0; i < max_iterations; ++i)
    {
        BlackPixelUpdate<<<grid_size_checkerboard, block_size_checkerboard>>>(texture_objects_cuda, texture_depths_cuda, cameras_cuda, plane_hypotheses_cuda, costs_cuda, pre_costs_cuda, rand_states_cuda, selected_views_cuda, prior_planes_cuda, plane_masks_cuda, params, i);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        RedPixelUpdate<<<grid_size_checkerboard, block_size_checkerboard>>>(texture_objects_cuda, texture_depths_cuda, cameras_cuda, plane_hypotheses_cuda, costs_cuda, pre_costs_cuda, rand_states_cuda, selected_views_cuda, prior_planes_cuda, plane_masks_cuda, params, i);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        printf("iteration: %d\n", i);
    }

    GetDepthandNormal<<<grid_size_randinit, block_size_randinit>>>(cameras_cuda, plane_hypotheses_cuda, params);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    BlackPixelFilter<<<grid_size_checkerboard, block_size_checkerboard>>>(cameras_cuda, plane_hypotheses_cuda, costs_cuda);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    RedPixelFilter<<<grid_size_checkerboard, block_size_checkerboard>>>(cameras_cuda, plane_hypotheses_cuda, costs_cuda);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    cudaMemcpy(plane_hypotheses_host, plane_hypotheses_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
    cudaMemcpy(costs_host, costs_cuda, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

/**
 * @brief 联合双边上采样(Joint Bilateral Upsampling, JBU)核函数
 *
 * @param jp JBU参数数组指针，包含以下信息：
 *          - height: 高分辨率参考图像的高度
 *          - width: 高分辨率参考图像的宽度
 *          - s_height: 低分辨率源深度图的高度
 *          - s_width: 低分辨率源深度图的宽度
 *          - Imagescale: 尺度比例（高分辨率与低分辨率的比值）
 *
 * @param jt JBU纹理对象指针，包含：
 *        - jt[0].imgs[0]: 高分辨率参考图像纹理
 *        - jt[0].imgs[1]: 低分辨率源深度图纹理
 *
 * @param depth 输出深度图数组（高分辨率）
 *
 * @details
 *   联合双边上采样算法原理：
 *
 *   1. 空间结构：从低分辨率上采样到高分辨率
 *      - 高分辨率像素(x, y) 对应低分辨率坐标：(x/scale, y/scale)
 *      - 邻域搜索范围：num_neighbors = WinWidth/2，其中WinWidth = Imagescale²+1
 *
 *   2. 双边加权公式：
 *      depth_up(x,y) = Σ w(i,j) * depth_src(i,j) / Σ w(i,j)
 *      其中权重 w(i,j) = exp(-Δspatial²/(2σ_d²)) × exp(-Δcolor²/(2σ_r²))
 *
 *      - 空间权重（σ_d=0.50）：基于在低分辨率空间中的距离
 *        w_spatial = exp(-((i-o_x)² + (j-o_y)²) / (2×0.50²))
 *
 *      - 色彩权重（σ_r=25.5）：基于高分辨率参考图像的色彩相似度
 *        w_color = exp(-|ref_pix - neighbor_pix|² / (2×25.5²))
 *
 *      - 总权重：w = w_spatial × w_color
 *
 *   3. 算法步骤：
 *      a) 计算线程所处的高分辨率像素坐标(p.x, p.y)
 *      b) 获取该点在参考图像中的像素值（用于色彩相似度）
 *      c) 计算对应的低分辨率中心坐标：(o_x, o_y) = (p.x/scale, p.y/scale)
 *      d) 遍历低分辨率邻域内的所有点
 *      e) 对于每个邻域点：
 *         - 计算空间高斯权重（低分辨率空间距离）
 *         - 计算色彩高斯权重（高分辨率参考图像色彩差异）
 *         - 累加加权的低分辨率深度值
 *         - 累加权重（用于归一化）
 *      f) 归一化：最终深度 = 加权深度和 / 权重和
 *
 *   4. 边界处理：
 *      - 使用镜像或截断方式处理超出边界的邻域坐标
 *      - 高分辨率：范围[0, width)×[0, height)
 *      - 低分辨率：范围[0, s_width)×[0, s_height)
 *
 *   5. 关键特性：
 *      - 保留高分辨率参考图像的边界和结构
 *      - 低分辨率深度值通过边界感知的加权融合上采样
 *      - 有效处理场景中的深度不连续（如物体边界）
 *
 *   数学公式汇总：
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │ o_x = p.x × scale，o_y = p.y × scale                            │
 *   │ w_spatial(i,j) = exp(-((i-o_x)² + (j-o_y)²)/(2σ_d²))           │
 *   │ w_color(i,j) = exp(-|ref[p]-ref_neighbor|²/(2σ_r²))            │
 *   │ w_total(i,j) = w_spatial × w_color                              │
 *   │                                                                  │
 *   │ depth_out = Σ_{i,j} w_total(i,j) × depth_src(i,j)              │
 *   │             ───────────────────────────────────────             │
 *   │                  Σ_{i,j} w_total(i,j)                          │
 *   └─────────────────────────────────────────────────────────────────┘
 */
__global__ void JBU_cu(JBUParameters *jp, JBUTexObj *jt, float *depth)
{
    // 计算当前线程对应的高分辨率像素坐标
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    // 获取图像尺寸参数
    const int rows = jp[0].height;       // 高分辨率高度
    const int cols = jp[0].width;        // 高分辨率宽度
    const int center = p.y * cols + p.x; // 线性索引

    // 边界检查：超出高分辨率图像范围则返回
    if (p.x >= cols)
    {
        return;
    }
    if (p.y >= rows)
    {
        return;
    }

    // 计算分辨率缩放因子：低分辨率/高分辨率的尺寸比
    const float scale = 1.0 * jp[0].s_width / jp[0].width;

    // 双边滤波的高斯核参数
    const float sigmad = 0.50; // 空间高斯标准差（低分辨率空间）
    const float sigmar = 25.5; // 色彩高斯标准差（参考图像灰度值）

    // 计算邻域搜索窗口大小
    const int WinWidth = jp[0].Imagescale * jp[0].Imagescale + 1; // 窗口宽度 = Imagescale²+1
    int num_neighbors = WinWidth / 2;                             // 邻域半径

    // 计算高分辨率像素在低分辨率空间中的对应坐标
    const float o_y = p.y * scale; // 低分辨率对应的y坐标
    const float o_x = p.x * scale; // 低分辨率对应的x坐标

    // 从高分辨率参考图像中读取当前像素的灰度值（用于色彩相似度计算）
    const float refPix = tex2D<float>(jt[0].imgs[0], p.x + 0.5f, p.y + 0.5f);

    // 邻域坐标变量
    int r_y = 0;  // 低分辨率邻域y坐标
    int r_ys = 0; // 高分辨率邻域y坐标
    int r_x = 0;  // 低分辨率邻域x坐标
    int r_xs = 0; // 高分辨率邻域x坐标

    // 权重和累积变量
    float sgauss = 0.0;             // 空间高斯权重
    float rgauss = 0.0;             // 色彩高斯权重
    float totalgauss = 0.0;         // 总权重 = 空间权重 × 色彩权重
    float total_val = 0.0;          // 加权深度值累积
    float normalizing_factor = 0.0; // 权重累积（用于归一化）

    // 像素值变量
    float srcPix = 0;      // 低分辨率深度图的像素值
    float neighborPix = 0; // 高分辨率参考图像邻域像素值

    // 双层循环：遍历邻域内的所有点
    for (int j = -num_neighbors; j <= num_neighbors; ++j)
    {
        // 计算低分辨率邻域y坐标（有边界处理）
        r_y = o_y + j;
        r_y = (r_y > 0 ? (r_y < jp[0].s_height ? r_y : jp[0].s_height - 1) : 0);

        // 计算高分辨率邻域y坐标（有边界处理）
        r_ys = p.y + j;
        r_ys = (r_ys > 0 ? (r_ys < jp[0].height ? r_ys : jp[0].height - 1) : 0);

        for (int i = -num_neighbors; i <= num_neighbors; ++i)
        {
            // 计算低分辨率邻域x坐标（有边界处理）
            r_x = o_x + i;
            r_x = (r_x > 0 ? (r_x < jp[0].s_width ? r_x : jp[0].s_width - 1) : 0);

            // 从低分辨率深度图中读取该邻域点的深度值
            srcPix = tex2D<float>(jt[0].imgs[1], r_x + 0.5f, r_y + 0.5f);

            // 计算高分辨率邻域x坐标（有边界处理）
            r_xs = p.x + i;
            r_xs = (r_xs > 0 ? (r_xs < jp[0].width ? r_xs : jp[0].width - 1) : 0);

            // 从高分辨率参考图像中读取邻域点的灰度值（用于计算色彩权重）
            neighborPix = tex2D<float>(jt[0].imgs[0], r_xs + 0.5f, r_ys + 0.5f);

            // 计算空间高斯权重
            // w_spatial = exp(-((i-o_x)² + (j-o_y)²)/(2×σ_d²))
            sgauss = SpatialGauss(o_x, o_y, r_x, r_y, sigmad);

            // 计算色彩高斯权重
            // w_color = exp(-|refPix - neighborPix|²/(2×σ_r²))
            rgauss = RangeGauss(fabs(refPix - neighborPix), sigmar);

            // 计算总权重 = 空间权重 × 色彩权重
            totalgauss = sgauss * rgauss;

            // 累积权重（用于后续归一化）
            normalizing_factor += totalgauss;

            // 累积加权的深度值
            total_val += srcPix * totalgauss;
        }
    }

    // 最终深度值 = 加权深度值之和 / 权重之和
    depth[center] = total_val / normalizing_factor;
}
void JBU::CudaRun()
{
    int rows = jp_h.height;
    int cols = jp_h.width;

    dim3 grid_size_initrand;
    grid_size_initrand.x = (cols + 16 - 1) / 16;
    grid_size_initrand.y = (rows + 16 - 1) / 16;
    grid_size_initrand.z = 1;
    dim3 block_size_initrand;
    block_size_initrand.x = 16;
    block_size_initrand.y = 16;
    block_size_initrand.z = 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaDeviceSynchronize();
    JBU_cu<<<grid_size_initrand, block_size_initrand>>>(jp_d, jt_d, depth_d);
    cudaDeviceSynchronize();

    cudaMemcpy(depth_h, depth_d, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total time needed for computation: %f seconds\n", milliseconds / 1000.f);
}
