#include <torch/extension.h>

template <typename scalar_t>
__global__ void trilinear_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp
){
    // 1. 计算编号
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;
    // 2. 排除不需要计算的thread，（防止 memory leak)
    if (n>=feats.size(0) || f>=feats.size(2)) return;

    // point -1~1
    const scalar_t u = (points[n][0]+1)/2;
    const scalar_t v = (points[n][1]+1)/2;
    const scalar_t w = (points[n][2]+1)/2;
    
    const scalar_t a = (1-v)*(1-w);
    const scalar_t b = (1-v)*w;
    const scalar_t c = v*(1-w);
    const scalar_t d = 1-a-b-c;
    feat_interp[n][f] = (1-u)*(a*feats[n][0][f] +
                               b*feats[n][1][f] +
                               c*feats[n][2][f] +
                               d*feats[n][3][f]) + 
                            u*(a*feats[n][4][f] +
                               b*feats[n][5][f] +
                               c*feats[n][6][f] +
                               d*feats[n][7][f]);
}

torch::Tensor trilinear_fw_cu(
    torch::Tensor feats,
    torch::Tensor points
){
    const int N = feats.size(0), F = feats.size(2);

    torch::Tensor feat_interp = torch::zeros({N, F}, feats.options());
    // torch::zeros({N, F}, torch::dtype(torch::kInt32).device(feats.device));

    // 有两个要平行运算，一个N，一个F，因此写成16 * 16
    // 如果说只有一个维度要平行运算，则为256
    const dim3 threads(16, 16); // 256(128, 512)
    const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y-1)/threads.y);
    
    // 1. "trilinear_fw_cu"仅仅为desc
    // 2. trilinear_fw_kernel 为要叫出来的内核
    // 3. <<<block, thread>>> 标准写法，dim3
    // 4. 参数的类型为tensor，要用packed_accesssor为CUDA内核准备数据访问器，这些访问器可以直接在GPU上使用
    // 5. 若参数类型是一个标量，可以直接用：
    // AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu", 
    // ([&] {
    //     trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
    //         a,
    //         feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
    //         points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
    //         feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
    //     );
    // }));
    // 6. scalar_t可以直接指定，例如float，写法如下：
    // AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu",     
    // ([&] {
    //     trilinear_fw_kernel<<<blocks, threads>>>(                        // 去掉了<scalar_t>
    //         feats.packed_accessor<float, 3, torch::RestrictPtrTraits>(),         // <scalar_t> 换成 float，因此size_t也就确定下来了，size_t去掉了
    //         points.packed_accessor<float, 2, torch::RestrictPtrTraits>(),        // 同上
    //         feat_interp.packed_accessor<float, 2, torch::RestrictPtrTraits>()    // 同上
    //     );
    // }));
    // 7. 3表示tensor的n_dimension, 2同理
    // 8. torch::RestrictPtrTraits 限制不同张量之间在内存上不重叠
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu", 
    ([&] {
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));
    return feat_interp;

    // 如果说回传两个 tensor，函数的类型可以改成 std::vector<torch::Tensor>
    // 返回 {feat_interp, feat_interp2}
}

template <typename scalar_t>
__global__ void trilinear_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dfeat_interp,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> dL_dfeats
){
    // 1. 计算编号
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;
    // 2. 排除不需要计算的thread，（防止 memory leak)
    if (n>=feats.size(0) || f>=feats.size(2)) return;

    // point -1~1
    const scalar_t u = (points[n][0]+1)/2;
    const scalar_t v = (points[n][1]+1)/2;
    const scalar_t w = (points[n][2]+1)/2;
    
    const scalar_t a = (1-v)*(1-w);
    const scalar_t b = (1-v)*w;
    const scalar_t c = v*(1-w);
    const scalar_t d = 1-a-b-c;

    dL_dfeats[n][0][f] = (1-u) * a * dL_dfeat_interp[n][f];
    dL_dfeats[n][1][f] = (1-u) * b * dL_dfeat_interp[n][f];
    dL_dfeats[n][2][f] = (1-u) * c * dL_dfeat_interp[n][f];
    dL_dfeats[n][3][f] = (1-u) * d * dL_dfeat_interp[n][f];
    dL_dfeats[n][4][f] = u * a * dL_dfeat_interp[n][f];
    dL_dfeats[n][5][f] = u * b * dL_dfeat_interp[n][f];
    dL_dfeats[n][6][f] = u * c * dL_dfeat_interp[n][f];
    dL_dfeats[n][7][f] = u * d * dL_dfeat_interp[n][f];
}

torch::Tensor trilinear_bw_cu(
    const torch::Tensor dL_dfeat_interp,
    const torch::Tensor feats,
    const torch::Tensor points
){
    const int N = feats.size(0), F = feats.size(2);
    
    torch::Tensor dL_dfeats = torch::empty({N, 8, F}, feats.options());

    const dim3 threads(16, 16);
    const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y-1)/threads.y);

    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_bw_cu", 
    ([&] {
        trilinear_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dfeat_interp.packed_accessor<scalar_t, 2 , torch::RestrictPtrTraits, size_t>(),
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dfeats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return dL_dfeats;
}