# FSEVCT_EVCJM - 基于极值理论与Copula联合模型的开集识别框架

> 基于 PyTorch 实现的雷达/图像开集识别（Open-Set Recognition, OSR）算法框架，提出了**特征空间极值收敛定理（FSEVCT）**与**极值Copula联合模型（EVCJM）**，解决了传统极值理论在开放环境下缺乏定量约束和忽略聚类间相关性的核心痛点。

## 📖 项目背景

在开集识别（OSR）任务中，模型不仅需要正确分类已知类别，还必须有效拒绝未知类别的样本。现有基于极值理论（EVT）的方法（如 MDT, EVBT）存在两个严重缺陷：
1. **缺乏可量化的理论条件**：仅凭“聚类紧凑度”、“样本充足”等模糊定性描述，缺乏严格的定量约束，导致理论适用性和实验可复现性差。
2. **独立聚类假设**：忽略特征空间中多个已知类簇之间的联合分布和统计依赖关系。

本项目通过 FSEVCT 和 EVCJM 彻底解决了上述两个缺陷，并在多个公开基准数据集上验证了算法的有效性与鲁棒性。

## 💡 核心贡献

- **提出 FSEVCT 定理**：严格定义了**聚类稳定性、尾部独立性、样本充足性**三个可量化的约束条件，证明了归一化特征距离在满足条件时收敛于广义极值（GEV）分布，打破了传统方法的定性模糊性。
- **设计 EVCJM 模型**：引入极值 Copula 理论构建多元联合极值分布，打破了传统方法中“独立聚类”的不合理假设。通过参数 \(\theta\) 刻画类簇间的极值距离相依结构（当 \(\theta \to 1\) 时退化为独立情形，当 \(\theta \to \infty\) 时完全相关）。
- **即插即用框架（P1/P2）**：本方法可无缝集成到现有的 SoftMax、GCPL、RPL、ARPL 等特征提取网络中，以极低的计算代价（\(O(N \cdot l^2)\) 后处理复杂度）带来稳定且显著的性能提升。

## 📊 框架图与实验结果

![Framework](/images/framework.png)
*图1：FSEVCT-EVCJM 整体训练与测试流程图*

## 📝 论文与代码链接

- **论文标题**：Feature space extreme value convergence theorem and copula joint model for open set recognition
- **发表期刊**：Pattern Recognition (Volume 180, 2026)
- **DOI**：[10.1016/j.patcog.2026.114352](https://doi.org/10.1016/j.patcog.2026.114352)
  
