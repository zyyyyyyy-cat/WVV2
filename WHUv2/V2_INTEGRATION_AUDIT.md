# WV V2 集成审查（2026-10-27）

## 本次启动故障

服务器为 PyTorch 1.12.1+cu113，而 `requirements.txt` 固定的是本机较新版本。1.12.1 的 `torch.load` 没有 `weights_only` 参数，旧调用把它传给 `pickle.Unpickler`，四个 rank 均在 UAD checkpoint 加载处退出。V2 的 `utils/checkpoint.py` 现在按 `torch.load` 签名选择参数；UAD 热启动、V2 恢复、V2 评测加载以及 Backbone 加载统一使用此入口。测试覆盖旧版和新版签名。服务器目录 `/project/code/WHU_base/ViSA+MD+MP/v2` 必须部署这些更新后再运行。

## 源码连接与最终方案核对

| 环节 | 原实现的连接 | V2 实际连接与判断 |
| --- | --- | --- |
| UAD Global | 原 UAD `base` CLS → BNNeck → Identity CE；pre-BN CLS → WRT、ProCA、GPD | 原 158 个张量全量热启动并逐键验形；保留相同的 Global 损失数据流；不使用 WV V1.1 权重。 |
| Specific | ViSA 的 view token 参与 view-aware 表示；WV V1.1 已有 Specific token | V2 三模态 Specific token 单向读取 CLS/Patch，原 CLS/Patch 不读取它；CLS/Patch11 初始数值与 UAD 路径相符。 |
| ETGM | ViSA 的 Shared 与 View MoE 路由出专家 Prompt | V2 是 Shared、RGB、共用 IR 三池，每池 4 Expert、Top-2、8 共享 Query Slot 加专家残差。Shared 输入 CLS 停梯度，Specific 输入可导。 |
| Patch/GCN | ViSA 的局部图利用所选 Patch；WV V1.1 只有 Top-40 GCN | V2 两路 Query 先对全部 128 个 Patch11 做可导 Cross-Attention，再各自选择 Top-40 构图；GCN Patch 值和邻接停梯度，Query 与 GCN 权重可导。 |
| Local 聚合 | WV V1.1 每路 16 Query，连同 Out Token 作 33 Token Self-Attention | V2 各路 8 Query，共 16 Key/Value；单个 CLS 条件 Out Query 做 Cross-Attention，经弱 CLS 残差和门控 FFN 得 768D Local。 |
| 新损失 | ViSA 对 Local 有身份及度量监督，原 UAD 对 Global 有 CE/WRT/ProCA/GPD | V2 使用 Specific WRT、Local Cosine CE、六方向 Local CM、Fusion WRT、Balance，并按 10 epoch ramp；保留全部 UAD 损失，没有 ViSA 原损失的直接移植。 |
| 检索 | ViSA 拼接其 Global/Local 输出；UAD 单一 Global 描述子 | V2 分别报告 Global、Local、Fusion；Fusion 为单位化的 `[L2(Global), 0.5 L2(Local)]`，训练 Fusion WRT 的 Global 直接分量停梯度，评测使用同一数值构造。 |
| 评测 | WHU-MARS 用统一 Gallery 和跨相机过滤 | `TOP_K_EVAL=0` 完整排序，报告总体 Rank-1/mAP/mINP、九个模态方向和跨平台子协议。 |

上述连接以 `../ideav2.md`、`../guidev2.md` 为准；ViSA/UAD 原源码仅用于核对机制，并不覆盖 V2 选择。

## 消融结果给 V2 的可执行经验

- `../ViSA.pdf` 第 7 页 Table 2：ViT Baseline 的 ALL Rank-1/mAP 为 61.54/53.54，加入 DLFM 的对应值为 67.31/62.86，三模块组合为 70.51/65.46。论文认为局部关系和专家条件信息互补。V2 因此保留 Query→Patch→GCN→Local 的可导路径，同时保留 ETGM；检查中已确认这些参数能从联合损失获得有限梯度。该表并不能单独证明 V2 的新 Patch Read 或新的损失必然提高 WHU-MARS 成绩。
- `../WHU.pdf` 第 8 页 Table 8：UAD Base、Base+ProCA、Base+GPD、全组合的 mAP/Rank-1 分别为 8.9/23.9、9.3/24.5、10.9/28.7、11.0/29.5。V2 保留 Global 的 CE/WRT/ProCA/GPD、完整热启动及训练集重建的 GPD Bank，避免新增分支破坏已学的 Global 结构。
- `../WHU.pdf` 第 8 页 Table 6：UAD 的 TIR→RGB Rank-1/mAP 为 9.04/6.32，TIR→NIR 为 8.57/5.69，弱于 RGB→RGB 的 40.47/26.83。V2 的六方向 Local CM 因而逐方向记录有效 Query 覆盖率；正式评测逐方向看 Global、Local、Fusion，而不是只看总体损失或单一 Rank-1。

## 本轮验收与边界

- 本机机制测试 9 项通过，包括 Local CM 数值/梯度对照、Specific/GCN 梯度边界、UAD Bank 重建和旧/新 `torch.load` 签名。
- 真实 UAD Epoch 120 checkpoint 的 158 个原有张量加载通过，CLS/Patch11 初始等价通过；三模态全尺寸合成 Batch 的 UAD 与 V2 联合损失前向/反向为有限值，关键 ETGM、Patch Read、GCN、聚合器、分类器梯度有限。
- 优化器覆盖全部 395 个可训练张量，旧/新参数学习率分组符合配置；合成的完整评测返回 Rank-1、mAP、mINP 并生成域报告。
- 本机 Windows PyTorch 构建缺少可用的 Gloo rendezvous，四进程测试未在本机通过。`tests/ddp_smoke.py` 已提供给 Linux 服务器运行；CUDA/NCCL、真实数据读取、首批显存和四卡吞吐仍需服务器验收。不要把本机 CPU 检查视为正式训练完成。
