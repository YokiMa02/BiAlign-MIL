# BiAlign-MIL V1：模型源码与使用说明

BiAlign-MIL 在空间配对的高、低倍率特征之间进行区域汇聚和实例反馈，增强高倍率实例后完成切片分类。代码中的原始类名保留为 `BiAlignMS_PTCMIL`，以保持现有导入和 checkpoint 参数名称兼容。

## 1. 交付范围与来源

| 文件 | 状态 |
|---|---|
| `models/model_bialign_ms.py` | 从作者此前上传的文本附件逐字节恢复，**未修改计算、参数名称或注释** |
| `config.py` | 根据作者提供的正式 V1 配置整理 |
| `data.py` | 根据作者提供的字典查找配对逻辑重新编写，增加输入检查 |
| `run_v1.py` | 新增的训练/推理入口，
| `verify_package.py` | 语法、原文件哈希与数据配对检查 |


```text
bialign_v1/
├── README.md
├── config.py
├── data.py
├── run_v1.py
├── verify_package.py
└── models/
    ├── __init__.py
    ├── model_bialign_ms.py              
    ├── model_vit_pmt_clu.py             
    ├── nystrom_attention_custom.py     
    └── tcformer/
        ├── __init__.py                 
        ├── tcformer_layers.py          
        ├── tcformer_utils.py           
        └── transformer_utils.py      
```
## 2. 实际模型结构

1. `high_model`：从High checkpoint加载，所有参数冻结，保持eval，输出参考概率。
2. `ms_model`：从同一High checkpoint初始化，只解冻 `layer1`、`_fc3`、`ctm`、`norm`。
3. 使用冻结的 `ms_model._fc1` 获得High嵌入，用 `low_proj` 获得Low嵌入，维度均为384。
4. 只对 `low_valid > 0.5` 的实例，按父级Low坐标分组；Low为query，同组High为keys/values，使用4头交叉注意力。
5. 将注意力输出与Low嵌入相加，形成父级上下文，并回传至对应High实例。
6. 根据High嵌入、父级上下文和有效掩膜计算实例标量门控。将上下文投影回768维后，以门控残差更新原始High特征。
7. 按原索引写回增强特征，送入MS聚合分支完成预测。

记父级为j、子实例为i，则关键运算为：
```text
u_j = e_low_j + MHA(query=e_low_j, keys=E_high_group_j, values=E_high_group_j)
g_i = sigmoid(gate_mlp(concat(e_high_i, u_j, m_i)))
h_enhanced_i = h_i + m_i * g_i * out_proj(u_j)
```

未配对实例保持原始特征。Low特征是父级区域表示，不是每个High实例单独生成的新Low特征；同组实例共享父级上下文，但门控不同。


## 3. 环境与依赖

优先使用原实验服务器已验证的环境。已知依赖包括Python、PyTorch、NumPy、scikit-learn，以及主干使用的 `nystrom-attention`、`einops`、`mmcv`；主干最终依赖以复制回来的源码导入为准。

当前未取得原环境版本清单，因此不提供猜测的版本锁定文件。在服务器上保存真实版本：
```bash
python -m pip freeze > environment-original.txt
```

原模型和checkpoint需要同一版主干。`strict=True` 加载若报参数缺失或多余，应检查依赖和checkpoint来源，不要改成 `strict=False` 来隐藏不兼容。

## 4. 输入格式

每次前向处理一张WSI，不拼接多张切片作为同一个bag。

```text
features: [N,1536]，前768维High，后768维Low
coords:   [N,5]，每行[hx,hy,lx,ly,low_valid]
label:    [1]，类别0或1
```



## 5. 配置与损失

详见 `config.py`。V1外层源码固定二分类、384维嵌入和4头跨尺度注意力。

| 项目 | 值 |
|---|---|
| High/Low输入维度 | 各768 |
| 主干层数/簇数 | 2 / 5 |
| Optimizer | Adam |
| 融合模块学习率 | 2e-4 |
| 解冻主干学习率 | 2e-5 |
| Weight decay | 1e-5 |
| 最大epoch | 300 |
| Early-stop参数 | MIN_EPOCHS=20, PATIENCE=20 |
| Batch size | 1张WSI |
| 输出投影初始化 | 权重与偏置均为零 |
| 门控末层偏置 | -2.0 |


