# 检测器

> ⚠️ **2026-10-09 起默认已换**：`detect.py:DEFAULT_MODEL_FILE = "yoloworld_alias_320.onnx"`
> ——**开放词汇**头（51 条烘焙 prompt / 39 类），**不再是**下面的闭集 COCO 模型。
> 这份 README 余下部分讲的是**闭集那条线**（yolo11n → yolo26n，COCO-80），
> 它现在是 `--model` 的**回退选项**，取舍与实测数字仍然有效。
> 开放词汇那一轮的 32 个臂在 `docs/开放词汇边缘化_第一轮实测_20261009.md`。

## 在用的是什么

| 文件 | 大小 | 说明 |
|---|---|---|
| **`yoloworld_alias_320.onnx`** | **50,368,392 B** | **当前默认**（`detect.py:DEFAULT_MODEL_FILE`；`online_pipeline.py` / `replay_explore.py` 共读这一个常量）。**开放词汇**头：**51 条烘焙 prompt / 39 类**，类别索引是**我们自己的 prompt 顺序**（不是 COCO 的），映射在同目录 `.classmap.json` sidecar |
| `yoloworld_*` 其余臂 | 13–50 MB | `open` / `persononly` / `robotonly` / `promptform` / `templ` / `s*` 是**词表臂**；`int8` / `pc` / `t4` / `t6` / `416` / `640` 是**量化与尺寸臂**。⚠️ **INT8 在 51 类上失效 ⇒ 只能 fp32** |
| `yolo26n-seg_320.onnx` | 11,026,122 B | **实例分割头**（`r2slam/seg.py`，`--seg-model`）。**与检测器并跑、不是替它**：它是闭集 COCO-80 ⇒ 只提供形状，`--seg-classes` 默认 `person` |
| `yolo26n.onnx` | 9,883,834 B | 闭集 COCO-80 检测器（2026-10-04 起曾是默认）⇒ 回退：`--model vendor/detector/yolo26n.onnx` |
| `yolo11n.onnx` | 10,741,326 B | 更早一版闭集默认，保留作对照 |
| `coco.names` | 625 B | 80 个类名（`labels.py` 里的 COCO id 都对着它） |
| `tiny-yolov3-11.onnx` | 35,511,756 B | 已退役（会把 youbot / 椅子认成人），只留档 |
| `_export/*.onnx` | yolo11s 38 MB / yolo11m 80 MB / yolo26n / yolo11n | 实验导出的候选，**不进包** |

⚠️ `*.onnx` 在 `.gitignore` 里 ⇒ **模型不在版本库里**，换机器/重新 clone 后必须手动补齐，
否则 `_resolve_vendor` 找不到文件会直接报错（不会静默退回别的模型）。

闭集那一对（`yolo11n` / `yolo26n`）的**原始输出相同**：`[1, 84, 8400]`，`cxcywh + letterbox`，**无自带 NMS**。
所以 `r2slam/detect.py` 的解码是共用的，换模型不需要改解码器 ——
但**每次换模型都必须跑 `tools/check_onnx_parity.py`**（它逐帧比对 ultralytics 的参考结果）：

```
python3 tools/check_onnx_parity.py --glob "vendor/detector/_export/frames/*.png" \
    --onnx vendor/detector/yolo26n.onnx --pt vendor/detector/_export/yolo26n.pt
# 实测 worst 偏差 1.26 px（容差 1.5），0 漏检、0 类错
```

## 为什么换成 yolo26n（实测，别再走一遍）

Ultralytics 的模型序列是 **YOLO11 → YOLO26**（2026-01 发布；**没有 v26/v30 这种叫法**）。
YOLO26 去掉 DFL、支持无 NMS 的端到端推理，官方称 **CPU ONNX 推理快 43%**，
训练配方里的 **STAL（小目标感知标签分配）** 专门改善小目标 —— 正对我们"远处人框只有 32 px"的痛点。

同一份仿真采集（1229 帧，真值投进来匹配，`tools/diagnose_fusion.py`）：

| | 人框总数 | 落地真人的框 | 过完融合门限后 真/虚 | 单帧耗时 | 体积 |
|---|---|---|---|---|---|
| yolo11n | 1227 | 214 | 178 / 115 | 77.4 ms | 10.7 MB |
| **yolo26n** | 1131 | 202 | **182 / 111** | **65.7 ms** | **9.4 MB** |

⇒ 更快、更小、**过完门限后保住的真人更多、虚多更少**。

**但真实鱼眼那一侧要如实记**：同一份真实采集上 `localised 757 → 630`，
**全类别均匀地少 17%**（person −74、chair −13、refrigerator −14、bottle −16、suitcase −8），
确认的人是**同一批 8 个位置**。真实侧没有真值，**分不清"去掉误检"还是"丢掉真物"**。
换成它，是因为在"感知占 50% 分"的权衡里**实例级没有退化** —— 这个取舍记在这里，
不是因为它只有优点。回退只需 `--model vendor/detector/yolo11n.onnx`。

## 候选里被否掉的（都实测过）

| 候选 | 结果 |
|---|---|
| `yolo11s` / `yolo11m` | 精度不比 11n 好（落空处的"人"框 880 vs 818），耗时 269 / 767 ms（预算 100 ms/帧），弃 |
| 只取检测框中心附近的点 | 见技能：**精度 44% → 24%**，因为框小 ⇒ 簇小 ⇒ `width_lat` 小 ⇒ 而它正是尺寸门限的充分条件 ⇒ 虚多更容易过关 |
| `yolov4-tiny`（darknet） | 输出**全部 NaN**，cfg 与 release 权重不匹配，弃 |
| 用 `cv2.dnn` 当运行时 | boxes 845 条 / scores 2535 条，长度对不上 ⇒ 解码不可能正确，弃 |
| `ultralytics` + torch 上线 | 要拖 2 GB torch，与赛题"需适配嵌入式计算平台"相悖 ⇒ **运行时只用 onnxruntime** |

## tiny-yolov3 时期的三个坑（留档：换模型时不要再踩）

1. 框是 **(y1, x1, y2, x2)** 且已是原图像素，不要乘图像尺寸；
2. 图内 NMS 的 `indices` **返回空**，必须自己写 NMS；
3. `scores` 的轴序随运行时变化，解码器要**按形状判断再转置**。

（YOLO11/26 是裸 `[1,84,8400]`，以上三条都不适用；它们的坑是"必须自己做 letterbox 反变换"。）

详见技能 `camera-lidar-person-fusion`。
