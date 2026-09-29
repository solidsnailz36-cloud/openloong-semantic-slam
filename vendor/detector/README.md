# 检测器（人物）

## 在用的是什么

| 文件 | 大小 | 内容 |
|---|---|---|
| `tiny-yolov3-11.onnx` | 35,511,756 B | tiny-yolov3，COCO 80 类，**person = 0 号**，opset 11，**图内自带 NMS** |
| `coco.names` | 625 B | 80 个类名 |

来源：ONNX Model Zoo
`https://media.githubusercontent.com/media/onnx/models/main/validated/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx`
（注意要用 `media.githubusercontent.com`，直接 `github.com/.../raw/...` 只会拿到 LFS 指针）

运行时用 **onnxruntime**（几 MB），**不拖 torch** —— 赛题要求"需适配嵌入式计算平台"。

```
云端 Ubuntu/py3.8:  pip3 install --target /root/sim/pylibs -i <mirror> onnxruntime==1.17.3
                    PYTHONPATH=/root/sim/pylibs python3 ...
本机 Windows/py3.13: pip install --index-url https://pypi.org/simple onnxruntime   # 国内镜像无 cp313 轮子
```

两平台结果**逐位一致**（人位、score、反投影都对得上）。

## 为什么不是别的（都实测过，别再走一遍）

| 候选 | 结果 |
|---|---|
| `yolov4-tiny`（darknet cfg + weights，24 MB） | 输出**全部 NaN**（12,675/12,675），cfg 与 release 权重不匹配，换图像、换后端都一样 ⇒ 弃 |
| 同机 `cv2.dnn` 当运行时 | 能加载，但 **boxes 只给 845 条、scores 有 2535 条**，两个张量长度对不上 ⇒ 解码不可能正确 ⇒ 弃 |
| 云端系统 `cv2 4.2` / `opencv-python-headless 4.8.1` | 读不了这个 ONNX（`convertInt64ToInt32` 越界 / `Unsupported data type: BOOL`）⇒ 只能上 onnxruntime |
| `ultralytics` / YOLOv8 | 要拖 2 GB 的 torch，与"适配嵌入式平台"相悖 |

## 用之前先记住三个坑

1. 框是 **(y1, x1, y2, x2)** 且**已是原图像素**，不要乘图像尺寸。
2. 图内 NMS 的 `indices` **返回空**，必须自己写 NMS。
3. `scores` 的轴序随运行时变化，解码器要**按形状判断再转置**。

详见技能 `camera-lidar-person-fusion`。
