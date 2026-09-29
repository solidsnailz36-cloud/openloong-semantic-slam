# 具身感知与导航 · 复赛算法栈（r2slam）

一套代码，两个数据源。这是复赛方案书 §4.0 的实现，也是交付物的产出地。

```
                    ┌──────── 统一数据接口（唯一差异点）────────┐
   SimSource ──────►│  Frame{ stamp, cloud, intensity, t_off,   │◄────── BagSource
   (ROS 1 / Gazebo) │         image, ring, meta }               │       (ROS 2 sqlite)
                    └──────────────────┬────────────────────────┘
                                       ▼
        deskew → odometry → detect → fuse → traversable → worldmap → 驱动 → render
                                       │                            (explore | goto)
              ┌────────────────────────┼────────────────────────┐
          运行视频                   建图 pcd                  逐帧性能表
                                  + 类别表 + 目标关联实例
```

## 为什么这么分

复赛评审规则原文看的是「**泛化性**」与「**不同环境下的算法部署和迁移能力**」。
所以"两个数据源"不是负担，是**要拿出来的证据**——前提是它得是真的：

- **`r2slam/config.py` 与 `r2slam/sources/` 是全项目唯一允许出现 `if sim / if real` 的地方**，
  而且分支不写在代码里，写在 `config/*.yaml` 里（相机模型、分辨率、外参、点云字段名、是否去斜、
  线程数、voxel、量程）。
- 其余模块只认 `Frame`，不认数据源。**任何下游模块里出现 `if sim`，就是设计失败。**

唯一的例外是 bag 的**路径**：它属于机器而不属于数据集，所以可以用 `R2SLAM_BAG_PATH`
环境变量覆盖，避免同一份配置在不同机器上分叉成两份。

## 模块

| 文件 | 职责 |
|---|---|
| `config.py` | 相机模型 / 点云字段 / 外参 / **实时档（线程数、voxel、量程）** |
| `frames.py` | `Frame` —— 下游唯一接口 |
| `sensors.py` | `PointCloud2` 通用解码（按字段表读偏移，两源同一份） |
| `camera.py` | 针孔 + 鱼眼等距投影、反投影方位表 |
| `deskew.py` | **真实侧必需**：匀速模型运动补偿（`p_ref = T_motion(dt) @ p`） |
| `odometry.py` | KISS-ICP 的唯一构造处（测量工具与管线共用，避免参数漂移） |
| `heading.py` | **只有扫描器不随车体转时用**：用上一条指令的角速度死推航向（见下） |
| `detect.py` | `tiny-yolov3` ONNX 检测（onnxruntime / cv2 双后端，**线程数受限**）；`detect_mapped()` 直接给**本项目的 label id** |
| `fusion.py` | 框 → 三维位置（取框内点 30 分位深度）+ **反投影回框内的硬判据**；人与物体走同一条几何 |
| `labels.py` | **类别表的唯一定义源**（id ↔ 名称 ↔ 颜色 ↔ COCO 映射 ↔ 门限），并导出 `label_table.yaml` |
| `traversable.py` | 地面 RANSAC + 高度带 / 闭运算 / 障碍膨胀 → 2D 可通行栅格 |
| `worldmap.py` | 世界系累加、俯视渲染（带类别图例）、**带标签 + rgb 的 pcd 输出** |
| `explore.py` | **frontier 探索**（官方仿真不含规划器）+ 卡住自救 + 速度发布 |
| `nav.py` | **决赛半场**：实例关联（`ObjectRegistry`）+ 指令解析（`find_label`）+ A* + `SemanticNavigator` |
| `viz.py` | 渲染（只用 cv2，两个平台都有） |
| `tools/online_pipeline.py` | **主程序**：一个进程、一帧一帧、产出全部交付物（`--explore` / `--goto`） |
| `tools/package_submission.py` | **终审脚本**：逐条核对交付物并打包，不通过就非零退出 |
| `tools/test_explore.py` `tools/test_nav.py` | 行为测试（零 ROS 依赖，两平台逐字一致） |

## 两个数据源的差异（全部落在配置里）

| 项 | `sim_house.yaml` | `real_corridor.yaml` |
|---|---|---|
| 中间件 | ROS 1 live（rospy 订阅） | ROS 2 bag（rosbag2/sqlite3 重放） |
| LiDAR | VLP-16，5,600 点/帧 @5 Hz | 48×1024，49,152 点/帧 @10 Hz |
| 点云字段 | `x y z intensity ring time` | `x y z intensity confidence offset_time` |
| 去斜 | **不需要**（`time` 实测 5,600/5,600 全 0） | **必须**（`offset_time` 0–0.094 s） |
| 相机 | 320×180，针孔，D 全 0，HFOV 114.6° | 1600×1296，鱼眼等距 `r = A11·θ` |
| 相机—LiDAR | **不是单位阵**：光学系 vs 机体系，差一个轴置换 | `Tcl_0`，同旋转 + 约 3 cm 平移 |
| **扫描器是否随车体转** | **不转**（见下）⇒ 航向死推 + 融合加 `Rz(-yaw)` | 转，正常 |
| 真值 | `/state_estimation`（200 Hz）**仅用于评分** | 无（用 `LiDAR Map.pcd` 做绝对校验） |

第一条外参那一行值得说清楚：**"相机和 LiDAR 装在同一位置"不等于"外参是单位阵"**——
位置相同，朝向不同。按单位阵投影，5,600 个点里 **0 个入画**。
消歧不能靠"点阵看起来在左边"（那是循环论证），要用**机器人自己的真实位移**：
驱动后取世界位移、用权威 TF 转进 LiDAR 系，实测 `(1,0,0)`。

### ★ 官方仿真器的 LiDAR 不随车体旋转

`vehicleSimulator.cpp` 给相机和车体赋的是完整姿态，但在移动 LiDAR 模型前**重新赋值四元数、把 yaw 写死为 0**：

```cpp
geoQuat = tf::createQuaternionMsgFromRollPitchYaw(vehicleRoll, vehiclePitch, vehicleYaw);
cameraState.pose.orientation = geoQuat;   // 相机随车体转
geoQuat = tf::createQuaternionMsgFromRollPitchYaw(terrainRoll, terrainPitch, 0);
lidarState.pose.orientation = geoQuat;    // ← yaw 没了
```

LiDAR 被搬到车体位置，但**朝向恒为世界对齐**。实测：我们的 yaw 全程在 ±1.6° 内，真值 yaw 扫过 359°，
指令累计转过 6656° —— **KISS-ICP 报"没转"是对的**。两个后果都要处理：

1. **航向不可观测** ⇒ 按航向控制的策略永远转不完（实测 586 s 只到过 15 个 1 m 网格）。
   `heading.py` 用上一条指令的角速度死推航向（轮式里程计语义，不是真值）。实测死推误差 p50 **1.85°**。
2. **相机—LiDAR 外参只在 yaw≈0 成立** ⇒ 正确变换是 `p_cam = Rz(-yaw) @ Tcl @ p_lidar`。

⚠️ **"反投影回原框"的自检抓不到第 2 条**：它是一次往返，同一个错误变换首尾相消
（实测 98.9% 通过，yaw 90–180° 区间 100% 通过）。**自检只能抓"前后不一致"，抓不到"整体错了一个角度"。**

两件事都写在 `LidarSpec.rotates_with_base` 里，因为它们是**数据源的属性**，不是算法选择。
换一份仿真环境必须重验这一条。

## 运行

```bash
# ---- 仿真：两阶段，唯一能拿到干净数字的流程 ----
# 阶段一：Gazebo 在跑，探索驱动 + 每帧落盘（不检测、不渲染）
bash run_online.sh sim_house --duration 110 --explore --no-detect --no-render \
     --dump /tmp/sim_capture
# 阶段二：停掉 Gazebo，用同一批帧复算（检测 + 渲染 + 编码）
bash /root/sim/kill_sim.sh
bash run_online.sh sim_house --npz /tmp/sim_capture --panel-h 540 --encode

# ---- 真实 bag（路径由 R2SLAM_BAG_PATH 指定）----
R2SLAM_BAG_PATH=/root/sim/three1 bash run_online.sh real_corridor \
     --out /tmp/online_real --encode

# ---- 行为测试（零 ROS 依赖，两平台结果一致）----
PYTHONPATH=. python tools/test_explore.py
PYTHONPATH=. python tools/test_nav.py

# ---- 决赛行为：先探索，直到目标类别上地图，再交棒去导航 ----
bash run_online.sh sim_house --out /tmp/goto --explore --goto '沙发' --duration 300

# ---- 终审 + 打包（不通过就非零退出）----
python tools/package_submission.py --root out --out 提交包

# ---- 只导出帧做静态检查 ----
python tools/export_frames.py --config config/real_corridor.yaml --out out/frames_real
```

**改完管线必须先两源各跑 5 帧冒烟**（`--frames 5 --no-render`）。这一条是拿一次白跑换来的：
一个"只在首帧就检出物体时才触发"的 `UnboundLocalError` 让仿真连跑三次完整数据集都没报错，
而真实走廊第一帧就炸——因为首帧没有检出时循环体被跳过，同一轮里那个变量随后被赋值。

**为什么要分两阶段**：仿真和我们共用同一个 8 核 cgroup 配额（`nproc` 报 80，`cpu.max` 给 8），
Gazebo 软渲染 + RViz 就能把它吃满，所有墙钟数字被放大数倍。实测同一份代码同一批数据：
阶段一 `trav_fit` 19.9 ms，阶段二（未限检测线程时）104 ms。

## 性能与硬规则（详见 `docs/复赛实施方案.md` §13–§16）

在线栈本体（停掉重负载后复算，`total` 已减去 `render`）：

| 环境 | 帧 | odom | detect | trav_fit | map | 在线栈合计 |
|---|---|---|---|---|---|---|
| 仿真 5,600 点 @5 Hz | 141 | 14.4 | 31.9 | 19.5 | 3.2 ms | **≈69 ms**（预算 200） |
| 真实 49,152 点 @10 Hz | 1,336 | 10.1 | 32.0 | 24.9 | 15.6 ms | **≈95 ms**（预算 100） |

**三条从坑里换来的硬规则**：

1. **配额受限的容器/嵌入式环境里，不要让任何库按 `nproc` 决定线程池大小。**
   同一个坑在**三个库**上各来一次，而且**罚的都是别人的代码**（cgroup 冻结落在当时正在跑的任意一行）：
   KISS-ICP/TBB、onnxruntime 的 intra-op 池、**OpenCV 的全局池**（`cv2.setNumThreads`，
   必须早设一次，且只有真·大图才触发）。先读 `/sys/fs/cgroup/cpu.max`，再显式设线程数。
   顺手把 ORT 的空转自旋关掉（`session.intra_op.allow_spinning=0`）。
2. **不要在饱和的机器上做优化决策。** 我照一个 150 ms 的读数把地面 RANSAC 向量化了，
   结果在空闲机上**没有任何提速**（18.6→18.6 ms）还把平面挪了 1.2 cm ⇒ 已回退。
   怀疑机器变慢时先跑 `vendor/sim_scripts/probe_cpu.py`（跨机可比的定标）
   或 `bench_fit_ground.py`（同数据跨机对比同一个函数）。
   管线现在**自带 cgroup 节流前后快照**，跑完直接告诉你这一轮的数字有没有被污染。
3. **bag 的记录时刻不是传感器的时间。** `rosbags.messages()` 的 `t` 是写入时刻；
   记录器一冲刷就成簇，而 `header.stamp` 仍然均匀。用错时钟 ⇒ `dt=0` ⇒ 里程计从那一帧起发散
   （实测 130 s 跑出 9,573 m，修好后 114.85 m）。另外 `offset_time` 的方向
   （`lidar.time_offsets_from`）在数据里**看不出来**，猜错是把畸变**翻倍**而不是没修。

## 交付物口径

- **视频**：≤2 min（两份组委会材料里取更严的）、必须含可视化界面。
  加速播放时**加速倍数会烧进每一帧**——否则那是对性能的不实声明。
- **pcd**：`x y z label rgb` 二进制，每点 17 B。
  - `label` 是 uint8，取值见 `label_table.yaml`（随 pcd 一起交付）。
  - `rgb` 是 PCL **packed-float**，**由 label 推导**，所以点云在 CloudCompare /
    pcl_viewer / Open3D 里**打开即为语义着色**，不需要先加载调色板。
  - **不导出 `intensity`**：真实数据有、仿真没有，而两源字段逐字相同正是"迁移能力"的证据；
    为一个源上恒为零的字段破坏它不划算。
  - FAQ 不提供类别表 ⇒ 交付物必须自带。`package_submission.py` 会检查
    "pcd 里出现过的每一个 label 是否都在表里"，不满足就 FAIL。

### 类别表（机制）

| 段 | 内容 |
|---|---|
| 几何类 0/1 | `background` / `traversable` —— LiDAR 几何判定，**零训练** |
| 物体类 2.. | 23 类室内物体（person、sofa、chair、bed、table、tv、fridge、plant、sink、toilet、oven、microwave、bench、suitcase、backpack、vase、clock、book、laptop、keyboard、bottle、cup、bowl），COCO 2D 检测经标定投影到三维 |

保留哪些 COCO 类别不是"能检出什么"（那是全部 80 类），而是两条物理约束：
**LiDAR 看得见**（太小就没有回波，只会变成误报）与**人能用一句话指出来**
（决赛是"语音指令 → 导航到目标物"）。此外每类还有两个门限
（`min_points` 支撑点数、`min_box_frac` 框高占画面比例），都是物理量而不是调参。
**检测器本身没有被训练或微调过。**

## 待确认（已知的不确定项）

1. `calib.yaml` 里 `FishPoly` 的 `k2..k7` **闭式形式未确定**——用纯等距 `r = A11·θ`
   已能对齐走廊透视结构（见 `docs/复赛实施方案.md` §2.1），多出的多项式项未使用。
   系数原样保留在配置里，需要向组委会确认公式。
2. 人物的评价口径（检测框还是分割掩码）未定；本栈给的是框 + 三维位置 + 高度区间，
   两种口径都能换算。
