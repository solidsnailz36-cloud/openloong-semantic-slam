# SLAM 大师赛 — 项目长期记忆

> 只放**跨会话必须记住的决策与坑**。数字与推导在文档里。**历史根因史在 `根因史_v4-v8.md`。**

## 索引
- **★★ `docs/交接_20261004_复赛感知精修.md`＝当前进度交接，新会话第一份读它**（含"第七轮/第八轮"全部结论；历史版本 `docs/_bak/`）。
- **`根因史_v4-v8.md`**（本目录）＝压缩后的逐轮根因，避免重复踩。
- `docs/复赛实施方案.md`（§10–§18）、`复赛自评与决赛差距.md`、`HANDOFF_项目交接.md`、`评审意见归因与修正方案.md`、`灾备与恢复清单.md`、`复盘_踩坑手册_小白向.md`（**改前备份到 `docs/_bak/`**）。
- 技能：`camera-lidar-person-fusion`（感知/融合/门限，含"仪器校准"章）、`gazebo-classic-headless-container`、`remote-job-and-snapshot`、`lidar-map-forensics`（**已 85 KB，待整理**）、`competition-submission-final-audit`。

## 家底 / 赛事硬约束
- 唯一"丢了不可再得"的只有 `D:\E-Downd\data.bag`（初赛 2,532 MB）。初赛 **81 分**，根因＝ RTK 降级段水平位置回退到里程计。邮箱 `open@openloong.org.cn` 抄送 `luqinghua@openloong.net`。
- **★ 官方答疑＝交付形态硬约束**：实时运行 + **整个数据集的视频** + 自建图 pcd；官方 pcd 只作提示；仿真 LiDAR yaw 按"**不修**"；输入只允许 **点云/图像/IMU**（`/state_estimation`·`/registered_scan` 只能事后评估）；**要输出建图 pcd + TF**；**建图 50% / 感知 50%**。

## 环境
- **仿真 = 本机 Docker 容器 `r2sim`**（ROS Noetic + Gazebo 11.15.1）；挂载 `D:/E-Downd/sim_official -> /work`；容器内代码 `/work/r2slam_ws`（**改完本机必须 `docker cp` 过去**）。`docker start r2sim`，`sim_up.sh` 起仿真（`Xvfb :99`）。**Docker Desktop 必须开着**。世界文件在容器内 `/root/catkin_ws/src/vehicle_simulator/world/house.world`。
- **真实侧在本机 `.venv` 跑**（1427 帧 ≈ 6–8 min）。云端 T4 已废弃。`wsl.exe`/`wmic.exe` 被安全策略拦截。

## ★★ 官方仿真器两条最贵的坑
1. **LiDAR 不随车体旋转**（`vehicleSimulator.cpp` 把 LiDAR yaw 写死 0）⇒ **航向不可观测**，`heading.py` 用上一帧指令角速度×实测 dt 死推（p50 1.85°）；**转圈不产生新信息**。
2. **相机—LiDAR 外参只在 yaw≈0 成立** ⇒ `p_cam = Rz(-yaw) @ Tcl @ p_lidar`，按 `LidarSpec.rotates_with_base` 配置化（仿真 false，真实 true）。
- ⚠️ **"反投影回原框"自检抓不到这两条**（往返自消，98.9% 通过）。有效判据＝**与 `/gazebo/model_states` 比对**。
- **★ 仿真硬精度上界**：无 IMU、无里程计话题（只有 `/state_estimation`=真值），航向只能死推，`err_yaw` p50 **0.46°–12.05°** ⇒ **单跑之间不能比分数**，要比就**同一份采集重放**。

## 感知（定论）
- 检测器 = **YOLO11n ONNX**（`vendor/detector/yolo11n.onnx`，640，COCO80；运行时只用 onnxruntime/cv2，**MAP 里没有 torch**）。旧 tiny-yolov3 退役。
  - 备选 **yolo26n.onnx**（9.4 MB，同 `[1,84,8400]` 输出、parity 已验）已在 `vendor/detector/_export/`；**实测仿真更好、真实鱼眼观测 −17% ⇒ 维持 11n**（细节见交接第八轮）。
- YOLO 输出裸 `[1,84,8400]`（cxcywh+letterbox，**无自带 NMS**）⇒ 解码自写 ⇒ **新模型必须跑 `tools/check_onnx_parity.py`**。
- `labels.py` = 类别表唯一定义源。**尺寸门限是"充分 or + 必要 and"两层**：`LABEL_MAX_WIDTH`/`LABEL_BOX_MIN_PX`/`LABEL_BOX_HEIGHT` 是充分（`or`），`LABEL_BOX_HEIGHT_MUST={PERSON:(1.40,3.50)}` 是必要（`and`）。⚠️ **`or` 会让被 `or` 掉的那条彻底失效** ⇒ 每条门限都要报"它独自拒了多少"。
- `fusion.py`：深度用**相机中心距离** → `is_floor` 剔地面 → **深度间隙切近组** → **体素连通域取大簇**；返回 `_idx` 供 `obj_mask`。位置＝簇的中位数。

## 语义地图三色定义（boss 问过）
```
绿 ground     = ground_ever & ~obstacle   这格确认是地板（有回波落在 |h|<容差）
灰 seen&~ground = 看过这里但没找到地板
黑 其它        = 从没有回波落在这里
```
**关键：格子只有"有回波落进去"才算 seen，射线穿过不算** ⇒ 真开着的门应画黑。
**障碍层三条规则**（`worldmap.py`）：`OBSTACLE_MIN_VOTES=2`（帧数投票）、
`OBSTACLE_RELEASE_FRAMES=2`（**自上次障碍证据以来看到地板 ≥2 帧就解除**，但 `votes>=12` 的强证据永不解封）、`blocked()=obstacle|obstacle_now`（**安全层**）。

## 真实数据集
- `openloong_r2.zip`(6.34 G) → `D:\E-Downd\openloong_r2_ex\`（`config/real_r2.yaml`，map_extent 60 ⇒ n=1200）。
- 相机 FishPoly 1600×1296, A11=730.56；`LiDAR2IMU_0`≈单位旋转 ⇒ **IMU 与 LiDAR 共姿态**；图像 topic `/camera/rgb`。
- 点云 49,152 点/帧带 `offset_time` 0–0.094 s ⇒ **必须去斜**；IMU 成串到达、基本不可用。标定用**等距 `r=f·θ`（f=730.6）**。

## 算法栈不变量
- `config.py` + `sources/` 是全项目**唯一**允许知道"仿真/真实"的地方；下游只认 `Frame`。**下游出现 `if sim` = 设计失败**。
- `tools/online_pipeline.py` = 产出交付物主程序；`tools/package_submission.py` = 终审（任一 FAIL 非零退出）。**终审 FAIL 绝不许写 zip**。
- 交付物 `r2slam/out/{online_sim,online_real}/`：run.mp4 + map.pcd + label_table.yaml + summary.json + frames.csv + tf.csv + tf/。pcd 字段 **`x y z label rgb`（17 B/点）**。
- **★ 当前包 = `提交包/…【固态蜗牛队】…zip` = 36,204,882 B / md5 `ec8260d28de6590033be36daca247b6c`**（终审 0 FAIL / 0 WARN，**是 v6 门限的产物；v7/v8 改动尚未进包**）。上一版 `129d3764…` 在 `_backup/20261004_2100_v5_129d3764.zip`。
- **`--dump-grids` 必须写全**：`ground/obstacle/seen/ground_ever/obstacle_votes/obstacle_now/ground_after_obstacle/traj` + meta（含 `obstacle_min_votes`/`obstacle_release_frames`）⇒ **交付产物自己就是仪器**，任何规则改动都能在真产物上离线比。
- **bag 记录时刻 ≠ 传感器时间** ⇒ 一律 `header.stamp`；`offset_time` 从帧戳倒推 ⇒ `time_offsets_from: end`。
- **`--npz` 复算没有驱动器** ⇒ 上一帧指令恒 0 ⇒ 航向恒 0 ⇒ **每个物体被绕机器人转过错误角度而地图看着正常** ⇒ `dump_frame(extra={"control_yaw": yaw})` 必须存。（**好处**：轨迹不随地图变 ⇒ 改地图规则时是干净 A/B；**坏处**：测不到规划反馈。）
- **采集 npz 存的是未去斜原始点云**；诊断工具要用它就必须自己 `Deskewer.apply`，否则测的不是那一跑（差别见 `根因史_v4-v8.md` v7）。
- **改完管线必须两源各跑 5 帧冒烟**；改函数签名后**必须真跑一次**。
- **传输/比对：尺寸不算证据** ⇒ 比 md5。`voxel_size=0.5`；`pcd_voxel 0.15` 与配准 voxel 解耦。

## 探索（5 条硬规则；详见方案 §17/§18）
1. 候选只按距离排⇒**极限环** ⇒ 前沿**簇连通分量** + `score=簇大小/(1+距离)` + `min_target_range`。
2. **可达性是"连通分量"不是直线射线**（症状＝看得见门口、进不去）。
3. **地面掩码是"一层点"** ⇒ 先闭运算，但**不能穿已知障碍** ⇒ `close(free) & ~dilate(obstacle)`。
4. **航向差不是停车理由——该画弧**（三档油门：原地转 / ≤25° 直行 / 走弧）。
5. 大视场传感器上 frontier 不够 ⇒ 加 `sweep`；目标指向**块的深处**。⚠️ 加"进度期限"实测更差。
- **★ 每个"什么都没发生"的判据都要说清盯的是哪个量**（帧数/直线/看到的地面 都曾盯错）。停条件改对过两次：盯"看到的地面"→改"**没走过的地面**"；**赶路时误判停滞** ⇒ 加 `_closing_on_target()`。
- ⚠️ 探索常被 `--duration` 截断（`done_reason: null`），**房子北半边从未靠近** ⇒ 漏检主要是**没走到**。
- ⚠️ **仿真单跑方差极大**（46–112 m）⇒ **最该修的东西之一**。

## 硬规则
- **每个会话开头先补 PATH**：`export PATH="/c/Users/ZhuanZ/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:$PATH"`。
- **含中文路径**：存在性判断一律用 Python `os.path.exists`，不要用 Git Bash 的 ls/cd/find。
- 同一文件多处修改**串行**发 Edit；`Write` 覆盖前先 `Read`。
- **★ 最普适：自检通过 ≠ 正确**。判据要落在**具体量**上。
- **★ commit message 含反引号不要用 shell 的 `-m "..."`**（会被命令替换吃掉）⇒ 用 `-F 文件`。
