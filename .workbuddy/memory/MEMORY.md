# SLAM 大师赛 — 项目长期记忆

> 只放**跨会话必须记住的决策与坑**。数字细节、完整链路、Gazebo 五个坑等在各文档里，按"细节索引"去找，不要往这里搬。

## 细节索引（先读这些）
- `docs/复赛实施方案.md` —— **复赛主文档**；**§10–§15 = 第 0/1/2/3 步实施记录**（含五个 Gazebo 坑、四判据实测、三个线程池坑、bag 时钟坑）。
- **`docs/复赛自评与决赛差距.md`（2026-09-29）**：自评按复赛评审口径（泛化/迁移）+ **距决赛的四个要件缺口**（物体级语义 / 目标关联 / 语义导航 / 真机适配）+ 三档改进计划。**判断"能不能进决赛"时先读它。**
- `docs/HANDOFF_项目交接.md`（8 阶段链路 / 不变量 / 回退链）、`docs/评审意见归因与修正方案.md`（初赛根因）。
- `docs/复盘_踩坑手册_小白向.md`（3,340 行 / 62 坑 / 术语表 101 条）—— 改前先备份到 `docs/_bak/`。
- **`docs/灾备与恢复清单.md`（2026-09-28）**：本机资产全清单 + 三级损失分级 + 新机器上重建 Gazebo 的照抄步骤。**怀疑"东西是不是丢了"时先读它。**
- 对外说明：`out/run5_keepdeg/{README.md,AUDIT.md}`。**写新对外说明前先读上面几份；踩新坑同时补进复盘手册。**
- 技能：`gazebo-classic-headless-container` / `camera-lidar-person-fusion` / `remote-job-and-snapshot` / `headless-rviz-recording` / `lidar-map-forensics` / `competition-dataset-triage` / `competition-submission-final-audit`。

## 本机家底（2026-09-28 清点，别重复清点）
- **初赛原始数据在本地**：`D:\E-Downd\data.bag`（2,532 MB，`/livox/lidar` 6,522 帧）——**唯一"丢了不可再得"的东西**。
  复赛数据集与官方仿真包（`openloong_ex\official\`）也在本地。代码/成品/权重全在本地（`.workbuddy/tmp/`、`r2slam/`、`out/`、`提交包/`）。
- **云端独占的只有 `run/scans/`**（6,522 帧去斜点云）——**而它恰好不需要**（初赛窗口已关）。
- **本机算力**：`nvidia-smi` 正常，GPU **RTX 2060**，CPU 16 线程。但 `wsl.exe` / `wmic.exe` **被本机安全策略拦截**。
- **复赛交付三项都不需要 GPU**（VLP-16 仅 5,600 点/5 Hz，llvmpipe 软渲染即可）；真实那份本机也能做。

## 赛事现状
- 初赛已提交评审：**81 分**。意见＝点云分类不错 / **轨迹平滑度欠缺** / **RTK 轨迹置信度偏高**。
- 已提交包 `out/run4_v2/submission.zip`（132,544,618 B / md5 `044801db…`）。
- **修复包（未提交，窗口已关）**：`out/run5_keepdeg/submission.zip` = **138,917,347 B / md5 `4065213c0ca9a16a57f8bbf4adcbf4d1`**，已复制为 `提交包/…—v3平滑度修复.zip`，终审 PASS。
- 邮箱 `open@openloong.org.cn`，抄送 `luqinghua@openloong.net`。语义选型 **SFPNet**（优于 Cylinder3D）；地面五类归并为 road。

## 初赛评审根因（已归档，细节见 `docs/评审意见归因与修正方案.md` §5–§6）
- 真凶**不是**"最近邻"，是 **RTK 降级段的水平位置回退到"里程计 + 弧长插值改正量"** ⇒ 里程计自己的周期被暴露。
  修法四条已全链重出（`out/run5_keepdeg/`）。收益：降级段 r1 **−0.929 → −0.337**、与同时刻插值 RTK 偏离 **12.49 → 0.08 cm**。

## 复赛：仿真环境（Gazebo Classic + ROS 1 Noetic）
- 官方包 `gazebo_models.zip` 47.9 MB / `src.zip` 315 KB（`openloong.obs.cn-east-3.myhuaweicloud.com/public/static/contests/master-1/`）；
  启动 `roslaunch vehicle_simulator house_vehicle.launch`。**场景 20×15 m 单层住宅**；**行人模型是静态道具**（`<actor>` 出现 0 次）。
- 传感器：**VLP-16 `/velodyne_points` 5 Hz / 5,600 点/帧**；相机 **HFOV 114.6° / 320×180 / 15 Hz / 零畸变**。
  真值 **`/state_estimation` 200 Hz** + `/registered_scan`；控制 **`/cmd_vel`**；**仿真器自己不规划路径**（"场景探索"要我们自己发 frontier）。
- **cam↔LiDAR 外参不是单位阵**（旧结论已推翻）→ 见"★ 仿真外参"。
- 云端 T4 `gpufree-container`：Ubuntu 20.04 + ROS Noetic + Gazebo 11.15.1 **预装**；80 核/376 GB；**llvmpipe 软渲染 + `Xvfb :99`**（GPU 没用到）。
  关键路径 `/root/sim/`、`~/.gazebo/models`（含补装 `hokuyo`）。
  **⚠️ `/root/sim/pylibs` 已废弃**（自带 numpy/cv2 会顶掉 venv 版本）——依赖现在全在 `slam_comp/venv`，
  入口是 `run_online.sh`；详见"算法栈"一节。
- **版本化副本 `vendor/sim_scripts/`**：`kill_sim.sh`（安全清场）/ `sim_up.sh` / `verify_sim.sh`（四判据）/ `person_probe.py` /
  `run_online.sh`（**唯一跑管线入口**）/ `t4_state.sh` / `probe_cpu.py` /
  `t4_diag.sh`·`t4_watch.sh`·`t4_retarget.sh`·`t4_jupyter_check.sh` / `put_pkg.sh` / `rmt.py` / `rr.sh`·`rr_proxy.sh`。
  **凭据只放 `.workbuddy/tmp/ssh_env.sh`，不进 vendor。** 补件模型 `vendor/gazebo_models_patch/hokuyo/`（务必保留）。
- **容器重启会重置 `/etc/hosts`** ⇒ `models.gazebosim.org` 的黑洞化**要每次开机后补一遍**（2026-09-29 实测计数 0）。
  不补就是那个经典卡死：世界加载永久阻塞、`sim_time` 恒 0、零报错。**这也是容器里唯一会丢的东西**——
  2026-09-29 复查：`~/.gazebo/models` 30 个、编译产物、`pylibs`、`slam_comp` 20 GB（含 **`run/scans` 6522 帧 / 767 MB**）全都还在。
- 相机实测 **~11.6–12.0 Hz**（URDF 标称 15）——软渲染下的真实值。

### 连不上远端时（2026-09-24 定型）
- **先跑 `bash vendor/sim_scripts/t4_diag.sh`**：一条命令出判决（本机 TCP → 平台主站 → **第三方节点带对照组**）。
- **TIMEOUT 有两种、本地分不出来**：外面也连不上＝实例没起来；**外面连得上＝是我们这条网络路径到不了它** ⇒ **改地址/重启实例是朝错方向使劲**。
- **对照组是硬要求**：必须同时探 `1.1.1.1:<port>` 与 `8.8.8.8:<port>`，阈值取 **≥2**（单个 OK 属噪声）。
- **2026-09-29 定性**：手机流量能正常打开 Jupyter ⇒ 实例正常、**只有校园网这条路径到不了**（CERNET↔移动回程）。
  工作通道：① 手机热点/USB 共享；② 本机 v2rayN SOCKS 在 **`127.0.0.1:10808`**，**必须切全局模式**（规则模式会把国内目的地放直连）
  ⇒ `bash vendor/sim_scripts/rr_proxy.sh '<cmd>'`。`remote_ssh.py`/`rmt.py` 都支持 `SSH_SOCKS=host:port`（**SFTP 也能过**）。
- `pc1.jupytergateway.gpufree.cn` 的 A 记录 = SSH 那台机 ⇒ Jupyter 与 SSH 同一入口、一起失效，不是独立问题。

## 复赛：算法栈 `r2slam/`（一套代码两个数据源）
- **纪律**：`r2slam/config.py` + `r2slam/sources/` 是全项目**唯一**允许知道"仿真/真实"的地方，分支写在 `config/*.yaml`；下游只认 `Frame`。
  **下游出现 `if sim` = 设计失败。** 唯一例外是 bag **路径**（属机器不属数据集）⇒ 用 `R2SLAM_BAG_PATH` 覆盖。
- 模块 `config/frames/sensors/camera/viz/detect/traversable/fusion/deskew/odometry/worldmap/explore.py` +
  `sources/{base,bag_ros2,sim_ros1}` + `tools/{online_pipeline,export_frames,dump_frames,perception_demo,test_explore}.py`；
  **`viz.py` 只用 cv2**（远端无 matplotlib）。
- **`tools/online_pipeline.py` 是产出交付物的主程序**：一条循环八阶段
  `ingest→deskew→odom→detect→fuse→traversable→worldmap→explore→render`，
  两个视频 + 建图 pcd 同一循环里出来。`--explore` 驱动 / `--dump` 落盘 / `--npz` 复算 / `--encode` 编码。
  **入口 `vendor/sim_scripts/run_online.sh`**（唯一入口，把三个依赖源合成一个解释器）。
- 实测：仿真 5.000 Hz / 5,600 点 / 配对 30/30；真实 10.328 Hz / 49,152 点 / 配对 40/40。**入画率差异是视场比**（相机 114.6° vs LiDAR 360°），不是 bug。
- **仿真器不提供规划器**（`src.zip` 无 planner），只有 `/cmd_vel`（**TwistStamped**）⇒ "场景探索"要自己写 frontier。
- **三项交付物已齐**：`r2slam/out/online_sim/`（92 s / 1500×540 / 轨迹 14.04 m / 可通行 123.7 m²）、
  `r2slam/out/online_real/`（**104.12 s / 1206×540 / 轨迹 114.85 m / 可通行 419.9 m² / pcd 92.4 MB**）；
  在线栈本体 ≈95 ms/帧（含渲染 170 ms）。入口 `vendor/sim_scripts/run_online.sh`。
- **bag 的记录时刻 ≠ 传感器时间**（`rosbags` 的 `t` vs `msg.header.stamp`）——**必须用 header.stamp**，
  否则记录器缓冲一冲刷 dt 就成 0/0.5 s 交替，里程计从那一帧起发散（实测 130 s 跑出 9,573 m）。
- **`offset_time` 从帧戳倒推** ⇒ `lidar.time_offsets_from: end`；猜错是把畸变**翻倍**。
- **远端 python 环境**：`slam_comp/venv`（把 `include-system-site-packages` 改成 `true` 才有 `rospy`；
  **`/root/sim/pylibs` 已废弃**——它自带 numpy/cv2 会顶掉 venv 的版本）。

## 复赛：感知（"能否检出人物"＝ **能**）
- 检测器 `vendor/detector/tiny-yolov3-11.onnx`（35.5 MB，person=0）+ **onnxruntime**；**不用 ultralytics**（要适配嵌入式计算平台）。
  12–36 ms/帧（10 Hz 预算 100 ms）。**有效距离 3–6 m**；1.5–2 m 检不出（上半身出画），8 m 被墙挡。
- **解码三坑（都静默出错）**：① 框是 **(y1,x1,y2,x2) 且已是原图像素**（当归一化会算出负高度）；② **图内 NMS 的 `indices` 是空的**，必须自写 NMS；
  ③ `onnxruntime` scores `(80,N)` vs `cv2.dnn` `(N,80)` 且长度都对不上 ⇒ **解码要轴序免疫**，`cv2.dnn` 后端弃用。
- **可通行（几何法、零训练）三个归零点**：① 障碍只取**离地 0.35–2.0 m 的带**（否则天花板把全图标成障碍 ⇒ 0.0%）；
  ② **先闭运算补地面圆环缝、再对障碍膨胀**（直接腐蚀自由空间 ⇒ 0.1%）；③ 地面容差**随距离增长** `0.10+0.035·d`。
  实测真实地面 `z≈−1.18 m`、可通行 = 已观测 **57%**；仿真 `z≈−0.79 m`。**两源阈值完全相同、没调参。**
- **融合硬判据**：三维人位**反投影回图像必须落回原框内**（符号错误肉眼看不出来）——真实侧 6/6 OK。深度取框内点 **30 分位**。

## ★ 仿真外参不是单位阵（实测推翻旧结论）
- "相机与 LiDAR spawn 在同一 `(vehicleX,Y,Z)` ⇒ 外参≈单位阵"——**那是位置相同，不是朝向相同**。
  相机=光学系(z前/x右/y下)，LiDAR=机体系(x前/y左/z上)。**按单位阵投影，5,600 点里 0 点入画。**
- 正解 `Tcl`：`cam z←LiDAR x`、`cam x←−LiDAR y`、`cam y←−LiDAR z`，平移 0。真实侧同旋转 + 3 cm 平移。
- **消歧方法（可复用）**：两候选只差绕光轴一个 yaw，而"地面在下/天花板在上"**对左右翻转不敏感** ⇒ 必须用**机器人自己的真实位移**
  （驱动后取 `/state_estimation` 世界位移，用权威 `map→sensor` TF 转进 LiDAR 系，实测 `(1,0,0)`）。**目视"点阵在左边"是循环论证。**
- 仿真 TF **只有 `map→sensor` 一个帧**（无独立 lidar/camera 帧）。

## 复赛：真实数据集（`D:\E-Downd\openloong.zip` 1.97 GB）
- 4 个文件：`calib.yaml` / `LiDAR Map.pcd`(先验地图，**仅墙体**，72×52×6.7 m) / `map_20260918_164047.bin`(554 MB) / `three1.zip`(1.45 GB)。
- **标定齐全不用自己标**：鱼眼 `FishPoly` 1600×1296、`maxIncidentAngle 120°`；`Tcl_0` 语义已解（同上旋转 + ~3 cm）。
  **投影已验证**：等距 `r=f·θ`（f=730.6、主点 839.1/657.9）与走廊透视图吻合；针孔边缘发散。
- **`three1` = 真机 ROS 2 bag**（rosbag2/sqlite3），130 s：`/odin1/cloud_raw`(1336) / `image/compressed`(1335，JPEG 鱼眼) / `imu`(559)。
- 点云 **48×1024 = 49,152 点/帧**，带 **`offset_time` 0–0.094 s ⇒ 必须去斜**（用自家里程计，不依赖 IMU）。
- **IMU 基本不可用**（突发采集、总有效 ~0.67 s）；**赛题也不要求用 IMU**。**`LiDAR Map.pcd` 与仿真 house 不是同一场景。**
- `map_*.bin`（`MAPV0001`）：**字段语义未破译（厂商私有）⇒ 我方不需要解析**。复算入口 `.workbuddy/tmp/{read_bag3,bag_timing,verify_calib3,probe_mapbin1..6,render_lidarmap}.py`。

## 复赛任务与交付
- **四处换轨**：离线→在线；有 RTK→无 RTK 室内；19 类静态语义→**人物 + 可通行区域**；新增**相机-LiDAR 融合**。数据两套（仿真 + 真实）。
- **交付三项**：① 仿真环境运行视频 ② 真实数据集运行视频 ③ **建图 pcd**。视频取更严口径 **≤2 min + 含可视化界面**。
- **语义方案**：可通行走**几何法**；人物走 **2D 检测 + 标定投影到 3D**。**不搬 SFPNet**（19 类 SemanticKITTI 与"人物/可通行"不是一回事）。
- **风险**：① **网络可达性会反复**（见"连不上远端时"）；
  ② **实时性 —— 已关闭（2026-09-29 定案）**：根因**不是算法**，是**容器 CPU 配额被线程池超订**
  （`nproc`=80 是宿主机的，`cpu.max` 只给 8；`cpu.stat` 22% 周期被节流）。
  解药 = **显式限制每个库的线程池**（KISS-ICP `registration.max_num_threads`；onnxruntime `intra_op_num_threads`），
  限线程**不改变轨迹**。完整表格与机制 → **`docs/复赛实施方案.md` §13/§14**，过程 → `.memory/2026-09-29.md`。
  测量坑：循环同一批帧会让 ICP 虚快（必须连续真序列）；**只看 p50/p90 会误判，要导出逐帧序列**；
  归因必须先做受控实验（"帧间运动量""地图膨胀"两个误归因都被实验推翻）。
- ③ 仿真相机 320×180 → **已验：能检出（3–6 m）**。

## 初赛交付口径（复赛 pcd 可沿用）
- 类别编号用 mmdet3d SemanticKITTI 19 类 0 基：8=road / 12=building / 13=fence / 14=vegetation / 15=trunk / 17=pole / 18=traffic-sign
  （官方表 road=9，**差 1**）。FAQ 不提供类别表 ⇒ **交付物必须自带 id↔名称↔色表**。
- `path.yaml` 的 `stamp` 是 **bag 原生相对秒**；schema ＝ `path:` 列表（`altitude/latitude/longitude/stamp`，缩进 4/4/6）+ 结尾 `total_points:`。
- **组委会材料互不覆盖**，终审必须两份都读、按更严的执行（录屏时长就吃过亏：主页 5 min vs 文档 2 min）。

## 场景物理事实 / 坐标系
- 初赛场地是**坡地**（真值 altitude 2.493~9.288 m）。`D:\E-Downd\path.yaml` 是**另一个采集会话**（仅格式样例）——
  **用任何 path.yaml 前先核 stamp 范围与点数**。初赛交付物统一在 **ENU**（原点＝首个 RTK 固定解帧）。
- `geo_trajectory.py` 只做 2D 水平拟合 + 竖直中值平移 ⇒ `up`/`altitude` 带 0.883° 倾角；"只改地图、轨迹不动"是错的。

## 已证伪 / 已定论（别重复劳动）
- **地面三级细分不可恢复**（分界是到传感器的距离，不是材质）⇒ 归并是唯一诚实做法。**car = 自车**（车顶 MID-360 扫到自身），应剔除。
- **平滑"改正量"是陷阱**：`e = r_local − P0` 含里程计抖动的负值，滤波会把抵消一起滤掉 ⇒ 抖动原样回到位姿。
  **改位姿一律 `PICK=1`；真要平滑就平滑位置。**
- **最近邻的时差是常量滞后（30 ms），不是锯齿来源**；归因靠分段实测，别用"时差×速度"讲故事。**字段存在 ≠ 字段有效**（仿真 `time` 全 0）。

## 硬规则（只留没别处可查的；其余见技能，避免重复）
- **每个会话开头先补 PATH**：`export PATH="/c/Users/ZhuanZ/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:$PATH"`
  （Bash 工具本机缺 coreutils，`dirname`/`tail` 都会 command not found）。
- **含中文的路径**：存在性判断一律用 Python `os.path.exists`，不要用 Git Bash 的 ls/cd/find。
- **调远端 / 传文件前先** `export MSYS2_ARG_CONV_EXCL='*' MSYS_NO_PATHCONV=1` —— Git Bash 会把远端 POSIX 路径改写成 Windows 路径，
  SFTP 于是报 **ENOENT「No such file」而远端明明存在**（本地路径含中文时更容易误判成中文问题）。
- 其余硬规则已分别固化在技能里，不要在这里重复：
  `remote-job-and-snapshot`（Windows↔远端四坑、长任务脱 channel、快照、**§7.1 第三方节点+对照组**、
  **§9.1 任何库的线程池都要限 / §9.2 别在饱和机器上优化 / §9.3 落盘→复算 / §9.4 后台线程要上报失败**）、
  `gazebo-classic-headless-container`（pkill 自杀、`set +u`、`check_world_models.py`、只认 `sim_time`、`stdbuf`、
  catkin 0 字节 CMakeLists、**驱动三要求 + frontier 四要点**）、
  `competition-submission-final-audit`（"新的一件+旧的三件"、近似公式先怀疑公式）、
  `lidar-map-forensics`（RTK 只取 `status==2`、平滑"改正量"陷阱）。
- **同一文件多处修改必须串行发 Edit**（并行会互相覆盖）；`Write` 覆盖前先 `Read`。
- **★ 两个跨项目通用的测量纪律**（本轮各栽一次，已写进技能但这里留一句指针）：
  ① **不要让任何库按 `nproc` 决定线程池**——KISS-ICP 的 TBB 和 onnxruntime 的 intra-op 池都会，
     超订 10 倍 ⇒ cgroup 周期性冻结 ⇒ **连进程里无关的单线程 numpy 代码都被拖慢 3–5 倍**；
  ② **不要在饱和的机器上做优化决策**——先跑 `vendor/sim_scripts/probe_cpu.py` 拿跨机可比定标。
