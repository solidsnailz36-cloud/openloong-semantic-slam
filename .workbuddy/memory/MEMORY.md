# SLAM 大师赛 — 项目长期记忆

> 只放**跨会话必须记住的决策与坑**。数字与推导在 `docs/`。**这份文件要保持短。**

## 索引
- **★★ `docs/交接_20261004_复赛感知精修.md` ＝当前进度交接，新会话第一份读它**（§34 是本轮）。
- **★★ `docs/K2多跑次验证_20261007.md` ＝最新的完整实验报告**（设计文档 §44）。
- **★ `docs/探索算法_v2_设计与参考.md` ＝探索/规划的全部现状与判据**（**§44 是本轮**；§24 有撤回指针）。
- **★ `docs/探索算法_成熟方案对标.md` ＝与 FUEL/NBV/ASE/ROS2 的**源码级**对照（设计文档 §37）。
- 技能：`robot-policy-replay-ab`（**新会话先看它**）、`camera-lidar-person-fusion`、
  `gazebo-classic-headless-container`、`remote-job-and-snapshot`、`competition-submission-final-audit`。

## 家底 / 赛事硬约束
- 唯一"丢了不可再得"的：`D:\E-Downd\data.bag`（初赛 2,532 MB，81 分，根因＝RTK 降级段水平位置回退到里程计）。
- **★ 官方答疑＝交付形态**：实时运行 + **整个数据集的视频** + 自建图 pcd；官方 pcd 只作提示；
  仿真 LiDAR yaw **不修**；输入只允许 **点云/图像/IMU**；**要输出建图 pcd + TF**；**建图 50% / 感知 50%**。

## 环境
- **仿真 = 本机 Docker 容器 `r2sim`**；挂载 `D:/E-Downd/sim_official -> /work`，容器内代码 `/work/r2slam_ws`。
  **改完本机必须 `bash vendor/sim_scripts/sync_to_container.sh`**，否则容器里**静默**跑旧代码。
  ⚠️ **`PKG_PATHS` 是手工白名单**：新增模块不加进去 ⇒ 容器 ImportError；
  **`tools/replay_explore.py` 不在白名单里 ⇒ 必须单独 `docker cp`**。
- **逐帧采集 `cap_rf0r1..cap_rf1r4`（8 份）与 `grids_*` 全在容器里**，**不拷回宿主机** ⇒
  只读复算**一律在容器里跑**（`docker cp` 脚本 + 读 `/work/cap_*`、`/work/k2/*`）。
- **真实侧在本机 `.venv` 跑**（1427 帧 ≈ 6–8 min）。`wsl.exe`/`wmic.exe` 被安全策略拦截。

## ★★ 官方仿真器两条最贵的坑
1. **LiDAR 不随车体旋转**（yaw 写死 0）⇒ **航向不可观测**，`heading.py` 用上一帧指令角速度×dt 死推；**转圈不产生新信息**。
2. **相机—LiDAR 外参只在 yaw≈0 成立** ⇒ `p_cam = Rz(-yaw) @ Tcl @ p_lidar`（按 `rotates_with_base` 配置化）。
- ⚠️ "反投影回原框"自检**抓不到**这两条（往返自消）；有效判据＝与 `/gazebo/model_states` 比对。
- **★ 仿真硬精度上界**：无 IMU、无里程计话题 ⇒ **单跑之间不能比分数**。

## ★★ 一个可搬运的"真值"通道（2026-10-07 发现）
`/root/.gazebo/models/house.world` 把每个 model/link/collision 的**世界位姿都烘焙好了**。
碰撞体：**box 20 + mesh 42 + cylinder 6**（`youbot` 要排除）。⇒ 能重建**完整真值占用栅格**。
- **COLLADA 是 XML，不需要三方库**（容器里 trimesh/meshio/open3d/stl **全无**）：
  走 `<vertices>`→`semantic=POSITION` 的 `<source>`→`float_array`；
  `<p>` 按 `stride = max(offset)+1` 拆（**别把所有 `<source>` 当顶点，NORMAL/UV 维度不同会崩**）；
  `<unit meter="0.01">` 要乘。
- **`kitchen_table` 是 STL 不是 DAE**；**`up_axis` 声明不可信**（`ikea_stefan_chair` 声明
  `Y_UP` 但数据是 Z-up，照声明翻会让它**离地悬空、静默贡献 0 格**）⇒ 按"**底部贴地**"自适应。
- **地图系 = 世界系**（FFT 互相关最优平移 ≈ youbot 出生位姿的相反数）。
  对齐用**距离型**目标，**别用薄墙的 F1**（0.15 m 厚的墙对亚格误差极敏感，距离是线性响应）。
- **判定口径**：分母只用"**客观测到的那部分**"（`|真值物体 ∩ A.obstacle|`）⇒ 不依赖整体对齐精度。
- 工具：`r2slam/out/_truth_geo.py`。

## 感知 / 地图（定论）
- 检测器 = **YOLO26n ONNX**；回退 `--model vendor/detector/yolo11n.onnx`。模型不入库，
  `ROOT_PATHS` 必须列上；换模型**必须**跑 `tools/check_onnx_parity.py`。`labels.py` = 类别表唯一源。
- 语义地图：`绿 = ground_ever & ~obstacle`；**格子只有"有回波落进去"才算 seen** ⇒ 真开着的门应画黑。
- **障碍层规则**（`worldmap.py`）：`OBSTACLE_MIN_VOTES=2`（**数帧数**）；`OBSTACLE_LO/HI = 0.35/2.00`；
  `OBSTACLE_RELEASE_FRAMES=2` + `OBSTACLE_RELEASE_RATIO=0.30`；`_on_path` 豁免；
  **`blocked() = obstacle | obstacle_now`（安全层）**。
  ⚠️ **别再退回 `votes<12` 那种票数上限**（它把**两扇门**都封死过）。

## 算法栈不变量
- `config.py` + `sources/` 是唯一允许知道"仿真/真实"的地方；**下游出现 `if sim` = 设计失败**。
- `tools/online_pipeline.py` = 主程序；`tools/package_submission.py` = 终审（**FAIL 绝不许写 zip**）。
- **★ 当前包 = `提交包/…【固态蜗牛队】…zip` = 36,204,882 B / md5 `ec8260d28de6590033be36daca247b6c`**
  （终审 0 FAIL/0 WARN；**是 v6 门限的产物，v7 以后与探索/地图的改动都没进包**）。
- bag 记录时刻 ≠ 传感器时间 ⇒ 一律 `header.stamp`；`--npz` 复算没有驱动器 ⇒
  `dump_frame(extra={"control_yaw": yaw})` 必须存；**npz 存的是未去斜原始点云**。
- 改完管线**两源各 5 帧冒烟**；改签名后**必须真跑一次**；**传输/比对比 md5**。

## 探索算法：架构 + 已判死方向
**架构**：规划 = A\*（8 连通、按半径腐蚀 + 障碍膨胀）；跟随 = 三档油门 + **视线可达前视点**；
探索 = 前沿簇 + 角扇区 + `score = size/(1+d_start)`（**离起点**）+ `sweep`（去没到过的地板）+
`island`（门口探针）+ `shadow`（死角观察位姿）。**★ 目标 = "观测位姿"**（`target_standoff_m=0.80`）。
`progress_eps_m=0.02` 必须**远小于** `sweep_stuck_improve_m=0.5`。**`n_frozen_frames` 必须永远 0。**
- **★ 每个"什么都没发生"的判据都要说清盯的是哪个量**（帧数/直线/看到的地面都曾盯错）。

**仍然成立的跨轮结论**
1. **单跑不能当战果**：同构建 4 次跑的逐房间极差 ≈ 跨构建极差 ⇒ **验收一律看 N≥3 的分布**。
2. **"抽签"的机理＝1e-7 m**：策略是输入的确定函数 ⇒ 别查 random seed，修法是**排序量化**。
3. **排序层根本没有"局部"可言**（§23–29 全族根因）：基准 `sep_min_med = 6.33 m`、`az_span_med = 157°`；
   **先量这两个数，再动排序层**。
4. **覆盖 ≈ 停留 ⇔ 被选上**：corr = 0.762、反例 0 个；Discover 已排除 ⇒ **瓶颈在"选谁"。**
5. **顺路可见性是主机制、不是附加收益**（§32/§33）：机制是**走廊位姿的视距**。
6. **距离的除法形式压倒价值差异**（§33）；但（§38）**在线那个 `V` 从来不结算**才是主因。
7. **不要用累积 `obstacle` 层算可视域**（§33）：首撞距离中位 0.20 m，**它不是墙图**。
8. **分离变量**（§34）：`rfirst`+`K=4`+`sectors` 的复合里 **`K` 才是主因** ⇒ **A/B 之前先数清两臂差几个变量。**
9. **成熟系统不拿"一坨未访问地板"当决策对象**（§37）：FUEL/ROS2 用 **frontier**、NBV 用 **RRT 采样点**、
   ASE 用 **轨迹**；**`sweep`（patch）无对应物却是实跑主力**；
   **"价值 ÷ 距离"这个形式只有我们在用**。
10. **一个 case 先问它在总体里占多少**。
11. **离线 A/B 里"价值"必须按决策点时刻结算**（§38）：静态 `V` ⇒ 塌缩看起来最优；`ΣV` 是假指标。
12. **要"降级到 X"之前，先确认 X 不是已经失败的那一层**（§39）。
13. **救援机制"跑了" ≠ "有用"**（§39）：**验收必须含"救回到哪里去"。**
14. **"地图不增长"不是失效判据，"原地不动"才是**（§39）：真失效是 `backoff` + `|cmd_v|`=0。
15. **"少走了 N m²"先问是"走得差"还是"被提前停掉"**（§40）：比较跑次必须**按各自起点做时间对齐曲线**。
16. **★★★ 第一不可逆分叉在**地图层**，不在候选层**（§41 修正 §40）：`rf1r3` 的 `reach` 在
    t≈66–69 s 一次性塌掉（143.3 → 12.9 m²），同期 `blocked` 只涨 0.8 m² ⇒ **两跑建了同样的图，
    可达面积差 11 倍。**
17. **2D 栅格分不出"地面"与"它正上方的悬空结构"**（§42）：障碍证据会清零 `_ground_after`；
    `seen_mask` 只处理**天花板**（h > 2.0）。**§24 与本次是同一族的镜像**（一头 `votes<12` 永久封门，
    一头 2 票门槛 + `0.3×votes` 比例释放让门**先通后封**）⇒ **要修的不是阈值，是"什么算障碍证据"。**
18. **★ 凡是"世界坐标 → 格索引"的格表，必须用 `floor` + 真实 extent，并先验命中**（§42）：
    格心落在**半整数**上，`round` 的 banker's rounding 会把一半的格送偏一格。
    第一次消融差异 **0 格、无报错** —— 靠"消融必须改变点什么"这个守恒式才发现。
19. **★★★ 换个"量"常常就是分离器**（§43）：真结构是**点云簇**（真墙 p25 6 / 中位 12 / p75 26），
    擦边是**单点**（门内 91.8% 单点，`h` 中位 **1.95–1.99**，紧贴 `OBSTACLE_HI=2.00`；
    真墙 h 是 0.43→1.55 的立面）⇒ **高度带是现成的分离器。**
20. **★★★ K=2（`--map-min-points 2`）判死**（§44，8 采集 × 2 臂多跑次）：
    - **释放 15.9–29.5%（中位 20.9%），8/8 个 run 普遍；新增障碍恒 0。**
    - **收益真实但有反例**：A 的 5 次 `reach` 塌陷 B 保住 4 次（rf1r3 末 reach 12.8→150.6），
      但 **rf1r2 上一分钱没赚**（A/B 都塌 2 次、末 reach 都 ≈11.5）。
    - **真值几何口径给出本轮最重要的分离**：**墙只丢 1.6–9.9%（中位 4.3%），
      家具丢 22.4–87.1%（中位 64.4%）—— 相差 15 倍。**
      **`kitchen_table` 在 6/8 个 run 被削 61–100%**；rf0r1 上它和 `sofa` 都是 **100%**。
      墙：内洞 2 格 + 开道 6 格（0.08 m²）；家具：内洞 2 + 开道 81 格。
    - **判据**：P2/P3/P4 **8/8 FAIL**；P1 6/8；P6 1/8；P5 0/8 ⇒ **暂停。**
    - **根因**：K=2 分的是**回波密度**不是**真实性**；**桌椅沙发在远处/掠射角本来就只有每帧 1 点。**
    - **安全层没坏**：`obstacle_now` 在过滤**之前**写好 ⇒ K=2 丢的是**记忆**不是当帧检测，
      危害是"规划以为能走"，**不是撞上去**。

**已判死（附重开条件）**
- **候选级局部效用 `local_u` 整族**（§23–25）：默认 λ=0。重开条件：候选集变得**真的局部**。
- **region 级饥饿债务 `region_lambda`**（§27）：默认 0。重开条件：`sep_min_med ≲ 5 m`。
- **region reachability gate `rfirst_gate`**（§29）：实测 no-op（`region_unreachable` 全量 0.00）。
  重开条件：换成有真死区的地图。
- **region-first 两级决策 `rfirst`**（§31–32）：默认 False。真因是**它把机器人锁在房子东半部**。
  重开条件：能说清"为什么它不会切断去某侧的路线"，且**必须有可见性论证**。
- **`V/(1+d)` → `V·exp(-λd)`**（§38）：**不是判死，是当前不值得做**。分水岭 λ≈0.18。
  **重开条件：候选层修好之后再回来看。**
- **`fb_relax`（零候选时逐级放宽）**（§39）：默认 0（精确 no-op）。**不开** ——
  救回 9 次里 7 次去同一个死点、`bl_places` 3→11。重开条件：在"别处还有可去的候选"的数据上。
- **`OBSTACLE_HOLD=total`（障碍证据永不清零）**（§43）：释放 16.56 m²，90% 属 ≥50 格大结构、
  452 格带 100+ 票 ⇒ **把真墙一起放掉**。`suspend`/`suspend_vote` **完全无效**（判据方向反了：
  门内"同帧 both"=0，真墙反而 228–258）。
- **`debt_gain_m2`、黑名单、进度期限**：实测 no-op 或更差，不再重试。

## ★★ 指标：比率型必须先查分母
- **权威口径 = `out/_room_cover2.py`**（分母 = **单调**的 `ground_ever ∩ 框`；< 2 m² 记
  **"定义不了"**）。**`undefined` = 从未被 LiDAR 扫到过**，不是"覆盖低"。
- **硬规则：报任何百分比必须同时报分母，或用单调量做分母。**

## ★★ 硬门槛：Decision-effect gate（进仿真前必须在 replay 上过门）
0. **★ Gate −1（最先做）：对照臂真的关掉了吗？** 看**对照臂与基准的产物 md5**是否一致。
   实测踩过：`--arms "F0:rfirst=False"` 的 `False` 被解析成**字符串** ⇒ `bool("False")==True`
   ⇒ 对照臂开着新功能跑完整场，**无报错无警告**。三个参数入口已统一到 `r2slam/overrides.py`。
1. **它作用的那一层有对象**：`p_ge2 ≥ 0.5` 且 `sep_min_med ≥ 0.5 m` 且 `az_span_med ≥ 45°`。
2. **它确实改变了决策**：`swaps / decisions ≥ 10%`（只数**真的被采纳**的）。
3. **方向条件**：改写必须朝"该去的地方"去，`→ less` 必须为 0。
4. **★ 空间局部性**：每次改写记 **位移 / 两侧旅行成本 / `Δu` / `Δu/Δd`**。
   **`max switch distance > R`（跨区域项 > 5 m）⇒ 直接失败。只报方向的仪器一律不合格。**
5. 任一不满足 ⇒ **先修候选集或换一层，不许跑仿真**。
6. **负结果也要落成规则**：被否决的方向 ⇒ 代码与用例保留，并**预先写下判死条件**。

**工具**：`tools/replay_explore.py --arms "name:k=v"`（**一次感知、N 臂逐帧交错**；分开重放只有
5e-7 m 可复现，**而那足以翻转离散选择**）；`vendor/sim_scripts/run_repeat.sh TAG N`；
`out/_room_cover2.py`；`out/_discover_timeline.py`；`out/_region_gate.py`；
`out/_traverse_{attr,obs,summary}.py`；`out/_rep_rules.py`；`out/_vantage_ab.py`；
`out/_fallback_probe.py`；`out/_rf1r3_fork.py`；`out/_door_series.py`；`out/_door_width.py`；
`out/_door_ablate.py`；`out/_hold_ab.py`。
**2026-10-07 新增**：`out/_k2_runs.sh`（8 采集 × 2 臂批量 replay）、
`out/_truth_geo.py`（**真值几何栅格**：box + COLLADA + STL 三解析 + 自适应朝向 + 对齐拟合 +
按物体拆分）、`out/_k2_ab.py`（5 指标 + 6 判据 + 真值取证）；产物快照 `out/k2/`。
replay 已加 `--probe-cells` / `--probe-points` / `--ablate-cells` / `--map-hold` / `--map-min-points`。

## 硬规则（工程）
- **会话开头先补 PATH**：`export PATH="/c/Users/ZhuanZ/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:$PATH"`。
- **含中文路径**：存在性判断用 Python `os.path.exists`，不要用 Git Bash 的 ls/cd/find。
- 同一文件多处修改**串行**发 Edit；`Write` 覆盖前先 `Read`。
- **★ 最普适：自检通过 ≠ 正确**。判据要落在**具体量**上（"机制跑了" ≠ "机制起作用了"）。
- **★ 只读复算流水线必须有跨全流程的守恒量做校验**，不能只有端点校验。
  实测踩过：`prev = wmap.ground_ever`（**引用而非拷贝**）⇒ 增量恒空，产出"11 m²"而真实 269 m²。
- **★ commit message 含反引号不要用 `-m "..."`** ⇒ 用 `-F 文件`。
- **参数扫描不许改源码**：`--explore-set KEY=VALUE`（在线）/ `--set`（重放），覆盖值进 `summary.json`。
- **影子重放有边界**：跟**记录轨迹**走 ⇒ **测决策、不测覆盖**；且**决策序列无法完全复现**
  （36/40 一致）⇒ 轨迹/地图侧可信，**选择侧取实跑记录**。
  ⚠️ replay 位姿来自 KISS-ICP 读点云、`(v,w)` 不回馈 ⇒ "同一份 npz = 同一条轨迹"是**结构性成立**的。
