# SLAM 大师赛 — 项目长期记忆

> 只放**跨会话必须记住的决策与坑**。数字与推导在文档里。
> 这份文件要保持**短**：写不进来的细节属于 `docs/`。

## 索引
- **★★ `docs/交接_20261004_复赛感知精修.md` ＝当前进度交接，新会话第一份读它**（§22 是本轮）。
- **★ `docs/探索算法_v2_设计与参考.md` ＝探索/规划的全部现状与判据**（**§32 是本轮**；§24 有撤回指针）。
- 技能：`robot-policy-replay-ab`（**新会话先看它**）、`camera-lidar-person-fusion`、
  `gazebo-classic-headless-container`、`remote-job-and-snapshot`、`lidar-map-forensics`（待整理）、
  `competition-submission-final-audit`。

## 家底 / 赛事硬约束
- 唯一"丢了不可再得"的：`D:\E-Downd\data.bag`（初赛 2,532 MB，81 分，根因＝RTK 降级段水平位置回退到里程计）。
- **★ 官方答疑＝交付形态**：实时运行 + **整个数据集的视频** + 自建图 pcd；官方 pcd 只作提示；
  仿真 LiDAR yaw **不修**；输入只允许 **点云/图像/IMU**；**要输出建图 pcd + TF**；**建图 50% / 感知 50%**。

## 环境
- **仿真 = 本机 Docker 容器 `r2sim`**；挂载 `D:/E-Downd/sim_official -> /work`，容器内代码 `/work/r2slam_ws`。
  **改完本机必须 `bash vendor/sim_scripts/sync_to_container.sh`**，否则容器里**静默**跑旧代码
  （`PKG_PATHS` 是**手工白名单**，新增模块不列进去容器会 ImportError）。
- **逐帧点云采集 `cap_*` 与 `grids_*` 全在容器里**（`run_repeat.sh` 会 dump；cap **不**拷回宿主机）⇒
  想做只读复算，**在容器里跑**（`docker cp` 脚本 + 读 `/work/cap_*`、`/work/out_*`、`/work/grids_*`）。
- **真实侧在本机 `.venv` 跑**（1427 帧 ≈ 6–8 min）。`wsl.exe`/`wmic.exe` 被安全策略拦截。

## ★★ 官方仿真器两条最贵的坑
1. **LiDAR 不随车体旋转**（yaw 写死 0）⇒ **航向不可观测**，`heading.py` 用上一帧指令角速度×dt 死推；
   **转圈不产生新信息**。
2. **相机—LiDAR 外参只在 yaw≈0 成立** ⇒ `p_cam = Rz(-yaw) @ Tcl @ p_lidar`（按 `rotates_with_base` 配置化）。
- ⚠️ "反投影回原框"自检**抓不到**这两条（往返自消）；有效判据＝与 `/gazebo/model_states` 比对。
- **★ 仿真硬精度上界**：无 IMU、无里程计话题 ⇒ **单跑之间不能比分数**。

## 感知 / 地图（定论）
- 检测器 = **YOLO26n ONNX**（默认已切）；回退 `--model vendor/detector/yolo11n.onnx`。
  ⚠️ 模型不入库，`sync_to_container.sh` 的 `ROOT_PATHS` 必须列上；换模型**必须**跑
  `tools/check_onnx_parity.py`。`labels.py` = 类别表唯一源。
- 语义地图：`绿 = ground_ever & ~obstacle`；**格子只有"有回波落进去"才算 seen** ⇒ 真开着的门应画黑。
- **障碍层四条规则**（`worldmap.py`）：`OBSTACLE_MIN_VOTES=2`；`OBSTACLE_RELEASE_FRAMES=2` +
  `OBSTACLE_RELEASE_RATIO=0.30`；**`_on_path` 豁免**；`blocked()=obstacle|obstacle_now`（安全层）。
  ⚠️ **别再退回 `votes<12` 那种票数上限**（它把**两扇门**都封死过）。

## 算法栈不变量
- `config.py` + `sources/` 是唯一允许知道"仿真/真实"的地方；**下游出现 `if sim` = 设计失败**。
- `tools/online_pipeline.py` = 主程序；`tools/package_submission.py` = 终审（**FAIL 绝不许写 zip**）。
- **★ 当前包 = `提交包/…【固态蜗牛队】…zip` = 36,204,882 B / md5 `ec8260d28de6590033be36daca247b6c`**
  （终审 0 FAIL/0 WARN；**是 v6 门限的产物，v7 以后与探索的改动都没进包**）。
- **`--dump-grids` 必须写全**（含 `ground_ever`/`obstacle_votes`/`traj`）⇒ 交付产物自己就是仪器。
- bag 记录时刻 ≠ 传感器时间 ⇒ 一律 `header.stamp`；`--npz` 复算没有驱动器 ⇒
  `dump_frame(extra={"control_yaw": yaw})` 必须存；**npz 存的是未去斜原始点云**。
- 改完管线**两源各 5 帧冒烟**；改签名后**必须真跑一次**；**传输/比对比 md5**。

## 探索算法：架构 + **已判死的方向**（重开条件都写在这里）
**架构**：规划 = A\*（8 连通、按半径腐蚀 + 障碍膨胀）；跟随 = 三档油门 + **视线可达前视点**；
探索 = 前沿簇 + 角扇区 + `score = size/(1+d_start)`（**离起点**）+ `sweep`（去没到过的地板）+
`island`（门口探针）+ `shadow`（死角观察位姿）。**★ 目标 = "观测位姿"**（`target_standoff_m=0.80`）。
`progress_eps_m=0.02` 必须**远小于** `sweep_stuck_improve_m=0.5`。**`n_frozen_frames` 必须永远 0。**
- **★ 每个"什么都没发生"的判据都要说清盯的是哪个量**（帧数/直线/看到的地面都曾盯错）。

**仍然成立的跨轮结论（不要再重新发现一遍）**
1. **单跑不能当战果**：同构建 4 次跑逐房间极差 ≈ 跨构建极差 ⇒ **验收一律看 N≥3 的分布**。
2. **"抽签"的机理＝1e-7 m**：策略是输入的确定函数 ⇒ 不要去查 random seed，修法是**排序量化**。
3. **第一次分叉在第 0 帧**：首帧地图只有 11 m²，就在信息最少时定了 6 m 级目标。
4. **★ 候选集本身就是全屋尺度的**：基准臂 `sep_min_med = 6.33 m`、`az_span_med = 157°`
   ⇒ **排序层根本没有"局部"可言**，任何改变榜首的项必然长出米级位移。这是 §23–§29 的共同根因。
5. **覆盖 ≈ 停留 ⇔ 被选上**：corr(覆盖, 框内停留秒)=0.762；**"进去待很久却 0%"的反例 0 个**。
   Discover **已被排除**（看到 40/40、可达 39/40、进候选 39/40；13 个 0% 里"从未候选" 0 个）。
   40 个房间-跑次只有 22 个曾成为目标 ⇒ **瓶颈在"选谁"，不在"能不能"。**
6. **顺路可见性是真的且可加总**（§32）：patch-first 每跑 **59.9 m²（22.7%）**的首见地面落在
   非目标房间；**损失高度集中在"左上"**（19.0 → 0.5 m²）。⚠️ 但机制**不是"路过房间"**，是
   **走廊位姿的视距** —— 左上那 20 m² 是**站在走廊一眼看清**的（站进房内只首见 0.4 m²）。

**已判死（附重开条件）**
- **候选级局部效用 `local_u` 整族**（§23–25）：默认 λ=0。它要买的"门前还剩多少"是 0.5–2 m²，
  而绕路代价是米级 ⇒ **买不起**。重开条件：候选集变得**真的局部**（见 `sep_min_med` 门槛）。
- **region 级饥饿债务 `region_lambda`**（§27）：默认 0。2 m 格分不出房间与走廊，
  改写落点**进房间框 0 次**、3 次把目标搬出房间。重开条件：`sep_min_med ≲ 5 m` 之前不再重启。
- **region reachability gate `rfirst_gate`**（§29）：假设被自己的实验证伪 ——
  `region_unreachable` 全量 **0.00**、`n_bad` 取值集合 `{0}`、R0/R1 md5 相同 ⇒ **no-op**。
  重开条件：换成有真死区的地图（case 32 在）。
- **region-first 两级决策 `rfirst`**（§31–32）：默认 False。平均 ↑ 但 `C_min` 塌
  （左上 4/4 `undefined`）；真因是**它把机器人锁在房子东半部**（`min pose_x` −2.6 vs baseline −12.4）。
  重开条件：能说清"为什么它不会切断去某侧的路线"，且**必须有可见性论证**，不是再调区域分数。
- **`debt_gain_m2`、黑名单、进度期限**：实测 no-op 或更差，不再重试。

## ★★ 指标：比率型必须先查分母
- **权威口径 = `out/_room_cover2.py`**（分母 = **单调**的 `ground_ever ∩ 框`；< 2 m² 记
  **"定义不了"**）。**`undefined` = 从未被 LiDAR 扫到过**，不是"覆盖低"。
  旧 `_room_cover.py` 的分母取"末帧可达格子"，会被累积障碍层封门口 ⇒ **10/40 个房间-跑次
  分母 < 2 m²**、0/0 印成 0%（§24 的 λ 结论因此撤回）。
- **新硬规则：报任何百分比必须同时报分母，或用单调量做分母。**

## ★★ 硬门槛：Decision-effect gate（进仿真前必须在 replay 上过门）
0. **★ Gate −1（最先做）：对照臂真的关掉了吗？** 看**对照臂与基准的产物 md5**是否一致。
   实测踩过：`--arms "F0:rfirst=False"` 的 `False` 被解析成**字符串** ⇒ `bool("False")==True`
   ⇒ 对照臂开着新功能跑完整场，**无报错无警告**。三个参数入口已统一到 `r2slam/overrides.py`。
1. **它作用的那一层有对象**：`p_ge2 ≥ 0.5` 且 `sep_min_med ≥ 0.5 m` 且 `az_span_med ≥ 45°`。
   **排序类改动先量基准的 `sep_min_med`/`az_span_med`**；`≳5 m` ⇒ 别动排序层。
2. **它确实改变了决策**：`swaps / decisions ≥ 10%`（`decisions` 只数**真的被采纳**的）。
3. **方向条件**：改写必须朝"该去的地方"去，`→ less` 必须为 0。
4. **★ 空间局部性**：每次改写记 **位移 / 两侧旅行成本 / `Δu` / `Δu/Δd`**。
   **`max switch distance > R`（跨区域项 > 5 m）⇒ 直接失败。** **只报方向的仪器一律不合格。**
5. 任一不满足 ⇒ **先修候选集或换一层，不许跑仿真**。
6. **负结果也要落成规则**：被否决的方向 ⇒ 代码与用例保留（记录 + 检验台），并**预先写下判死条件**。
**工具**：`tools/replay_explore.py --arms "name:k=v"`（**一次感知、N 臂逐帧交错**；分开重放只有
5e-7 m 可复现，而**那足以翻转离散选择**）；`vendor/sim_scripts/run_repeat.sh TAG N`；
`out/_discover_timeline.py`；`out/_room_cover2.py`；`out/_region_gate.py`（只读产物）；
`out/_traverse_{attr,obs,summary}.py`（顺路可见性，**观测级那支要在容器里跑**）。

## 硬规则（工程）
- **会话开头先补 PATH**：`export PATH="/c/Users/ZhuanZ/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:$PATH"`。
- **含中文路径**：存在性判断用 Python `os.path.exists`，不要用 Git Bash 的 ls/cd/find。
- 同一文件多处修改**串行**发 Edit；`Write` 覆盖前先 `Read`。
- **★ 最普适：自检通过 ≠ 正确**。判据要落在**具体量**上（"机制跑了" ≠ "机制起作用了"）。
- **★ 只读复算流水线必须有跨全流程的守恒量做校验，不能只有端点校验**。
  实测踩过：`prev = wmap.ground_ever`（**引用而非拷贝**）⇒ 增量恒空，产出"11 m²"而真实 269 m²，
  **而末帧校准当时是过的**。抓住它的是总量守恒。
- **★ commit message 含反引号不要用 `-m "..."`** ⇒ 用 `-F 文件`。
- **参数扫描不许改源码**：`--explore-set KEY=VALUE`（在线）/ `--set`（重放），覆盖值进 `summary.json`。
- **影子重放有边界**：跟**记录轨迹**走 ⇒ **测决策、不测覆盖**；且**决策序列无法完全复现**
  （36/40 一致，其余某时刻起走岔）⇒ 轨迹/地图侧可信，**选择侧取实跑记录**。
