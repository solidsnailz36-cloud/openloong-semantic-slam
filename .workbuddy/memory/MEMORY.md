# SLAM 大师赛 — 项目长期记忆

> 只放**跨会话必须记住的决策与坑**。数字细节与完整链路在各文档里，按下面的索引去找，不要往这里搬。

## 细节索引（先读这些）
- **`docs/复赛实施方案.md`＝主文档**：§10–§17 = 五轮实施记录。**`docs/复赛自评与决赛差距.md`＝状态与差距**（含 §8 更新，判断"能不能进决赛"先读它）。
- `docs/HANDOFF_项目交接.md`（链路/不变量/回退链）、`docs/评审意见归因与修正方案.md`（初赛根因）、`docs/灾备与恢复清单.md`（"东西是不是丢了"先读它）、`docs/复盘_踩坑手册_小白向.md`（改前先备份到 `docs/_bak/`）、对外说明 `out/run5_keepdeg/`。
- 技能：`gazebo-classic-headless-container` / `camera-lidar-person-fusion` / `remote-job-and-snapshot` / `headless-rviz-recording` / `lidar-map-forensics` / `competition-dataset-triage` / `competition-submission-final-audit`。

## 家底 / 赛事现状（2026-09-28 清点，别重复清点）
- **唯一"丢了不可再得"的是 `D:\E-Downd\data.bag`**（初赛原始 2,532 MB / `/livox/lidar` 6,522 帧）。
  复赛数据集、官方仿真包、代码、成品、权重**全在本地**；云端独占的只有 `run/scans/`（恰好不需要）。
- 本机：GPU RTX 2060 / CPU 16 线程；`wsl.exe`·`wmic.exe` 被安全策略拦截。**交付三项都不需要 GPU**。
- 初赛 = **81 分**（点云分类不错 / **轨迹平滑度欠缺** / **RTK 置信度偏高**）。
  根因**不是**"最近邻"，是 **RTK 降级段的水平位置回退到里程计**（`docs/评审意见归因与修正方案.md` §5–§6）；
  修复包 `out/run5_keepdeg/` 未提交（窗口已关）。邮箱 `open@openloong.org.cn`，抄送 `luqinghua@openloong.net`。

## 复赛：仿真环境（Gazebo Classic + ROS 1 Noetic）
- 官方包 `gazebo_models.zip` / `src.zip`；`roslaunch vehicle_simulator house_vehicle.launch`。
  **场景 20×15 m 单层住宅（凹形）**；**行人模型是静态道具**（`<actor>` 出现 0 次）。
- 传感器：**VLP-16 `/velodyne_points` 5 Hz / 5,600 点**；相机 **HFOV 114.6° / 320×180 / 零畸变**（实测 ~11.6–12.0 Hz）。
  真值 **`/state_estimation` 200 Hz**（**只用于评分，绝不进产物**）+ `/registered_scan`；控制 **`/cmd_vel`（TwistStamped）**；
  **仿真器不提供规划器** ⇒ "场景探索"要自己写 frontier。
- **cam↔LiDAR 外参不是单位阵** + **LiDAR 不随车体旋转**（**且量程 40 m** ⇒ 一个视点基本看完屋子，
  见"★★ 这份仿真器上探索是假的"）→ 见下面两个 ★ 节。
- 云端 T4 `gpufree-container`：Ubuntu 20.04 + ROS Noetic + Gazebo 11.15.1 预装；80 核/376 GB；llvmpipe + `Xvfb :99`。
- **版本化副本 `vendor/sim_scripts/`**：`run_online.sh`（**唯一入口**）/ `kill_sim.sh` / `sim_up.sh` / `t4_state.sh` /
  `probe_cpu.py` / `probe_kiss_rotation.py` / `probe_coverage.py` / `put_pkg.sh` / `rmt.py` / `rr.sh`·`rr_proxy.sh` / `t4_diag.sh`…
  **凭据只放 `.workbuddy/tmp/ssh_env.sh`，不进 vendor。** 补件模型 `vendor/gazebo_models_patch/hokuyo/`（务必保留）。
- **容器重启会重置 `/etc/hosts`** ⇒ `models.gazebosim.org` 黑洞化**每次开机后要补一遍**，不补就是经典卡死
  （世界加载永久阻塞、`sim_time` 恒 0、**零报错**）。这是容器里唯一会丢的东西。

### 连不上远端时（2026-09-24 定型）
- **先跑 `bash vendor/sim_scripts/t4_diag.sh`**：一条命令出判决（本机 TCP → 平台主站 → **第三方节点带对照组**）。
- **TIMEOUT 有两种**：外面也连不上＝实例没起来；**外面连得上＝是我们这条路径到不了它** ⇒ 改地址/重启是**朝错方向使劲**。
  **对照组是硬要求**：同探 `1.1.1.1:<port>` 与 `8.8.8.8:<port>`，阈值 **≥2**（单个 OK 属噪声）。
- **2026-09-29 定性**：手机流量能开 Jupyter ⇒ 实例正常、**只有校园网这条路径到不了**（CERNET↔移动回程）。
  通道：① 手机热点；② 本机 v2rayN SOCKS `127.0.0.1:10808`，**必须全局模式**
  ⇒ `bash vendor/sim_scripts/rr_proxy.sh '<cmd>'`。`rmt.py` 支持 `SSH_SOCKS=host:port`（**SFTP 也能过**）。

## 复赛：算法栈 `r2slam/`（一套代码两个数据源）
- **纪律**：`config.py` + `sources/` 是全项目**唯一**允许知道"仿真/真实"的地方，分支写在 `config/*.yaml`；
  下游只认 `Frame`。**下游出现 `if sim` = 设计失败。** 唯一例外是 bag **路径**（属机器不属数据集）⇒ `R2SLAM_BAG_PATH`。
- 模块清单与职责见 `r2slam/README.md`（别在这里重复）。要点：**`labels.py` 是类别表的唯一定义源**
  （25 类 = 2 几何 + 23 类室内物体），导出 `label_table.yaml` 随 pcd 交付（FAQ 不提供 ⇒ **交付物必须自带**）。
- **`tools/online_pipeline.py` 是产出交付物的主程序**（一条循环里出两个视频 + pcd）：
  `--explore` / `--goto` / `--dump` / `--dump-grids` / `--npz` / `--encode` / `--person-only`。
  **入口 `vendor/sim_scripts/run_online.sh`（唯一入口）**。
- **`tools/package_submission.py` = 终审脚本**：逐条核对（视频≤2min / pcd 字段与字节自洽 /
  **pcd 里每个 label 都在表里** / 两源同表 / 包≤500MB），**任一条 FAIL 就非零退出**。
- **`nav.py` = 决赛半场**：`ObjectRegistry`（跨帧实例关联）+ `find_label`（中英同义词/最长匹配/模糊）
  + `PathPlanner`（已知可通行格上 A*，按半径膨胀）+ `SemanticNavigator`（与 FrontierExplorer 同接口）。
  **`explore.py` 现在也用同一个 `PathPlanner`** —— 探索与导航只差"目标从哪来"。
- 实测：仿真 5.000 Hz / 5,600 点；真实 10.328 Hz / 49,152 点。**入画率差异是视场比**，不是 bug。
- **交付三项**：仿真运行视频 + 真实运行视频 + **建图 pcd**（视频取更严口径 **≤2 min + 含可视化界面**）。
  四处换轨＝离线→在线、RTK→室内、19 类→**人物/可通行/物体**、新增**相机-LiDAR 融合**。
  **不搬 SFPNet**（19 类 SemanticKITTI 与"人物/可通行"不是一回事）——**放弃初赛资产本身就是泛化能力的论证**。
- **交付物与提交包**：`r2slam/out/{online_sim,online_real}/`（各 run.mp4 + map.pcd + label_table.yaml +
  summary.json + frames.csv）；包 = `提交包/…—【固态蜗牛队】—【张煜晨】—【156****6887】.zip`，
  131,022,899 B / md5 `70ae5a09a1dca9fb1d64b80395787e3b`，终审 **0 FAIL / 0 WARN**。
  **pcd 字段 `x y z label rgb`（17 B/点，rgb 由 label 推导，不导 intensity）**；真实侧 114.85 m / 407.5 m²。
- **bag 的记录时刻 ≠ 传感器时间** ⇒ **必须用 header.stamp**（否则 dt 成 0/0.5 s 交替，里程计发散，130 s 跑出 9,573 m）。
  **`offset_time` 从帧戳倒推** ⇒ `time_offsets_from: end`；猜错是把畸变**翻倍**。
- **远端 python 环境**：`slam_comp/venv`（`include-system-site-packages` 要改 `true` 才有 `rospy`；
  **`/root/sim/pylibs` 已废弃**——自带 numpy/cv2 会顶掉 venv 的版本）。
- **改完管线必须先两源各跑 5 帧冒烟**：一个"只在首帧有检出才触发"的 `UnboundLocalError`
  让仿真连跑三次都没报错，真实走廊第一帧就炸。
- **改函数签名后必须真跑一次，不能只编译**：本轮两次运行时错（`yaw` 未定义、`has_big` 用在定义前）
  `compileall` 都是绿的。

## 探索策略（四个坑，同一族，都已修 + 都有测试 + 详见方案 §16.5/§17）
1. **候选只按距离排 ⇒ 极限环**：586 s 只到过 15 个 1 m 格 / 1254 帧停在同格（地图一填满，
   身边不断冒出孤立前沿格：地面回波射程边缘、椅腿阴影）。→ `_frontier_sizes()` 连通分量 +
   `score = 簇大小/(1+距离)`；簇大小在连通环上退化时用 `min_target_range`（≥1 m）兜。
2. **★ 可达性必须是"连通分量"，不是"直线射线"**。射线＝假设机器人能飞：只有门口正对着才可能
   选中门后的前沿，门在侧面时门后整片区域**永远**被否掉（症状＝看得见门口、进不去）。
   正解：取**机器人所在可走连通域**，直线不通就 **A\* 规划 + 前视跟随**（复用 `nav.PathPlanner`，
   探索与导航只差目标从哪来）。实测西侧独立结构 47 格候选、同块、8.4–9.3 m 远 —— 老口径一次没去。
   **边界**：门口若从未被观测成地面，门后与主域不连通 ⇒ 判为走不到（比射线保守，但射线在那图上连候选格都没有）。
3. **★★ 地面掩码是"一层点"不是"地面"** ⇒ 谈"塞不塞得下机器人"前**必须先闭运算**。
   实测 146.7 m² 可通行分成 **57 块**；**第一帧 11.0 m² 分成 327 块**；0.22 m 腐蚀：不闭合 103.1 / 闭合 135.5 m²；
   第一帧更是 **0.00 m²** ⇒ 策略在第 1 帧宣布"探索完成"。**但闭运算不能穿过已知障碍**
   （1 格厚的墙会被填掉、两侧就"通"了）⇒ `close(free) & ~dilate(obstacle,3×3)`，只准填"未观测格"。
4. **停条件不能用"驱动 N 帧没长面积"**：跨 20 m 房子要 180 帧，阈值 40 帧＝**保证半路停下**
   （实测停下时还剩 516 格可达前沿 / 28 簇 / 最近 6.7 m² 在 1.2 m 外）。
   正解＝**"到过 N 个目标却没学到新东西"**（到达才携带信息），帧数只留 300 帧兜底。
   **判据落在"是否真的去过"，不是"过了多少帧"**（第 3 条老坑的同一条道理）。
- 探索报 done 后管线会 **break**，否则视频后段是定格画面。
- **覆盖率别用"面积/地皮面积"讲故事**：`vendor/sim_scripts/probe_coverage.py` 读回 `--dump-grids` 的栅格，
  算"可通行 / 塞得下机器人 / 从终点走得到 / 未知里哪块走得到 + 一张图"。
  **"146.7/~280≈52%"是错的分母**（280 是地皮、房子是凹的）；正解＝**走遍了可走的 89.5%，没有未进入的房间**。
- **★★ 这份仿真器上"探索"是假的**。逐帧导出来看：`turn` 464 / `drive` 369 帧（**56% 在转**）、
  `|cmd_w|>0.1` 占 **96%**、位置包围盒 **2.7×3.6 m**、净位移 1.74 m / 路径 25.97 m（**比值 14.9**）。
  根因＝LiDAR 世界对齐 ⇒ **转圈不产生任何新信息**；40 m 量程 ⇒ 一个视点基本看完屋子
  ⇒ **"探索"退化为"微动填地板掩码的洞"**。覆盖率 146.7 → **153.0 m²**（屋内未知 19.7 → 13.8 m²，−30%）
  是真的，但**不许讲成"探索能力提升"**。下一步＝**"边转边走"（画弧）而不是原地转**（一个参数 + 一条判据，未做）。
- **`barren_targets` 必须只在"剩下的都是小簇"时才允许停**：第一版无条件生效 ⇒ 270 帧就停、
  8.0 m² 开口没去、还剩 772 格可用前沿 —— **比它取代的旧判据更差**。
- 真值对照（`/state_estimation`）**位置 p50 0.20 m** ⇒ 位姿是可信的，上面的"停在 3 m 盒子里"是真的，不是里程计坏。

## 复赛：感知（"能否检出人物"＝ **能**）
- 检测器 `vendor/detector/tiny-yolov3-11.onnx`（35.5 MB，COCO **80 类**，person=0）+ **onnxruntime**；
  **不用 ultralytics**（要适配嵌入式计算平台）。12–36 ms/帧（10 Hz 预算 100 ms）。**未被训练或微调过。**
- **只取 23 类室内物体**（`labels.py` 的门限 `min_points`/`min_box_frac`/`min_conf` 都是物理量不是调参）。
  `detect.detect_mapped()` 返回**本项目的 label id**；规则＝**取网络自己的 top-1，只有叫得出名字才保留**
  （不"挑最好的已映射类"，否则 0.9 的 bicycle 会变成 0.2 的 chair），被丢弃的框数进 summary。
- **解码三坑（都静默出错）**：① 框是 **(y1,x1,y2,x2) 且已是原图像素**（当归一化会算出负高度）；
  ② **图内 NMS 的 `indices` 是空的**，必须自写 NMS；
  ③ `onnxruntime` scores `(80,N)` vs `cv2.dnn` `(N,80)` ⇒ **解码要轴序免疫**，`cv2.dnn` 后端弃用。
- **可通行（几何法、零训练）三个归零点**：① 障碍只取**离地 0.35–2.0 m 的带**（否则天花板把全图标成障碍）；
  ② **先闭运算补地面圆环缝、再对障碍膨胀**；③ 地面容差**随距离增长** `0.10+0.035·d`。
  真实地面 `z≈−1.18 m`；仿真 `z≈−0.79 m`。**两源阈值完全相同、没调参。**
- **融合硬判据**：三维位置**反投影回图像必须落回原框内**（真实侧 6/6 OK）。
  ⚠️ **但它抓不到"整体错一个角度"**（往返自消）；深度取框内点 **30 分位**。
- **真实走廊加类别几乎不增产出**（400 帧只多 1 个 chair）；**仿真是主战场**。

## ★★ 官方仿真器：LiDAR **不随车体旋转** + **外参不是单位阵**（最贵的两条）
- **`vehicleSimulator.cpp` 在移动 LiDAR 模型前把四元数 yaw 写死为 0**，而相机拿到完整姿态
  ⇒ LiDAR 被搬到车体位置但**朝向恒为世界对齐**。
  实测：我们的 yaw 全程 ±1.6°，真值 yaw 扫过 359°，指令累计转过 6656°。**KISS-ICP 报"没转"是对的。**
- 后果一：**航向不可观测** ⇒ 按航向控制的策略永远转不完。解药 `heading.py`：用**上一帧指令角速度 ×
  实测帧间隔**死推航向（轮式里程计语义），dt 上限夹住并计数。死推误差 **p50 1.85°**（原 79–89°）。
- 后果二：**相机—LiDAR 外参只在 yaw≈0 成立** ⇒ 正解 `p_cam = Rz(-yaw) @ Tcl @ p_lidar`；
  已按 `LidarSpec.rotates_with_base` 配置化（仿真 false / 真实 true）。
- 后果三（本轮补）：**转圈不产生任何新信息** ⇒ 见"★★ 这份仿真器上探索是假的"。
- **外参正解**：相机=光学系(z前/x右/y下)、LiDAR=机体系(x前/y左/z上)
  ⇒ `cam z←LiDAR x`、`cam x←−y`、`cam y←−z`，平移 0（真实侧同旋转 + 3 cm）。
  **"位置相同"≠"朝向相同"**：按单位阵投影，5,600 点里 **0 点入画**。
- **消歧方法（可复用）**：两候选只差绕光轴一个 yaw，而"地面在下"对左右翻转不敏感 ⇒ 必须用
  **机器人自己的真实位移**（驱动后取 `/state_estimation` 世界位移，用权威 `map→sensor` TF 转进 LiDAR 系，
  实测 `(1,0,0)`）。**目视"点阵在左边"是循环论证。** 仿真 TF **只有 `map→sensor` 一帧**。
- ⚠️ **"反投影回原框"自检抓不到外参错**：往返自消（实测 98.9% 通过，yaw 90–180° 区间 100%）。
- 诊断顺序：① 先查里程计（被自己的**弧度当度**打印坑了一次）→ ② `probe_kiss_rotation.py` 测 voxel
  （1.0 每步短 8–15°，0.5/0.3 正常）但**换 voxel 无效** ⇒ 不是参数 → ③ 读官方源码看到那一行。
- **★ 两阶段流程自身的坑**：`--npz` 复算**没有驱动器** ⇒ 上一帧指令恒 0 ⇒ 航向恒 0；
  而融合用的正是航向 ⇒ **复算出的每个物体都被绕机器人转过一个错误角度，而地图看着完全正常**。
  处置：`dump_frame(..., extra={"control_yaw": yaw})` 存进 npz 优先采用；summary 增
  `frames_missing_control_yaw` 并**大声报**。

## 复赛：真实数据集（`D:\E-Downd\openloong.zip` 1.97 GB）
- 4 个文件：`calib.yaml` / `LiDAR Map.pcd`(**先验图，仅墙体**，72×52×6.7 m) / `map_*.bin`(554 MB) / `three1.zip`(1.45 GB)。
- **标定齐全不用自己标**：鱼眼 `FishPoly` 1600×1296；`Tcl_0` 语义已解。**等距 `r=f·θ`（f=730.6）投影已验证**。
- **`three1` = 真机 ROS 2 bag**，130 s：`/odin1/cloud_raw`(1336) / `image/compressed`(1335 JPEG 鱼眼) / `imu`(559)。
  点云 **49,152 点/帧**，带 `offset_time` 0–0.094 s ⇒ **必须去斜**。**IMU 基本不可用**，赛题也不要求。
- `map_*.bin`（`MAPV0001`）厂商私有、**我方不需要解析**；**`LiDAR Map.pcd` 与仿真 house 不是同一场景**。

## 初赛遗留口径 + 已证伪（归档；细节见 `docs/HANDOFF_项目交接.md`、`评审意见归因与修正方案.md`）
- 类别编号用 mmdet3d SemanticKITTI 19 类 0 基（官方表 road=9，**差 1**）。
  `path.yaml` 的 `stamp` 是 bag 原生相对秒；初赛场地是**坡地**，交付物统一 **ENU**；
  `geo_trajectory.py` 只做 2D 水平拟合（`up` 带 0.883° 倾角）⇒ **"只改地图、轨迹不动"是错的**。
- **组委会材料互不覆盖** ⇒ 终审两份都读、按更严的执行（录屏时长吃过亏：主页 5 min vs 文档 2 min）。
- **地面三级细分不可恢复**（分界是到传感器的距离，不是材质）⇒ 归并是唯一诚实做法。**car = 自车**，应剔除。
- **平滑"改正量"是陷阱**：`e = r_local − P0` 含里程计抖动的负值，滤波会把抵消一起滤掉 ⇒ 抖动原样回到位姿。
  **改位姿一律 `PICK=1`；真要平滑就平滑位置。**
- **最近邻的时差是常量滞后（30 ms），不是锯齿来源**；**字段存在 ≠ 字段有效**（仿真 `time` 全 0）。
- 实时性根因**不是算法**，是**容器 CPU 配额被线程池超订**（`nproc`=80 是宿主机的，`cpu.max` 只给 8）；
  解药＝**显式限制每个库的线程池**，限线程**不改变轨迹**（表格 → 方案 §13/§14）。
  测量坑：循环同一批帧会让 ICP 虚快；**只看 p50/p90 会误判，要导出逐帧序列**；归因必须先做受控实验。

## 硬规则（只留没别处可查的；其余见技能，避免重复）
- **每个会话开头先补 PATH**：`export PATH="/c/Users/ZhuanZ/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:$PATH"`
  （Bash 工具本机缺 coreutils，`dirname`/`tail` 都会 command not found）。
- **含中文的路径**：存在性判断一律用 Python `os.path.exists`，不要用 Git Bash 的 ls/cd/find。
- **调远端 / 传文件前先** `export MSYS2_ARG_CONV_EXCL='*' MSYS_NO_PATHCONV=1` —— 否则 Git Bash 会把远端
  POSIX 路径改写成 Windows 路径，SFTP 报 **ENOENT「No such file」而远端明明存在**。
- 其余硬规则已固化在技能里，不在这里重复：`remote-job-and-snapshot`（Windows↔远端四坑、长任务脱 channel、
  快照、**§7.1 第三方节点+对照组**、**§9.1 任何库的线程池都要限 / §9.2 别在饱和机器上优化 / §9.3 落盘→复算 /
  §9.4 后台线程要上报失败**）、`gazebo-classic-headless-container`（pkill 自杀、`set +u`、`check_world_models.py`、
  只认 `sim_time`、`stdbuf`、catkin 0 字节 CMakeLists、**驱动三要求 + frontier 六要点**）、
  `competition-submission-final-audit`（"新的一件+旧的三件"、近似公式先怀疑公式）、
  `lidar-map-forensics`（RTK 只取 `status==2`、平滑"改正量"陷阱）。
- **同一文件多处修改必须串行发 Edit**（并行会互相覆盖）；`Write` 覆盖前先 `Read`。
- **★ 跨项目通用的两条测量纪律**：① **不要让任何库按 `nproc` 决定线程池**——超订 10 倍 ⇒ cgroup 周期性冻结
  ⇒ **连无关的单线程 numpy 代码都被拖慢 3–5 倍**；② **不要在饱和的机器上做优化决策**
  ——先跑 `vendor/sim_scripts/probe_cpu.py` 拿跨机可比定标。
- **★ 最普适的一条：自检通过 ≠ 正确**。"反投影回原框"是往返自消、抓不到整体错一个角度；
  "地图看着对"抓不到所有物体被转过一个角；两阶段复算"跑通了"抓不到它丢了航向；
  "两源类别表语义相同"抓不到 CRLF 与 LF 的差别。**判据要落在具体量上**
  （npz 里到底有没有那个数、同一份产物是否逐字节一致）。
