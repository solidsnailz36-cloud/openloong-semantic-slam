# SLAM 大师赛 — 项目长期记忆

> **只放跨会话必须记住的**：判死清单、环境硬约束、指标口径、最新结论。
> **数字与推导在 `docs/`；流程细则与踩过的坑在技能 `robot-policy-replay-ab`（新会话先看它）。**

## ★ 文档地图（新会话先看这里）
- **★★★ 第一份读**：`docs/交接_20261008_当前状态与后续计划.md`（~740 行）—— 当前状态 + 后续工作项（P0/P1/P2）。
- **★ 当前主线**：`docs/语义地图朝向与灰斑修复_20261009.md` —— §6 渲染↔规划一致性 / §7 靠墙灰带 /
  §10 `OBSTACLE_GROUND_RATIO` A/B / §11 3D 检测调研 / **§12 边缘算力重排**。
- `docs/B3成本剖析_20261008.md` §11–§13：**B3 收线记录（可停投）**。
- `docs/交接_20261004_复赛感知精修.md`（2278 行）：10-04~10-07 逐轮细节。
- `docs/HANDOFF_项目交接.md`（605 行）：初赛时代；**§9 复赛计划已过时**，其余有效（**§5.2 类别表 / §8 已知限制**必读）。
- `docs/复盘_踩坑手册_小白向.md`（3340 行）：为什么会错、当时怎么查出来的（复习用）。
- `docs/探索算法_v2_设计与参考.md`（413 KB）：§47–§71 地图层/探索层全部实验记录。
- ⚠️ 技能 `robot-policy-replay-ab/SKILL.md` 已 **187 KB**，单文件过大，**该按主题拆**（未做）。

## 索引（最新结论；推导全在 docs，这里只留结论）
1. **§74 漂移**：**真实侧不显著**（厂商独立对照 ATE p50 **0.276 m**/102 m）；**仿真侧＝"少数几帧被打偏且
   永不恢复"**，根因＝**官方仿真器航向不可观测（LiDAR 不随车体转）**，**不是 `heading.py` 的 bug**。
   **不是换包造成。** ⇒ 详见 `docs/漂移归因_20261008.md`。
2. **★★★ B3 已收线，可停投**（`docs/B3成本剖析_20261008.md` §11–§13）：
   **成本门 PASS**（`ms_map` p90 OFF ≤31.4 / ON ≥94.9 ms 完全不重叠；**RTF 完全重叠** ⇒ 「打崩 5 Hz」**不复现**；
   ⚠️ **只看 p90，别看 max** —— max 是单帧噪声主导）。
   **效果门 FAIL、撤回**：里程 **128.90 → 82.43 m**（p=0.0047）、`ms_map` p90 20.75 → 108.6（p=0.0002）、
   可达 154.3 → 124.4 m²。**不是"慢"**（帧率只掉 8%），而是 `arrivals`/1k **8.66 → 0.72**、`plan_failures` **×12.7**
   ⇒ **规划器在走不通的地图上反复失败**。**归因＝我加的成本上界 `max-eval` 有语义缺陷**
   （扣住的格变成 `ground` ⇒ 被当可通行地板，而 `blocked()` 仍说不通）；
   **消融"无上界"根本跑不完** ⇒ **两条路都堵死**。**`OBSTACLE_TOPO_FUSE_M2` 默认仍 0.0。**
3. **★★★★ 渲染↔规划一致性**（§6，`b0bd756`）：地图曾有**空类 `seen & ~ground & ~obstacle`（6466 格）**
   ——看到过、无障碍证据、也没测到地面（回波 64.9% 在障碍高度带但**凑不满 2 票**；相机证实是空地）。
   我曾把它画成障碍灰、也在渲染层打补丁 ⇒ **一个格两个答案**（boss：「很严重」）。
   **定论：`GROUND_COVER_INFORMATIVE`（`ground_ever |= seen & ~obstacle & ~object`）
   ＋ 删掉渲染补丁，`render` 只画 `ground`（＝ `free_space` 读的同一个数组）。
   ★ 画面可以有颜色，不能有自己的判据。** 真实侧第三类 6466→1443、`obstacle` 一格未动、
   **决策列唯一变化是 `traversable_m2`** ⇒ 行为零影响。仿真 A/B n=3/3 **无分离**。
   ⚠️ **默认仍 False**；`summary["map"]["seen_not_classified_cells"]` 就是这个一致性指标（**归零即一致**）。
4. **★★★★ 靠墙灰带 / 绿→灰**（§7/§10，`fe9e0a3`）：**`ground = ground_ever & ~obstacle` 非单调**
   ⇒ 障碍层占的 178.4 m² 里 **120.9 m² 曾是地面**；`ground_after_obstacle==0` 占 165.5 m²
   ⇒ **`OBSTACLE_RELEASE` 永不触发**（车开过去不回看）。形态：**77% 的灰在墙 0.3 m 内**，
   0.1–0.3 m 带灰 vs 地面 ≈ **70.9 : 62.1 m²**，这些灰 votes 中位只 **6–9**（真墙 23+）
   —— **与 ATE p50 = 0.276 m 吻合**（墙点被横向抹到墙边地面格）。
   **修法 `OBSTACLE_GROUND_RATIO = 1.0`**：ground 489→549 m²、obstacle 178→118.5 m²、面板墙色 −38.8%。
   **★ 安全判据：放掉的 60 m² 里 99.8% 是 `ground_ever=True` 的矛盾格，"纯墙且 votes≥20" 一格没碰、新晋升 0。**
   ★★ **闭环 A/B 已过**（仿真 n=3 vs 3，`out/_ratio_sim.sh 3 1.0`）：**12 个指标无一分离**
   （`traj`/`strand`/`arrivals`/`pf`/`wedge`/`blocked` p=1.00；`reach` 趋势**变好** 166.9→183.4）；
   **唯一 3/3 分开的是 `obstacle` −15%（4145→3446）＝正是预期效果**。⚠️ n=3v3 置换检验 **p 下限 0.10**。
   ★★ **默认已改为 1.0**（boss 批准）。**Gate −1 已验**：同 bag 各 1427 帧，`online_real_def`（不传）
   与 `ratio_1.0`（显式传）**决策 md5 逐位相同 `b664e5cfb6f76076`、差异列 0** ⇒ 无隐性行为变化。
   ⚠️ `GROUND_COVER_INFORMATIVE` **仍是 False**（+55% 真实侧算数代价，未决）。
5. **★★ 类别表是我们自己定的，已扩到 39 类**（交接 §5.2：**组委会 FAQ 明确「不提供类别列表」**）
   ⇒ `labels.py::_SPECS` 是**我们的选择**；画面里没标的（灭火器箱/栏杆/地毯垫子/健身器材/人形机器人）
   **一个都不在 COCO 80 里，`door` 也不在**（⚠️ 我曾把 `door` 写成"没用的 COCO 类"，**错**，已改）。
   **已完成**：追加 13 个 COCO 类（umbrella/handbag/wine glass/fork/knife/spoon/mouse/remote/cell phone/
   toaster/scissors/teddy bear/sports ball，**推理成本为零**——检测器一直输出 80 个槽位，是这张表丢掉的）
   ＋ **2 个几何类：38 `unclassified object`、39 `railing`**（无 COCO id，由 `instances.py` 形状判定；
   `write_table` 输出 `kind: geometric-object`）。**`PERSON=2` 等 23 个旧 id 全部稳定未动。**
   **"很多没标出来"= 四个不同原因**：①**词表缺类** ②**远处点云太稀被 LiDAR 门拒**
   （⚠️ person `min_conf` 只有 0.30 ⇒ **不是置信度问题**）③**没有实例层** ④YOLO 对小目标本来就漏。
   **⇒ 换 3D 检测器只治 ③④，治不了 ①②；而扩 COCO 只治 ① 的一小半（缺的类多半根本不在 COCO 里）。**
6. **★★★★ 边缘算力约束 ⇒ 建议顺序全变**（§12，`e2877b1`）。这是**规则要求**：
   「注意算法的计算效率，**需适配嵌入式计算平台**」＋ FAQ「**模型大小需适合在边缘计算设备上运行**」。
   **现状（真实侧 p50，本机桌面）：算数代价 `total − render − dump` = 170–264 ms，而 5 Hz 预算 200 ms。**
   **★ 地图层是平方律**：真实侧 `n=1200`（1.44 M 格）；`free_space` 缓存键是 `int(ground.sum())`
   ⇒ **命中也要两次全量归约**，每帧调用 4+ 处；`ground` 大 11% ⇒ `ms_map` p50 **+46%**。
   **★ `render` 占 24–35% 帧时间，而它不是算法**（交付视频）⇒ **唯一一块不碰判据就回血的预算**。
   **重排（按"每瓦覆盖率"）**：① **先拿回预算**（render 下关键路径 → 地图层平方律，**都 A/B 门**）
   ② **免费的覆盖**（**零新模型**）：全场景欧式聚类出实例 id（复用 `fusion.object_3d` 深度切簇）＋扩词表
   ＋几何规则类 ③ 若需任意类别 → **2D `YOLO-World` 替换 `yolo26n`**（文本编码器重参数化进 1×1 卷积
   ⇒ **在线成本为零**；已发表边缘实测 **115 MB→32 MB / Jetson Orin Nano Super 32.6 FPS**；**替换而非叠加**）
   ④ **3D 大模型判死**（MinkowskiEngine CUDA 算子 **导不出 ONNX** ⇒ 边缘端跑不了）；**IGLOSS 降级**
   （需 torch+GPU，比整条栈还大）。**⇒ 第一件要做的是在**指定参考平台**上量逐阶段预算，不是加模型。**
7. **★★ 实例层（`r2slam/instances.py`，`--instance-layer`，默认关）—— 能力有、判据已被真值证伪**
   全场景聚类：非地面回波 → **BEV 栅格 + `cv2.connectedComponents` + `reduceat`**（全 numpy/OpenCV；
   **不用 `fusion.voxel_components`** —— 逐点 Python 循环扛不住 49k 点/帧）。
   **时延**：合成 49152 点 **3.7 ms**；全量 1427 帧 **p50 9.5 ms**（首次 20.7 ms 是 OpenCV 惰性初始化）。
   ★★ **行为中性已证**：与只差这一个开关的跑**决策列 md5 逐位相同**（`b664e5cfb6f76076`）。
   ★★★ **真值评测已做 —— 判据确实错了，且有数字**：新增 `--instance-log`（每帧每簇落盘）＋
   `out/_truth_geo.py --dump-models`（逐模型世界系 AABB）＋ `out/_inst_truth.py`（评测器）。
   仿真 3136 簇 / 874 帧：**落在家具上 16.2%、落在真人上 19.3%、只落在墙上 63.4%、落在空地上 0.0%**。
   ⇒ **精度（家具+人）35.9%，64.1% 是"把建筑当物体"。** ★ **`on nothing = 0%` ⇒ 错误是单向的**
   （把墙当东西，不是乱撒点，比"到处都错"好修）。★ **召回也低：22 个家具模型只有 12 个（55%）被认领**
   （Bed/Refrigerator/TVCabinet/TV/Vase/Wardrobe **一次都没被认领**）。
   ⚠️ 真值对齐 p50 **0.200 m**（=2 格，该口径精度上限）—— 结论要带这条不确定度。
   **判据目标从此明确**：提高"落在家具上"占比、保持"落在空地上"≈0 ⇒ **在此之前不再改几何阈值**
   （前三轮调参每次都能把数字挪一点，却没有一次能回答"好到什么程度算够"，这就是它在原地打转的原因）。
8. **★★ ⑤ 开放词汇 2D 已跑通，但算力账算不过来**（`yoloworld_s.onnx`，50.37 MB）
   CLIP/MobileCLIP 文本塔（**338 MB**）重试时**下载成功** —— 之前的 `URLError: SSL UNEXPECTED_EOF`
   **是瞬时失败不是被墙**。链路：36 prompts（来自 `labels.py`）→ `set_classes` 烤进权重 →
   ONNX `[1,40,8400]`（4 框 + 36 类）＋ `*.classmap.json` sidecar。⚠️ **`.onnx` 是 gitignore 的（模型不入库）**。
   **parity（我们的解码 vs ultralytics 解同一张 ONNX）：8/8 帧通过，最大 1.00 px** ✓。
   ⚠️ **代价：体积 5.1×（9.88 → 50.37 MB）、`detect` 耗时 3.8×（99 → 381 ms）**；
   `yolov8s-worldv2` = 12.75 M 参数 / 32.3 GFLOPs，**YOLO-World v2 没有 nano** ⇒ 下不去。
   ⇒ **detect 一项会从 55–72 ms 变成 210–275 ms，光它超 200 ms 预算** ⇒ **量化前不进配置**。
   ⚠️ **收益未验证且预计很小**：新增 13 类**全是小物件**；boss 报的缺失项（灭火器箱/栏杆/地毯/
   健身器材/人形机器人）**仍然一个都烤不进去** —— **开放词汇解决"词表宽度"，不解决"我们没见过的具体物件"。**
   ★ **两条路的瓶颈其实是同一个**：缺一个能判"这是不是物体"的判据。
9. **§73**：**报覆盖必须同时报 `wall_s` 与 `sim_span`（RTF）**；"进不去房间"曾是 `--duration` 墙钟截断。

## ★ 渲染（定论）
- **`render()` 只画 `ground`，不许自己算判据。** 画面可以有颜色，不能有第二个真相源。
- **"会动"是渲染 bug 的指纹，不是位姿 bug 的指纹**（裁剪窗口变非正方形 + 叠加层只用一个 `k` ⇒ 比例每帧变）。
- **★ 面板是俯视图：+x 向右、+y 向上。** 曾把格行号直接当图像行号 ⇒ **上下镜像 ⇒ 左右互换**。
  ⚠️ **老测试当年在镜像图上照样全绿**（它的辅助函数手写复述了渲染器口径）
  ⇒ **复述实现的测试抓不到实现里的约定错误。**
- **灰斑 = 画面在说谎**（索引 3/4）。**写渲染规则要按语义写，不按形状写**（只填孤立单格的规则
  遇到 2 格宽的条就整条漏掉）。
- **帧尺寸必须钉死**：真实 bag 里有 2 帧拼图宽度不同 ⇒ 编码器**静默**把它压进固定尺寸。

## 家底 / 赛事硬约束
- 唯一"丢了不可再得"：`D://E-Downd//data.bag`（初赛 2,532 MB，81 分，根因＝RTK 降级段水平位置回退到里程计）。
- **★ 交付形态**：实时运行 + **整个数据集的视频** + 自建图 pcd；官方 pcd 只作提示；仿真 LiDAR yaw **不修**；
  输入只允许**点云/图像/IMU**；**要输出建图 pcd + TF**；**建图 50% / 感知 50%**。
- **★ 边缘算力是规则要求**（索引 6）；**类别表组委会不给**（索引 5）；
  **真实侧无真值 ⇒ 任何 mIoU / 识别率数字都是编的（要敢说"不知道"）。**

## 环境
- **仿真＝本机 Docker 容器 `r2sim`**；**`/work` ＝ `D:/E-Downd/sim_official`**（**权威查法**：
  `docker inspect r2sim --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}'`；
  ⚠️ **容器内 `mount` 打的 `aname=drvfs;path=D:\` 是假像 —— 按它推的宿主路径不存在**）。
  容器内代码 `/work/r2slam_ws`。★ **产物直接落在宿主盘 ⇒ 回拷用宿主文件操作，不必 `docker cp`**。
  **改完本机必须 `sync_to_container.sh`**，否则容器里**静默**跑旧代码。
  ⚠️ `PKG_PATHS` 是**手工白名单**；**`tools/replay_explore.py` 不在里面 ⇒ 必须单独 `docker cp`**。
  `out/*.py`（只读工具）跑法 `cd /work/r2slam_ws && python3 out/xxx.py`（拷到 `/work/ab/` 会 `ModuleNotFoundError`）。
- 只读复算在容器里跑：`export MSYS_NO_PATHCONV=1 && docker exec -i r2sim python3 - <<'EOF'`
  （不设 `MSYS_NO_PATHCONV` 会被 Git Bash 改写路径；**漏 `-i` 则 stdin 读不到、静默无输出**）。
- **真实侧在本机 `.venv` 跑**；`wsl.exe` / `wmic.exe` 被安全策略拦截（查进程改用 `tasklist //FO CSV`）。
- **★ `Xvfb :99` 被 kill 后不清理 `/tmp/.X99-lock`** ⇒ 新 Xvfb 拒启 ⇒ **相机完全不发，而 LiDAR 照常 5 Hz**；
  就绪门**只查 velodyne** ⇒ 仿真"看起来完全健康"。修：起前删锁 ＋ **就绪门查两个话题**。
  **可迁移：就绪判据必须覆盖全部输入。**
- ⚠️ **"Docker 死机"不是容器 OOM**（`oom_kill 0`、无 cgroup 上限、磁盘 190 GB 空闲）
  ⇒ 在 Docker Desktop / WSL2 层；**`/work` 是 bind mount ⇒ 容器死产物全在**（已加内存记录器）。
- **★ 短跑机理（代码级）**：`sim_ros1.py:122` 循环条件是 `not rospy.is_shutdown()` ⇒
  **ROS master 一断就 `return`：不抛异常、退出码 0、summary 完整**。**有效性门是必需品**：
  **`exit==0 && frames.csv 跨度 ≥ 0.97×目标`**。`--sim-duration` 只保证"跑到就停"。

## ★★ 官方仿真器两条最贵的坑
1. **LiDAR 不随车体旋转**（yaw 写死 0）⇒ 航向不可观测。⚠️ `yaw_odom_deg` 不转（3–5°），
   而 `yaw_deg`(=`yaw_proj_deg`) 与 `truth_yaw_deg` 是转的 ⇒ **判航向问题前先看这三列各属哪条通道**。
2. **相机—LiDAR 外参只在 yaw≈0 成立** ⇒ `p_cam = Rz(-yaw) @ Tcl @ p_lidar`。
- ⚠️ "反投影回原框"自检**抓不到**这两条；有效判据＝与 `/gazebo/model_states` 比对。
- **★ 无 IMU、无里程计话题 ⇒ 单跑之间不能比分数。**

## ★★ 真值通道（洞已修）
`/root/.gazebo/models/house.world` 烘焙了每个 model/link/collision 的世界位姿 ⇒ 重建真值占用栅格
（`out/_truth_geo.py`，**必须传 `--map`**，否则静默 `return 0`）。⚠️ **COLLADA 是 XML**；
**`up_axis` 声明不可信**（照声明翻会离地悬空、静默贡献 0 格）⇒ 按"底部贴地"自适应。
**地图系＝世界系**；对齐用距离型目标（别用薄墙 F1）。★ live 栅格（`grids_A0I_r*`）可直接量世界坐标点。

## 感知 / 地图（定论）
- `labels.py`＝类别表唯一源；检测器＝**`yolo26n.onnx`（9.43 MB）**（回退 `yolo11n.onnx`），模型不入库，
  `ROOT_PATHS` 必须列上；换模型**必须**跑 `tools/check_onnx_parity.py`。
- 语义地图：`绿 = ground_ever & ~obstacle`；**格子只有"有回波落进去"才算 seen**。
- **障碍层**（`worldmap.py`）：`MIN_VOTES=2`（数**帧**）｜`LO/HI=0.35/2.00`｜`RELEASE_FRAMES/RATIO=2/0.30`｜
  `OPEN_PX=3`｜**`STRONG_VOTES=20`**｜**`HOLD="reset"`**｜`MIN_POINTS=1`｜`GROUND_RATIO=0.0`｜
  **`GROUND_COVER_INFORMATIVE=False`**｜**`blocked() = obstacle | obstacle_now`**（`obstacle_now` 在过滤**之前**写好）。
  ⚠️ **别再退回 `votes<12` 那种票数上限**（封死过两扇门）。
  ⚠️ **`obstacle_now` 写入 bug 已修**（`now[cyo,cxo] |=` 对重复索引**不保证 OR**）⇒
  改前 vs 改后**决策 md5 完全相同 ⇒ 行为零变化**。
  ★ **`free_space` 读 `wmap.ground` 不是 `blocked()`** ⇒ `reach` 不受当帧层影响。
  ★ **`add()` 末句是 `ground = ground_ever & ~obstacle`** ⇒ `ground` 是 `obstacle` 的**纯函数**，
  所以"某格从未晋升"可在**同一张图**上离线算（§68 反事实靠这条）。
- **格索引**：`floor((xy + extent) / resolution)`；**真实侧 `res 0.10 / extent 60.0 / n 1200`**、
  仿真 `extent 30.0 / n 600`；`cell_of` 返回 `(cx, cy)`；**必须 `floor`**（`round` 送偏一格）。

## 算法栈不变量
- `config.py` + `sources/` 是唯一允许知道"仿真/真实"的地方；**下游出现 `if sim`＝设计失败**。
- **`--explore` 是 `store_true`、默认关**（仿真不传 ⇒ **机器人一动不动**，但 summary 照样完整、日志照样健康
  ⇒ **判据要落在"机器人真的动过"上**；真实侧不传是对的，bag 自带运动）。
- `online_pipeline.py`＝主程序；`package_submission.py`＝终审（**FAIL 绝不许写 zip**）。
- **★ 当前提交包 `…【固态蜗牛队】…zip` = md5 `b8cb38957c64f379277f4fa6148b1a31` / 37,481,494 B**
  （2026-10-08 16:16，随包终审 **0 FAIL/0 WARN**）。**此后所有改动都没进包，也没动包。**
  ⚠️ **选跑依据已更正**：旧口径"三房最小 90%"是假的；权威 `_room_cover2.py` 下 **subr2=32% / subr3=50%**
  ⇒ **subr3 覆盖更好**，subr2 胜在漂移。**换不换是"覆盖 vs 漂移"的权衡，等 boss 拍**（两边各 n=1 且都被挑过）。
  ⚠️ **包里没有任何代码** ⇒ "包里 `worldmap.py` 是旧的"是错的；**v7 后探索/地图改动本来就不进包。**
  ⚠️ 真实视频**必须 1.4× 加速**（> 终审 120 s 上限）；`encode()` 字体候选**只有 Linux 路径**；
  ⚠️ `vendor_check.json` **只在包内、不在 `out/online_real/`** ⇒ 重打包会丢，要从包内拷回。
- 一律用 `header.stamp`；`--npz` 复算无驱动器 ⇒ `dump_frame(extra={"control_yaw": yaw})` 必须存；
  **npz 存的是未去斜原始点云**。改完管线**两源各 5 帧冒烟**；**改签名后必须真跑一次**；**比 md5**。

## 探索算法：结构
规划＝A\*（8 连通、按半径腐蚀 + 障碍膨胀）；跟随＝三档油门 + **视线可达前视点**；
探索＝前沿簇 + 角扇区 + `score = size/(1+d_start)` + `sweep` + `island`（门口探针）+ `shadow`。
**★ 目标＝"观测位姿"**（`target_standoff_m=0.80`）。**`n_frozen_frames` 必须永远 0。**
**★ 四态**：`drive`(aligned & ahead)｜**`wedge`(aligned & !ahead ⇒ 立即 backoff)**｜
`arc`(|err|≤90 且转向方向 clear)｜`turn`(否则，v=0)。
**★ 状态机分支对地图细节极敏感：跑间差一格就换分支 ⇒ 单跑会得相反结论。**

## ★★ 地图层：已封板清单（**细节全在 `docs/探索算法_v2_设计与参考.md` §56–§71**）
**已封板（不再动）**：D1 / D3 / D1+D3 / `island_trigger_m2` / `barren_blacklist` / sweep·frontier 参数 / 价值函数。
- 共同上游：**少数格卡在窄口上 × 障碍膨胀半径 3 格** ⇒ 一刀切断 174 / 136 m²。
- **机制 A（§62）**：门口 3 格第 2 票同在 f490 升格（`MIN_VOTES=2`），自愈 f603。
- **机制 B（§64/§68）**：开运算在 `STRONG_VOTES=20` 处的阶跃。★ θ=1 对它完全无效；`20` 不是"强"。
- ⚠️ replay 无墙钟预算 ⇒ **验收必须闭环 + 报 `ms_*` 的 p90/max**。
  ⚠️ **不要调参**：动 `MIN_VOTES`／`STRONG_VOTES`／release／开运算／`island_*`／`0.5`／`link_m`／`d0`／
  价值函数**仍是调错层**（一个数都没试过）。

## 已判死 / 登记不落地（理由与重开条件在技能里）
- **不再重试**：`local_u`｜`region_lambda`｜`rfirst_gate`｜`fb_relax`｜`OBSTACLE_HOLD=total`｜
  `debt_gain_m2`/黑名单/进度期限。**`rfirst`**：锁在房子东半部 ⇒ 重开**必须有可见性论证**。
- **B3（`OBSTACLE_TOPO_FUSE_M2`）已停投**（索引 2）。
- **当前不值得做（非判死）**：`V/(1+d) → V·exp(-λd)`。
- **登记不落地**：**G7 地图形态学清理器**｜`fresh_split_m`｜**点云层判别器**（在线 18 维留一 AUC **0.638 < 0.65**
  ⇒ **观测层信息不足，不是特征不够，别再堆特征**）。
- **K=2 已封板**：分的是回波密度不是真实性（**墙丢 4.7% / 家具丢 58.6%**）。
  ⚠️ **"高度带能排除桌椅"已被 8-run 真值推翻**；**单 case 通过 ≠ 判据成立**。
- **已装开关（都默认关，`grep` 找）**：`GROUND_COVER_INFORMATIVE`｜`OBSTACLE_GROUND_RATIO`｜`RENDER_*`｜
  `disc_lambda`｜`fresh_split_m`｜`patch_probe`(只读)｜`sweep_value_mode`｜`island_probe`｜`strand_dump`(只读)｜
  **`--cf-hold-cells`(只读反事实)**。
  ⚠️ `_fold` **统一三处折叠**（sweep/shadow/frontier），只改一处会被两个 picker 不一致污染。
  ⚠️ **出厂运行时只有 numpy+cv2+onnxruntime ⇒ 诊断不许 `import scipy`。**

## ★★ 指标：比率型必须先查分母
- **权威口径＝`out/_room_cover2.py`**（分母＝**单调**的 `ground_ever ∩ 框`；< 2 m² 记"定义不了"）。
  **报任何百分比必须同时报分母。** **★ 每-pick 中位数不要跨臂比** ⇒ 比比例。
- **★★ 项目级硬规则（boss 立，优先于任何单项判据）**：① 任何 count 型指标只要受"策略做了多少次决策"
  影响，**必须同时报 normalized rate**；② 任何"切换/放弃/重规划减少"类指标，**必须同时检查完成量/覆盖是否下降**
  ⇒ `switches` **永远不能单独出现**。
- **★★ 两个 n=1 的比较不是实验，是抽签**；且要查**参考集本身有没有被挑过**
  （`subr*` 是"被留下来当提交候选"的 ⇒ **选择偏置，不能混进基线分布**）。
  **n=3 vs 3 置换检验 p 下限 = 0.10** ⇒ 只能说"没分离"，不能说"等价"。
- **★★ 算"行为是否不变"的 md5 必须剔除 `traversable_m2` 这类统计列**。
- **★★ 两源量纲不同**：`replay_explore.py` 的 `yaw` 是**弧度**，`online_pipeline.py` 的 `yaw_deg` 是**度**
  ⇒ 跨源三角运算**先 `repr()` 打原始值**。
- **★★ 跨仪器对帧只用 `stamp`，不用 `frame`**（`--probe-cells` 的 `frame` ＝ 决策 CSV +1；`strand` 又 −1）。
- **★ 报"掩码变化量"必须同时报格坐标。**
- **★ 影子重放测决策、不测覆盖，也测不了"指令→运动"** ⇒ **"机器人最终路径"在这台仪器上不可测**。

## ★★ 硬门槛 = Decision-effect gate（细则在技能 §"本项目工程硬规则"）
- **Gate −1：对照臂真的关掉了吗** ⇒ 默认值与不传**必须 md5 逐位相同**，**且必须与"历史记录的 md5"逐位相同**；
  **哈希"决策列"不是整个文件**。★ r3 基线决策 md5 ＝ `94df03cd8a087fc9b15d2ff4086a5d7f`。
- Gate 0/1/2/3 数值门槛、闭环评测口径、反事实前先做守恒 —— 全在技能里。

## 硬规则（工程）
- **会话开头先补 PATH**：`export PATH="/c/Users/ZhuanZ/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:$PATH"`。
- **含中文路径**：存在性判断用 Python `os.path.exists`，不要用 Git Bash 的 `ls/cd/find`。
- **★ 最普适：自检通过 ≠ 正确**；判据要落在**具体量**上（"机制跑了" ≠ "机制起作用了"）。
  ⚠️ "新面孔"：**坏在输入的开关**时，连 span/帧率/summary 完整性照样全绿。
- **★ `pgrep -c -f <pat>` 会算进调用者自己的命令行和僵尸**；**`pkill` 跳过自己、但不跳过父进程**
  ⇒ `docker exec bash -lc '...pkill -f X...'` **会杀掉执行它的 shell**（连踩两次）。
  **清理写成脚本文件**（自己的 cmdline 不含任何模式），读 `/proc/*/cmdline` 决定杀谁：`out/_cleansim.py`。
  ⚠️ **`pkill -x gz` 是错的，`gz` 是真实文件。**
- **★ 有重试的驱动必须"每次尝试一份日志"**（`run_${TAG}_a${attempt}.log`）；写同一文件会让**失败原因被覆盖
  ⇒ 归因链断掉**。
- 其余（docker/后台跑/Edit 纪律/守恒量/事件 vs 帧/参数扫描/commit）见技能同名小节。
