# SLAM 大师赛 — 项目长期记忆

> 只放**结论 / 判死 / 硬约束**，一条一句。推导全在 `docs/`，流程细则与踩坑在技能。
> 本文件是**索引**，不是教材：数字要追问就去看对应文档。

## 文档地图
- 状态/P0-P2：`docs/交接_20261008_当前状态与后续计划.md`
- **★ 现役落定 + 双跑 + 交付包 + 分割实测**：`docs/现役落定_双跑与交付包_20261010.md`（§5 分割、§7 并跑）
- **★ 仿真退化归因 + 掩码接入融合**：`docs/仿真退化归因与掩码接入融合_20261010.md`
- 评分规范：`docs/评分规范对照_20261010.md` · 复评：`docs/评审小灶_评分标准全面复评_20261010.md`
- 呈现/M3：`docs/呈现层修复与M3复核_20261010.md`（§3 ATE 旋钮）
- 语义地图主线：`docs/语义地图朝向与灰斑修复_20261009.md` · IMU：`docs/IMU启用_陀螺仪去斜与航向核对_20261010.md`
- 开放词汇：`docs/开放词汇边缘化_第一轮实测_20261009.md`（§32–37 YOLOE）
- 历史：`docs/HANDOFF_项目交接.md`（§9 过时，§5.2/§8 必读）、`docs/复盘_踩坑手册_小白向.md`、`docs/探索算法_v2_设计与参考.md`
- ⚠️ 技能 `robot-policy-replay-ab/SKILL.md` 已 187 KB，该按主题拆（未做）。

## 结论索引
1. **漂移**：真实侧不显著（ATE p50 0.276 m / 102 m）；仿真侧＝少数帧被打偏不恢复，根因＝**官方仿真器航向不可观测**，非 `heading.py` bug。
2. **B3 已收线停投**：成本门 PASS（只看 `ms_map` p90）；效果门 FAIL 撤回（128.90→82.43 m）。归因＝成本上界 `max-eval` 语义缺陷。
3. **渲染↔规划一致性**：`render` 只画 `ground`（＝`free_space` 读的同一数组）。**画面可有颜色，不能有第二个判据。** 指标 `summary["map"]["seen_not_classified_cells"]` ⚠️ **不是 0，代码里"0 by construction"是假的**：真实 1464 / 仿真 277，其中 `ground_ever` 全 False、`obstacle_votes==1` 占 91% ⇒ 是"回波只落障碍带且没凑够 2 票"的格，两边都判"不可走"，**一致**。
4. **灰斑/靠墙灰带**：`OBSTACLE_GROUND_RATIO=1.0`；**`GROUND_COVER_INFORMATIVE` 默认 True**（flag 反转为 `--no-...`）。1427 帧背靠背：决策 md5 逐位同、灰 6824→1464、可通行 495.3→548.9 m²。旧"`ms_map` +46%"不成立（负载污染）。
5. **类别表自定 39 类/51 prompt**（`labels.py::_SPECS`，3 个不在 COCO80）。"没标出来"＝①词表缺类②远点稀被门拒③无实例层④漏小目标；换 3D 检测器只治③④。
6. **边缘算力是规则要求**。顺序：①拿回预算②零模型覆盖③需任意类别→换检测器④3D 大模型判死。**先量预算。** render 已搬 worker 线程 + `render_view()` 快照 ⇒ `ms_render` 60.46→1.04 ms（**只在多核有效**）。**seg 的真实门槛不是算力**：现役检测器全路径 66.42 ms；`yolo11n-seg`/`yolo26n-seg`@320 原始 **13.5**、**全路径 23.96**（解码只 1.9）⇒ 320 上的小号 seg 比现役快 2–2.6×，卡的是解码与投影。⚠️ 老结论"seg 撞预算"只对 **YOLOE-seg@640**（159/581）成立。**FastInst-AA / UniPercepNet-S 按其自报 GPU 延迟判死**（后者 RTX A5000 上 43 ms/帧＝我们整个 detect 预算）⇒ 不装 MMCV。
7. **实例层**（`instances.py` 默认关）：判据被真值证伪（家具 16.2%/真人 19.3%/只落墙 63.4%/空地 0% ⇒ 精度 35.9%）。**在此之前不再改几何阈值。**
8. **开放词汇**：默认 `yoloworld_alias_320.onnx`（51 类 fp32@320，opset 19）。唯一增量＝`fire extinguisher` 进地图。`gym equipment`/`humanoid robot` 检不到**不是措辞问题** ⇒ 只能换骨干。**INT8 在 51 类失效 ⇒ 只能 fp32。** `detect_mapped` 只留 top-1 ⇒ 必须报 `max`/`any≥x`/`top1`/`top1@conf` 四列。
9. **改进空间**：① CLIP 喂裸名词——形式是杠杆但整表套是净回归 ⇒ 必须逐类标定。② **最大结构性缺口＝没有尺子**：真实侧无真值 ⇒ 一切"检出/改善"都是自证的；最小可行＝人工标 30–50 帧。
10. **YOLOE**（doc §32–37）：文本提示 0；**视觉提示在 `11l-seg`@640 上 10/10 命中**，`11s` 只 9/17 ⇒ 靠骨干容量；收紧参照框无效。**烘焙成立**（vpe 2.4 kB、**模型专属**）。⚠️ seg 头通道＝`4+nc+32`。免提示（4585 类）对 `robot` 0 命中。⚠️ 更正：上一版"导出把类头弄死"**是错的** ⇒ **导出验证必须同分辨率、同权重、同前置条件**。
11. **IMU：三臂无一是赢家，不采纳**（ATE p50 0.276/0.300/**0.352**，1 秒 RPE max 0.256/0.715/**0.209**）。以前没用＝`deskew.py` 一条过期注释。n=1。★ 真收获：`tools/check_imu_heading.py`＝**真实侧第一条独立航向证据**（陀螺 −354.1°/我们 −363.0°/厂商 −361.2°）。
12. **★★★ 官方 100 分表**：P1/P2/P3 全是"人物"共 30；G1–G3 25；M1–M5 40；V 5。G2 要"显式区分可通行/障碍/未知"⇒ 正对我们三分类。M5 唯一按公式：**ATE + 1 秒 RPE + 时间覆盖率**。★★ **投入≠分数**：开放词汇三轮只影响 M4（10 分）；呈现层半小时改动直接对 V+P1。
13. **★★★ 腐蚀核方核→圆核**（2026-10-10）：玻璃房那块 3.14 m² 地板被方核误杀。核按运动学生成 `res·hypot(max(|a|-0.5,0),max(|b|-0.5,0)) < r` ⇒ 5×5 减四角 21 格。⚠️ **`cv2.MORPH_ELLIPSE` 不能用**（放行 0.158 m 间隙）。reachable 392.7→401.8、真·另一分量 3.61→0.67 m²。**默认已翻 `disc`**；`--footprint square` 逐位复现旧产物。**真实侧地图/决策 md5 不变**；⚠️ 仿真策略会变（replay n=1）。`tools/test_footprint.py` 16 项。
14. **⚠️ `diagnose_reachability.py` "外壳检验"曾有 10× 单位错**（2.0 m 打成 0.20 m）⇒ 三份文档"98.9% 在 0.20 m 内"全作废。更根本不该用距离阈值。改无阈值精确分解。★ 教训：**引用工具输出前先看它算的是什么。**
15. **★★ M4 标签口径**：`label` 改**按点**判（`traversable.geometric_label(h, ground_mask, obj_mask)`），`h>2.0`→ceiling、`0<h≤2.0`→wall；新类 id **43/44 排在物体类之后**（`person=2` 与家具类不重编号 ⇒ pcd 与 `label_table.yaml` 对得上）。background 48.0→0.38%、traversable 48.8→22.4%、wall 26.6%、ceiling 47.5%。★ 非 person 具名物体仍只 263 点（0.104%）＝检测器能力。`tools/test_geometric_label.py` 15 项。
16. **★★★ 现役落定 + 双跑 + 出包（2026-10-10）**：四项默认**本来已是默认**（disc / GCV=True / 按点标签 / render worker）。**出包前两个红测试**：①`test_nav` 夹具**把地板铺到墙底下**（旧测试是"重复下标后写覆盖"的**侥幸绿**）⇒ 加 `wall_occludes_floor()`；②**`explore.py` 真 bug**——区域秩移到 `c[22]` 后**三处读者仍写 `c[-1]`**，三个计数器全被钉死（`rfirst` 默认关⇒流水线看不出来，是测试抓的）⇒ 模块级 `CAND_REGION=22`。全套 **264 项全绿**。**双跑**：仿真 1436 帧/287.1 s（`--duration 400` 是**墙钟**，RTF 0.72 正常）、真实 1427 帧 @10.27 Hz，**真实侧地图逐项复现**（障碍 11860/可通行 548.9 m²/标签 0.38-22.39-26.56-47.47）**⇒ 真实侧决策未被动**。**交付包** `提交包/20261010/…zip` md5 `408d97bce998f5b571ee975f04c98f78`/37,059,063 B/**0 FAIL 0 WARN**。⚠️ 本场 `ms_*` 带负载（真实侧含诊断 `--dump` 275 ms/帧）⇒ **不能当预算结论**。
17. **★★ 分割头并跑**：`r2slam/seg.py`（解 `4+nc+32`+原型，`nc` 由形状推、对不上就抛；套**与检测器同一套**逐类门槛）+ `viz.blend_masks` + `--seg-model/--seg-conf/--seg-alpha/--seg-classes/--seg-fuse` + `summary["segmentation"]`。**检测器不动**。**Gate：真实 `map.pcd` 与交付包逐位相同**。⚠️ 320 原型 80×80 ⇒ 真实鱼眼上一格≈20 px，掩码边界就是这个台阶。★ **闭集代价**：`clock 74`（贴海报整面墙被判钟）⇒ `--seg-classes` **默认 `person`**（P1/P2/P3 全是人物 30 分，其余 79 类无得分项）。★ **边界**：真实 frame 31 检测器 `person 0.30` 落图而**分割头没看见** ⇒ **它不是更好的人检测器，是形状提供者**。★ **做 2 已接通**（`fusion.object_3d(mask_test=...)`+`seg.mask_cells`+`_match_seg_box` IoU≥0.35 同类；`--seg-fuse`**默认关**⇒pcd 逐位不变）。⚠️ **未证明更好**：要仿真 `/gazebo/model_states` 真值 A/B 才算验收。⚠️ **踩坑**：`w=int(round(x1-x0))` ≠ `int(round(x1))-int(round(x0))` ⇒ 真实侧一帧即崩、**仿真不复现** ⇒ 先裁整数像素再按它 resize。`test_seg.py` 19 项。演示件 `r2slam/out/seg_demo/`。**交付包未重建**。
18. **★★ 仿真"探索退化"归因**：**结论＝已知双峰失败的一次坏抽，没有找到可归因的代码回归。** **76** 个归档跑 `traj_xmin` 中位 **−11.03**、本次 **−6.60** ⇒ 底部约 15%、倒数第 12（**不是唯一离群**）。机制：**三次往西**均 `ahead=False`→掉头；瞄准的通道 x≈−6.0 是 49 票真墙、墙后一格 **`seen=0`**，而 y≈−1.0 整条可走（有通路，当次没走上）。★ **两个嫌疑都被测量排除**：①圆核——同图 disc 与 square 的可达分解**逐项相同**；②GCV——10-09 闭环 **n=3v3 无任何指标分离**。★ **"雷达探到墙外"不是新的**：66 跑里 **22** 跑有（本次 2088 格，最多 9370）；机理＝**顶墙时 LiDAR 原点落进墙碰撞体、射线穿出打到房外地面** ⇒ 与"卡墙"同源。⚠️ **待做**：圆核**闭环 n=3v3**（50 min）是唯一能关掉问号的办法。★ 结构性：`ahead_clear` 判死就整条放弃，没有"绕一下再试"的中间态。

## 渲染（定论）
- `render()` 只画 `ground`。**不能有第二个真相源。**"会动"是渲染 bug 的指纹。
- 面板俯视图 +x 右、+y 上（曾上下镜像⇒左右互换）。**灰斑＝画面在说谎。**
- 写渲染规则按语义不按形状。帧尺寸必须钉死（bag 有 2 帧宽度不同，编码器静默压缩）。
- ⚠️ 复述实现的测试抓不到实现里的约定错误。**同一规则只能有一份实现**（`mask_cell` 委派给 `mask_cells`）。

## 挂起的问题（boss 说"后面再讨论"，回来必须能接上）
1. **"M5 的 ATE 旋钮"**：`runtime.voxel_size` 0.50→0.25，ATE p50 0.276→0.140 m（1 秒 RPE 0.041→0.027，路径比 1.039→1.018），外部佐证（对齐 yaw +0.00° vs +2.00°）。代价 `ms_odom` 15.3→53.9 ms、p90 26→131（须同时把 render 移出关键路径）。⚠️ `ate_tail.py` 证明 ATE 主要是坏段计数（0.50 有 41.9% 帧超 0.30 m、一段 36.7 s 占平方误差 56.2%）⇒ 跨序列稳的是 RPE 与路径比。**默认仍 0.50。** 三候选：①0.25+render 出路径 ②0.30 ③保持。数字：`docs/呈现层修复与M3复核_20261010.md` §3。

## 家底 / 硬约束
- 唯一不可再得：`D://E-Downd//data.bag`（初赛 2532 MB，81 分）。
- 交付：实时 + 整集视频 + 自建 pcd；官方 pcd 只作提示；仿真 LiDAR yaw **不修**；输入只允许点云/图像/IMU；要输出 pcd + TF；建图 50%/感知 50%。
- 边缘算力是规则要求；类别表组委会不给；**真实侧无真值 ⇒ 任何 mIoU/识别率都是编的（要敢说"不知道"）。**

## 环境
- 仿真＝容器 `r2sim`；`/work`=`D:/E-Downd/sim_official`；代码 `/work/r2slam_ws`。**改完必须 `sync_to_container.sh`**（`PKG_PATHS`+`ROOT_PATHS` 手工白名单，已含 `frames.py`/`sources/*`/`viz.py`/**`seg.py`**/`replay_explore.py`/`yolo26n-seg_320.onnx`）。⚠️ **漏文件的后果是静默的**（容器跑旧代码，copy 步骤不报）。
- 容器运行时：py 3.8.10 / **onnxruntime 1.16.3（只到 opset 19）** / cv2 5.0.0 / 无 `onnx` ⇒ **导出一律 `--opset 19`**。
- **杀仿真**：`docker exec r2sim bash /work/killbot.sh`。⚠️ 宿主跑 `vendor/sim_scripts/kill_sim.sh`＝没跑。
- 只读复算：`export MSYS_NO_PATHCONV=1 && docker exec -i r2sim python3 - <<'EOF'`（漏 `-i` 静默无输出）；需 cwd 用 `-w /work/r2slam_ws`。
- 真实侧本机 `.venv`（py 3.13.14 / ORT 1.30.0 / torch 2.14.1+cpu / ultralytics 8.4.172 / onnx 1.23.1）；`wsl.exe`/`wmic.exe` 被拦（用 `tasklist //FO CSV`）。
- `Xvfb :99` 被 kill 不清 `/tmp/.X99-lock` ⇒ 相机不发、LiDAR 照常 ⇒ **就绪门必须查两个话题**。
- 短跑机理：ROS master 一断 `sim_ros1.py` 就 return（退出码 0、summary 完整）。**有效性门：`exit==0 && frames.csv 跨度 ≥ 0.97×目标`**（目标指 `--sim-duration`；`--duration` 是墙钟）。
- Python 的 `/tmp/x` 会落到**当前盘根**（`E:\tmp\x`），不是 Git Bash 的 `/tmp` ⇒ 跨工具传文件用项目内路径。

## 官方仿真器三坑
1. **LiDAR 不随车体旋转**（yaw=0）⇒ 航向不可观测。
2. **相机—LiDAR 外参只在 yaw≈0 成立**：`p_cam = Rz(-yaw) @ Tcl @ p_lidar`。
3. **顶墙时 LiDAR 原点落进墙体碰撞体 ⇒ 射线穿出打到房外地面**（`y<−8` 的 `seen&ground_ever` 格，66 跑里 22 跑有，最多 9370 格）。⇒ "雷达探到墙外世界"与"卡墙"同源，不是我们的 bug。
- 有效判据＝比对 `/gazebo/model_states`。**无 IMU/里程计 ⇒ 单跑间不能比分数。**

## 真值通道
`/root/.gazebo/models/house.world` 烘焙 model/link/collision 世界位姿 ⇒ `out/_truth_geo.py`（**必须传 `--map`**，否则静默 `return 0`）。COLLADA `up_axis` 不可信。**地图系＝世界系。**

## 感知 / 地图（定论）
- `labels.py`＝类别表唯一源；`OnnxYoloDetector` 自己加载 sidecar（`class_map` 默认哨兵 `_AUTO_CLASS_MAP`）；换模型**必须**跑 `tools/check_onnx_parity.py`。
- 语义地图：绿＝`ground_ever & ~obstacle`；**格子有回波才算 seen**。
- 障碍层：`MIN_VOTES=2`｜`LO/HI=0.35/2.00`｜`RELEASE 2/0.30`｜`OPEN_PX=3`｜`STRONG_VOTES=20`｜`HOLD="reset"`｜`GROUND_RATIO=1.0`。⚠️ 别再退回 `votes<12`。`free_space` 读 `wmap.ground` 不是 `blocked()`。`add()` 末句 `ground = ground_ever & ~obstacle` ⇒ ground 是 obstacle 的纯函数。
- 格索引 `floor((xy+extent)/resolution)`；真实 res 0.10/extent 60.0/n 1200，仿真 extent 30.0/n 600。**必须 floor。**

## 算法栈不变量
- `config.py`+`sources/` 是唯一知道"仿真/真实"处；**下游出现 `if sim` = 设计失败**。
- `--explore` 默认关。`online_pipeline.py`＝主程序；`package_submission.py`＝终审（**FAIL 绝不写 zip**）。**包里无代码**。真实视频必须 1.4× 加速。
- 一律用 `header.stamp`；`--npz` 复算须存 `control_yaw`；npz 是未去斜原始点云。改完两源各 5 帧冒烟；改签名必须真跑；比 md5。

## 探索算法 / 地图层封板
A\*（8 连通、半径腐蚀+障碍膨胀、**圆核**）；跟随＝三档油门+视线可达前视点；探索＝前沿簇+角扇区+`score=size/(1+d_start)`+`sweep`+`island`+`shadow`。目标＝观测位姿（`target_standoff_m=0.80`）。**`n_frozen_frames` 永远 0。** 四态 `drive`/`wedge`/`arc`/`turn`。
**⚠️ 状态机对地图细节极敏感 ⇒ 单跑会得相反结论。** 归档 76 跑的 `traj_xmin`：中位 −11.03、p75 −9.41 —— 好坏两峰并存。
⚠️ replay 无墙钟预算 ⇒ **验收必须闭环 + 报 `ms_*` 的 p90/max**。⚠️ **不要调参。**
已封板：D1/D3/D1+D3/`island_trigger_m2`/`barren_blacklist`/sweep·frontier 参数/价值函数。

## 已判死 / 登记不落地
- **不再重试**：`local_u`/`region_lambda`/`rfirst_gate`/`fb_relax`/`OBSTACLE_HOLD=total`/`debt_gain_m2`/黑名单/进度期限。`rfirst` 重开须有可见性论证。
- B3 停投。不值做：`V/(1+d)→V·exp(-λd)`。
- **登记不落地**：G7 形态学清理器/`fresh_split_m`/点云层判别器（在线 18 维留一 AUC 0.638<0.65 ⇒ 别再堆特征）。
- **K=2 已封板**（分的是回波密度不是真实性）。**FastInst-AA / UniPercepNet-S 判死**（自报 GPU 延迟已超预算）。
- 已装开关（默认关，grep 找）：`OBSTACLE_GROUND_RATIO`/`RENDER_*`/`disc_lambda`/`fresh_split_m`/`probe`/`sweep_value_mode`/`island_probe`/`strand_dump`/`--cf-hold-cells`/`--footprint`/`--sync-render`/`--seg-fuse`/`--seg-classes`。**出厂只有 numpy+cv2+onnxruntime ⇒ 诊断不许 `import scipy`。**

## 指标规则
- 权威口径＝`out/_room_cover2.py`（分母＝单调的 `ground_ever ∩ 框`）。**报百分比必须报分母。**
- ① count 型必须同时报 normalized rate；② "切换/放弃/重规划减少"类必须同时查完成量/覆盖 ⇒ **`switches` 永不单独出现**。
- **两个 n=1 的比较是抽签**；查参考集是否被挑过。**n=3v3 置换 p 下限 0.10 ⇒ 只能说"没分离"。**
- 算"行为不变"的 md5 **必须剔除 `traversable_m2` 类统计列**。跨源先 `repr()` 打原始值。报"掩码变化量"必须同时报格坐标。
- **★ `ms_*` 只能比"背靠背、同一会话"的一对**；同对里无关阶段也大幅变化 ⇒ 整对被污染。
- **★ 量之前先问"这是不是同一路输入"**：在渲染出来的 `camera|map` 拼图上跑检测器 ≠ 在相机帧上跑。

## 工程硬规则
- 会话开头补 PATH：`export PATH="/c/Users/ZhuanZ/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:$PATH"`。
- 含中文路径：存在性判断用 Python `os.path.exists`，不用 Git Bash 的 `ls/cd/find`。
- Gate −1：默认值与不传**必须 md5 逐位相同**；**哈希"决策列"不是整个文件**。r3 基线决策 md5 `94df03cd8a087fc9b15d2ff4086a5d7f`。
- **自检通过 ≠ 正确**；判据落在具体量上（坏在输入的开关：span/帧率/summary 照样全绿）。
- `pgrep -c -f` 算进调用者自己；`pkill` 不跳过父进程 ⇒ 清理写成脚本文件。⚠️ `pkill -x gz` 是错的。
- **有重试的驱动必须"每次尝试一份日志"。** 其余见技能同名小节。
