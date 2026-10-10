# SLAM 大师赛 — 项目长期记忆

> 只放跨会话必须记住的**结论 / 判死 / 硬约束**。推导一律在 `docs/`，流程细则与踩过的坑在技能。

## 文档地图
- **先读**：`docs/交接_20261008_当前状态与后续计划.md`（状态 + P0/P1/P2）。
- **当前主线**：`docs/语义地图朝向与灰斑修复_20261009.md`（§6 渲染↔规划 / §7 靠墙灰带 / §10 比值 A/B / §12 边缘算力）。
- **★ 评分规范对照**：`docs/评分规范对照_20261010.md`（12 项逐条 + 缺口 + **§6 建议顺序**）。
- **IMU**：`docs/IMU启用_陀螺仪去斜与航向核对_20261010.md`（§0 结论 / §5.5–5.7 三臂）。
- 开放词汇：`docs/开放词汇边缘化_第一轮实测_20261009.md`（§32–§37 = YOLOE）。
- 收线/归因：`docs/B3成本剖析_20261008.md` §11–§13、`docs/漂移归因_20261008.md`、`docs/改进空间评估_20261009.md`。
- 历史：`docs/HANDOFF_项目交接.md`（§9 已过时；§5.2 类别表 / §8 限制必读）、`docs/复盘_踩坑手册_小白向.md`、`docs/探索算法_v2_设计与参考.md` §47–§71。
- ⚠️ 技能 `robot-policy-replay-ab/SKILL.md` 已 187 KB，该按主题拆（未做）。

## 结论索引
1. **漂移**：真实侧不显著（ATE p50 0.276 m / 102 m）；仿真侧＝少数帧被打偏不恢复，根因＝**官方仿真器航向不可观测（LiDAR 不随车体转）**，非 `heading.py` bug。
2. **B3 已收线停投**：成本门 PASS（**只看 `ms_map` p90**）；效果门 FAIL 撤回（里程 128.90→82.43 m，p=0.0047；`plan_failures` ×12.7）。归因＝成本上界 `max-eval` 语义缺陷。
3. **渲染↔规划一致性**：**`render` 只画 `ground`**（= `free_space` 读的同一数组）。**画面可以有颜色，不能有第二个判据。** 指标＝`summary["map"]["seen_not_classified_cells"]`（**归零即一致**）。
4. **灰斑 / 靠墙灰带**：`OBSTACLE_GROUND_RATIO=1.0`；**`GROUND_COVER_INFORMATIVE` 已是默认 `True`**（2026-10-09 晚 boss 拍，flag 反转 `--no-...`，旧拼写**直接报错**）。全量 1427 帧背靠背：决策 md5 逐位相同、障碍格 11860 未动、灰 6824→1464（−78.5%）、可通行 495.3→548.9 m²。★ 旧记录"`ground`+11% ⇒ `ms_map` **+46%**"**不成立**（整跑被机器负载污染）；背靠背重测 **−6.1%** ⇒ 代价测不出来。
5. **类别表自定：39 类 / 51 条 prompt**（`labels.py::_SPECS`）。含 3 个**不在 COCO 80** 的类（fire extinguisher / gym equipment / humanoid robot）+ 15 条别名。**"没标出来"= ①词表缺类 ②远处点云稀被 LiDAR 门拒 ③无实例层 ④YOLO 漏小目标**；换 3D 检测器只治 ③④。
6. **边缘算力是规则要求** ⇒ 顺序：① 先拿回预算 ② 零新模型的免费覆盖 ③ 需任意类别 → 换检测器 ④ 3D 大模型判死。**先量逐阶段预算，别加模型。** ★ **2026-10-10 已做①**：`render` 搬到 worker 线程 + `WorldMap.render_view()` 快照 ⇒ **`ms_render` p50 60.46→1.04 ms、`ms_total` 236.75→174.13**（**p50 进 200 ms 预算，p90 207.37 仍差 7 ms**，剩下 detect 58 + map 65 + odom 16）。★ **只在多核上有效**（worker 吃第二个核），单核边缘设备总 CPU 不变 —— 报告必须带这句。★ **seg 头撞预算不落**：实测 `yoloe-11s-seg`@640 原始推理 **159.0 ms**（现役 `yoloworld_alias_320` **40.6**），换上去 `ms_total≈293 ms`；`11l-seg` 581.6。
7. **实例层**（`instances.py`，默认关）：能力有、**判据被真值证伪** —— 家具 16.2% / 真人 19.3% / **只落墙 63.4%** / 空地 0.0% ⇒ **精度 35.9%**，召回 12/22。**在此之前不再改几何阈值。**
8. **开放词汇**：默认 `detect.py::DEFAULT_MODEL_FILE = yoloworld_alias_320.onnx`（51 类 fp32@320，opset 19）。唯一增量＝`fire extinguisher` 进地图。`gym equipment` / `humanoid robot` 检不到**不是措辞问题**（15 别名 + 单类隔离 @320/@640 全零 + 阳性对照 `person` 0.816）⇒ 只能换骨干。**INT8 在 51 类失效** ⇒ 只能 fp32。`detect_mapped` 只留 **top-1 类** ⇒ 必须报 `max`/`any≥x`/`top1`/`top1@conf` 四列。换检测器**只改检测派生列**。
9. **改进空间**：① **CLIP 喂的是裸名词**（`get_text_pe` 不加模板）—— 形式是杠杆（`a photo of a fire extinguisher` 0.942 vs 裸名 0.563）**但整表套是净回归**（框 124→386、`person` 115→3）⇒ **必须逐类标定门槛**。② **最大结构性缺口＝没有尺子**：真实侧无真值 ⇒ 一切"检出/改善"都是自证的；最小可行＝**人工标 30–50 帧**。
10. **★★ YOLOE —— 能力成立、能导 ONNX，卡的是预算不是可行性**（doc §32–§37）。文本提示仍 **0**；**视觉提示（给参照框、不给词）在 `yoloe-11l-seg` @640 上 10/10 命中**（人眼复核），`11s` 只 9/17 ⇒ **靠骨干容量**；**收紧参照框无效**。**烘焙成立**：`set_classes(embeddings=vpe)`，vpe **2.4 kB**、不用参照图；★ **vpe 是模型专属的**（跨权重必死）。⚠️ **seg 头通道 = `4+nc+32`** ⇒ `detect.py` 现在解不了这个导出（会抛 `IndexError`），要落地得先改成取 `[4:4+nc]`。免提示模式（4585 类）对 `robot` **0 命中**。⚠️ **更正**：上一版"导出把类头弄死"**是错的**（拿 640 PyTorch 比 320 ONNX + 用 11l 的 vpe 烤 11s）⇒ **导出验证必须同分辨率、同权重、同前置条件**。
11. **★★ IMU：铺通、标定、量过 —— 三臂无一是赢家，不采纳**（全部数字：`docs/IMU启用_陀螺仪去斜与航向核对_20261010.md`）。以前没用的原因＝`deskew.py` 里**一条过期注释**（那是初赛包 `three1` 的事实；`r2` 的 `/odin1/imu` 是 **55,627 条 @399.9 Hz 无 gap**）。动机：扫描内旋转 CV 0.333° vs 陀螺 0.229°、**|CV−IMU| p50 0.205°**（5 m 处 1.8 cm）⇒ 误差与信号同量级。三臂（ATE p50 0.276/0.300/**0.352**；1 秒 RPE max 0.256/0.715/**0.209**；轨迹 102.54/103.58/102.64）⇒ 去零偏**不是空操作、且修好 RPE 尾巴，但 ATE 更差；无臂全面占优 ⇒ 默认仍走定速**。疑点＝**roll/pitch**（逐轴相关 z 0.986 但 x/y 0.45，且无第二条独立参照）。⚠️ n=1 对。**★ 真收获**：`tools/check_imu_heading.py` = **真实侧第一条独立航向证据**（陀螺 −354.1° / 我们 −363.0° / 厂商 −361.2°）。
12. **★★★ 评分规范（官方 100 分表）—— 它改变了优先级**。**P1/P2/P3 全是"人物"共 30 分**；G1–G3 25；M1–M5 40；V 5。★ **G2 要"显式区分可通行/障碍/未知"且"仅地面分割计部分"** ⇒ 我们正好三分类 ⇒ 那个默认翻转在分数口径下是真金。★ **M5 唯一按公式计分**：**ATE + 1 秒 RPE + 时间覆盖率**（**前五轮报的 5 秒 RPE 不对应规范任何输入**）；现状 ATE p50 **0.276 m**、1 秒 RPE p50 **0.041**、覆盖率 **1427 帧 @10.27 Hz、dt p99 0.0975 s 无空洞、99.9%**。⚠️ 尺子是厂商里程计不是官方真值；工具 `compare_vendor.py --horizons 1,5`。★★ **投入与分数不匹配**：开放词汇三轮只影响 **M4（10 分）**；**半天的呈现层改动直接对着 V + P1（5 分以上）**。
13. **★★★ 腐蚀核：方核 → 圆核（2026-10-10，boss 指 `stranded_map.png` 白圈）**。**结论：那块 3.14 m² 玻璃房地板被我们自己的方核误杀。** 机理：障碍 `(7.35,2.05)` 48 票 → `solid` 3×3 把邻格变成 1 格空洞 → 5×5 **方核**投出 5×5 阴影盖住门口；而门口格到障碍**表面 0.212 m > 半径 0.19 m**，**本就不该删**（方核对角半宽 0.283 m）。★ **核必须按运动学条件生成**：`res·hypot(max(|a|-0.5,0), max(|b|-0.5,0)) < r` ⇒ `r=0.19/res=0.10` 时正好是「5×5 减四角」21 格。⚠️ **`cv2.MORPH_ELLIPSE` 不能用**（5×5 椭圆 17 格，会放行 0.158 m 间隙的格）。**结果**：`reachable 392.7→401.8 m²`、`stranded 159.6→150.5`，其中**真·另一分量 3.61→0.67 m²**；**地图列与决策列 md5 逐位不变**（真机 `explore:false` 只改报告）；**代价在噪声里**（同臂重复性 4.6–5.6%，两臂差 −0.2%/+0.9%）。**默认已翻成 `disc`**，`--footprint square` 可逐位复现旧产物。⚠️ **仿真策略行为会变**（1229 帧 A/B：picks 17→23、deep 0→2、arrivals 同 4；23 列里 12 列不同），**n=1 只能说"变了"**。`tools/test_footprint.py` 16 项。
14. **⚠️ 更正：`diagnose_reachability.py` 的"外壳检验"有 10× 单位错**（`k_shell=round(2.0/res)`=20 格=**2.0 m** 邻域，打印却写 `<= 0.20 m`）⇒ 三份文档里"98.9% 落在 0.20 m 以内"**全部更正**。**更根本的是不该用距离阈值**（一格被拿掉可能只是因为旁边是**未知**区）。改成无阈值精确分解：`stranded = (ground & ~free) ∪ (ground & free & lab != seed)` ⇒ 方核 155.95（97.7%）/ **3.61（2.3%）**，圆核 149.86（99.6%）/ **0.67（0.4%）**。★ 教训：**引用工具输出的数字前先看它算的是什么**（这和"读错通道"是同一类错）。
15. **★★ M4 标签口径（2026-10-10）**：`label` 列原来**按格**判（"脚下这格是不是地面"）⇒ **天花板点继承 `traversable`**（实测 z 到 3.43 m）。改成按点：`traversable.geometric_label(h, ground_mask, obj_mask)`，`h>2.0`→`ceiling`、`0<h≤2.0`→`wall`、`obj_mask` 内→`background`。新类 id **43/44，排在物体类之后**（`person=2` 及家具类不重编号 ⇒ 交付的 pcd 与 `label_table.yaml` 继续对得上）。**结果**：`background 48.0%→0.38%`、`traversable 48.8%→22.4%`、`wall 26.6%`、`ceiling 47.5%`。★ 非 person **具名物体仍只有 263 点（0.104%）—— 那是检测器能力，标签口径修不了**。`tools/test_geometric_label.py` 15 项。

## 渲染（定论）
- `render()` 只画 `ground`。**画面可以有颜色，不能有第二个真相源。**
- "会动"是渲染 bug 的指纹。**面板俯视图：+x 右、+y 上**（曾上下镜像 ⇒ 左右互换）。**灰斑 = 画面在说谎。**
- 写渲染规则按语义写不按形状写。帧尺寸必须钉死（bag 有 2 帧宽度不同，编码器静默压缩）。
- ⚠️ 复述实现的测试抓不到实现里的约定错误。

## 挂起的问题（boss 明确说"后面再讨论"—— 他回来时必须能立刻接上）
1. **★ "M5 的 ATE 旋钮"（2026-10-10 挂起）**：`runtime.voxel_size` 从 0.50 换到 0.25，
   **ATE p50 0.276→0.140 m**（1 秒 RPE 0.041→0.027，路径比 1.039→1.018），
   且有**外部佐证**（对官方 pcd：对齐 yaw +0.00° vs +2.00°、≤0.1 m 占 97.8% vs 96.7%）；
   **但代价是 `ms_odom` p50 15.3→53.9 ms（×3.5）、p90 26→131 ms**（独占机器两对背靠背，重复性 2–3%），
   换的话**必须同时把 `render`(58.6 ms) 移出关键路径**。
   ⚠️ **"ATE 腰斩"要打折**：`tools/ate_tail.py` 证明这条序列的 ATE 主要是"几段坏段的计数"
   （0.50 的 41.9% 帧超 0.30 m、**一段 36.7 s 占平方误差 56.2%**；0.25 只剩 8 帧）。
   ⇒ 有机理、跨序列稳的是 RPE 与路径长度比；**默认值仍是 0.50，未改配置**。
   三个候选：①换 0.25＋render 出关键路径 ②折中 0.30 ③保持 0.50。
   全部数字与图：`docs/呈现层修复与M3复核_20261010.md` §3。

## 家底 / 硬约束
- 唯一不可再得：`D://E-Downd//data.bag`（初赛 2532 MB，81 分）。
- 交付：实时 + 整集视频 + 自建 pcd；官方 pcd 只作提示；仿真 LiDAR yaw **不修**；输入只允许点云/图像/IMU；要输出 pcd + TF；建图 50% / 感知 50%。
- 边缘算力是规则要求；类别表组委会不给；**真实侧无真值 ⇒ 任何 mIoU / 识别率都是编的（要敢说"不知道"）。**

## 环境
- 仿真＝Docker 容器 `r2sim`；`/work` = `D:/E-Downd/sim_official`；容器内代码 `/work/r2slam_ws`。产物落宿主盘。**改完必须 `sync_to_container.sh`**。`PKG_PATHS` 手工白名单（2026-10-10 已补 `frames.py` + `sources/*` + **`viz.py`** + `tools/replay_explore.py`）。⚠️ **白名单漏文件的后果是静默的**：`viz.py` 改过却不在名单里，容器里跑的是旧 viz，而新的 `online_pipeline` 调 `viz.banner_size` ⇒ 第一帧就 `AttributeError`，而 copy 步骤什么都不报。
- **★ 容器运行时**：py 3.8.10 / **`onnxruntime 1.16.3`（只保证到 opset 19，20 直接拒收）** / cv2 5.0.0 / **无 `onnx` 模块** ⇒ **导出模型一律 `--opset 19`**。
- **★ 杀仿真**：容器里 **`docker exec r2sim bash /work/killbot.sh`**。⚠️ `vendor/sim_scripts/kill_sim.sh` 在宿主上跑等于没跑（pkill 的是宿主进程，还照印 `done`）。
- 只读复算：`export MSYS_NO_PATHCONV=1 && docker exec -i r2sim python3 - <<'EOF'`（漏 `-i` 静默无输出）；**需要 cwd 时用 `docker exec -w /work/r2slam_ws`**。
- 真实侧本机 `.venv`（py 3.13.14 / ORT 1.30.0）；`wsl.exe` / `wmic.exe` 被拦（用 `tasklist //FO CSV`）。
- `Xvfb :99` 被 kill 不清理 `/tmp/.X99-lock` ⇒ 相机不发、LiDAR 照常 ⇒ **就绪门必须查两个话题**。
- 短跑机理：`sim_ros1.py:122` 循环条件 `not rospy.is_shutdown()` ⇒ ROS master 一断就 `return`（退出码 0、summary 完整）。**有效性门：`exit==0 && frames.csv 跨度 ≥ 0.97×目标`。**
- "Docker 死机"不是容器 OOM；`/work` 是 bind mount ⇒ 产物全在。

## 官方仿真器两坑
1. **LiDAR 不随车体旋转**（yaw=0）⇒ 航向不可观测。判航向先看 `yaw_odom_deg`（不转）/ `yaw_deg` / `truth_yaw_deg` 各属哪条通道。
2. **相机—LiDAR 外参只在 yaw≈0 成立**：`p_cam = Rz(-yaw) @ Tcl @ p_lidar`。
- 有效判据＝与 `/gazebo/model_states` 比对。**无 IMU / 里程计 ⇒ 单跑间不能比分数。**

## 真值通道
`/root/.gazebo/models/house.world` 烘焙 model/link/collision 世界位姿 ⇒ `out/_truth_geo.py`（**必须传 `--map`**，否则静默 `return 0`）。COLLADA 的 `up_axis` 不可信（按底部贴地自适应）。**地图系＝世界系。**

## 感知 / 地图（定论）
- `labels.py` ＝类别表唯一源；`OnnxYoloDetector` **自己加载旁边 sidecar**（`class_map` 默认哨兵 `_AUTO_CLASS_MAP`，显式 `None` 才当 COCO）；换模型**必须**跑 `tools/check_onnx_parity.py`。
- 语义地图：绿 = `ground_ever & ~obstacle`；**格子有回波才算 seen**。
- 障碍层（`worldmap.py`）：`MIN_VOTES=2`｜`LO/HI=0.35/2.00`｜`RELEASE_FRAMES/RATIO=2/0.30`｜`OPEN_PX=3`｜`STRONG_VOTES=20`｜`HOLD="reset"`｜`GROUND_RATIO=1.0`｜`blocked()=obstacle|obstacle_now`。⚠️ **别再退回 `votes<12`**。`free_space` 读 `wmap.ground` 不是 `blocked()`。`add()` 末句 `ground = ground_ever & ~obstacle` ⇒ **ground 是 obstacle 的纯函数**。
- 格索引：`floor((xy+extent)/resolution)`；真实 res 0.10 / extent 60.0 / n 1200，仿真 extent 30.0 / n 600。**必须 floor。**

## 算法栈不变量
- `config.py` + `sources/` 是唯一知道"仿真/真实"处；**下游出现 `if sim` = 设计失败**。
- `--explore` 默认关。`online_pipeline.py` ＝主程序；`package_submission.py` ＝终审（**FAIL 绝不写 zip**）。
- 当前提交包 md5 `b8cb38957c64f379277f4fa6148b1a31` / 37,481,494 B（2026-10-08，0 FAIL/0 WARN）。此后改动未进包。**包里无代码。** 真实视频必须 1.4× 加速。
- 一律用 `header.stamp`；`--npz` 复算须存 `control_yaw`；npz 是未去斜原始点云。改完两源各 5 帧冒烟；改签名必须真跑；比 md5。

## 探索算法 / 地图层封板
规划＝A\*（8 连通、半径腐蚀 + 障碍膨胀）；跟随＝三档油门 + 视线可达前视点；探索＝前沿簇 + 角扇区 + `score=size/(1+d_start)` + `sweep` + `island` + `shadow`。目标＝**观测位姿**（`target_standoff_m=0.80`）。**`n_frozen_frames` 永远 0。** 四态 `drive`/`wedge`/`arc`/`turn`。**状态机对地图细节极敏感 ⇒ 单跑会得相反结论。**
已封板：D1 / D3 / D1+D3 / `island_trigger_m2` / `barren_blacklist` / sweep·frontier 参数 / 价值函数。⚠️ replay 无墙钟预算 ⇒ **验收必须闭环 + 报 `ms_*` 的 p90/max**。⚠️ **不要调参。**

## 已判死 / 登记不落地
- **不再重试**：`local_u` / `region_lambda` / `rfirst_gate` / `fb_relax` / `OBSTACLE_HOLD=total` / `debt_gain_m2` / 黑名单 / 进度期限。`rfirst` 重开**须有可见性论证**。
- B3 停投。不值做（非判死）：`V/(1+d) → V·exp(-λd)`。
- **登记不落地**：G7 地图形态学清理器 / `fresh_split_m` / 点云层判别器（在线 18 维留一 AUC **0.638 < 0.65** ⇒ 观测层信息不足，**别再堆特征**）。
- **K=2 已封板**（分的是回波密度不是真实性：墙丢 4.7% / 家具丢 58.6%）。
- 已装开关（默认关，grep 找）：`OBSTACLE_GROUND_RATIO` / `RENDER_*` / `disc_lambda` / `fresh_split_m` / `patch_probe` / `sweep_value_mode` / `island_probe` / `strand_dump` / `--cf-hold-cells`。`_fold` 统一三处折叠。**出厂运行时只有 numpy+cv2+onnxruntime ⇒ 诊断不许 `import scipy`。**

## 指标规则
- 权威口径＝`out/_room_cover2.py`（分母＝单调的 `ground_ever ∩ 框`）。**报百分比必须报分母。**
- **项目级硬规则**：① count 型指标必须同时报 normalized rate；② "切换/放弃/重规划减少"类必须同时查完成量/覆盖 ⇒ **`switches` 永不单独出现**。
- **两个 n=1 的比较是抽签**；查参考集是否被挑过。**n=3v3 置换 p 下限 0.10 ⇒ 只能说"没分离"。**
- 算"行为不变"的 md5 **必须剔除 `traversable_m2` 类统计列**。
- 两源量纲：replay `yaw` **弧度** / online `yaw_deg` **度** ⇒ 跨源先 `repr()` 打原始值。跨仪器对帧只用 `stamp`。报"掩码变化量"必须同时报格坐标。
- **★★ `ms_*` 只能比"背靠背、同一会话"的一对。** 判据：**同对里与本改动无关的阶段（`deskew`/`odom`/`detect`）也大幅变化 ⇒ 整对被机器负载污染，不能归因给这个开关**（那条 +46% 就是这样来的，重测是 −6%）。

## 工程硬规则
- 会话开头补 PATH：`export PATH="/c/Users/ZhuanZ/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:$PATH"`。
- 含中文路径：存在性判断用 Python `os.path.exists`，不用 Git Bash 的 `ls/cd/find`。
- Gate −1：默认值与不传**必须 md5 逐位相同**，且与历史 md5 相同；**哈希"决策列"不是整个文件**。r3 基线决策 md5 = `94df03cd8a087fc9b15d2ff4086a5d7f`。其余门槛见技能。
- **自检通过 ≠ 正确**；判据落在**具体量**上。⚠️ 坏在输入的开关：span/帧率/summary 照样全绿。
- `pgrep -c -f` 算进调用者自己；`pkill` 跳过自己但不跳过父进程 ⇒ `docker exec bash -lc '...pkill...'` **会杀掉执行它的 shell**。清理写成脚本文件。⚠️ `pkill -x gz` 是错的。
- **有重试的驱动必须"每次尝试一份日志"**。
- 其余（docker / 后台 / Edit 纪律 / 守恒量 / 事件 vs 帧 / 参数扫描 / commit）见技能同名小节。
