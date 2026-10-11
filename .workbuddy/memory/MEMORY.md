# SLAM 大师赛 — 项目长期记忆

> 只放**结论/判死/硬约束**。数字与推导全在 `docs/`，这里只留"结论 + 去哪查"。

## 文档地图
- **★ 交接（现役规格＋数据流＋清单；§12＝接手回合结果与更正）**：`docs/交接_20261010_仿真导航三修.md`
- 落定/双跑/交付包/分割：`docs/现役落定_双跑与交付包_20261010.md`（§5 分割、§7 并跑）
- 归因两轮：`docs/仿真退化归因与掩码接入融合_20261010.md` · `docs/仿真退化归因_第二轮_航向跳变与擦墙路径_20261010.md`（★三修依据）
- 评分：`docs/评分规范对照_20261010.md` · `docs/评审小灶_评分标准全面复评_20261010.md`
- 呈现/M3（ATE 旋钮 §3）：`docs/呈现层修复与M3复核_20261010.md`
- 语义地图：`docs/语义地图朝向与灰斑修复_20261009.md` · IMU：`docs/IMU启用_陀螺仪去斜与航向核对_20261010.md`
- 开放词汇（§32–37 YOLOE）：`docs/开放词汇边缘化_第一轮实测_20261009.md`
- 历史：`docs/HANDOFF_项目交接.md`（§9 过时，§5.2/§8 必读）· `docs/复盘_踩坑手册_小白向.md`（09-13 初赛版，10 月后未补）
- ⚠️ 技能 `robot-policy-replay-ab/SKILL.md` 已 ~200 KB，该按主题拆（未做）

## 归档结论（早期条目，细节查 docs）
- **航向**＝仿真退化根因 A：官方 LiDAR 不随车体旋转⇒**航向不可观测**，非 `heading.py` bug；同码两跑跳变次数相同、差在**恢复快慢**；`yaw_odom_deg` 恒 ~−0.8°＝无信息。
- **B3 收线停投**：成本门 PASS、效果门 FAIL；归因＝成本上界 `max-eval` 语义缺陷。
- **渲染**：`render()` 只画 `ground`，**画面可有颜色不能有第二判据**。灰斑已修（`OBSTACLE_GROUND_RATIO=1.0`、GCV=True）；旧"`ms_map` +46%"＝负载污染。
- **检测**：开放词汇默认 `yoloworld_alias_320.onnx`，**INT8 在 51 类失效⇒只能 fp32**；YOLOE 文本提示无效、视觉提示要 `11l-seg`@640；`detect_mapped` 只留 top-1⇒须报 `max`/`any≥x`/`top1`/`top1@conf`。换模型**必须**跑 `tools/check_onnx_parity.py`。
- **★ 最大结构性缺口＝没有尺子**：真实侧无真值⇒一切"检出/改善"都是自证的；最小可行＝人工标 30–50 帧。
- **算力**：边缘算力是规则要求，**先量预算**。`ms_render` 60.46→1.04；现役 detect 66.42 ms、seg 头 23.96⇒**分割头不是算力问题**。FastInst-AA/UniPercepNet-S 按自报 GPU 延迟判死⇒不装 MMCV。
- **实例层**判据被真值证伪（35.9%）⇒不再改几何阈值。**IMU 三臂无赢家不采纳**。★真收获：`tools/check_imu_heading.py`＝**真实侧第一条独立航向证据**。
- **评分（100 分）**：人物 30、G 25、M 40、V 5。
- **腐蚀核方→圆**：默认 `disc`（r=0.25 时与 `square` 重合，见第 20 条）；⚠️ `cv2.MORPH_ELLIPSE` 不能用。
- **M4 标签**：按点判；`h>2.0`→ceiling、否则 wall；新类 43/44 排物体类之后（`person=2` 不重编号）。
- **教训**：`diagnose_reachability.py` 曾有 10× 单位错⇒三份文档作废。**引用工具输出前先看它算的是什么。**

## 落定 / 分割（16–18）
16. **现役落定＋双跑＋出包**：出包前两红测试：`test_nav` 夹具把地板铺到墙底（加 `wall_occludes_floor()`）；`explore.py` 真 bug——区域秩 `c[22]` 后三处读者仍写 `c[-1]`⇒模块级 `CAND_REGION=22`。⚠️ 自检数 **258**（非 264），三修后 **261**。双跑：仿真 1436 帧/287.1 s、真实 1427 帧@10.27 Hz。交付包 md5 见 README／37 MB／0 FAIL 0 WARN。
17. **分割头并跑**：`r2slam/seg.py`＋`--seg-model/--seg-conf/--seg-alpha/--seg-classes/--seg-fuse`＋`summary["segmentation"]`；**检测器不动**，Gate＝真实 `map.pcd` 与交付包逐位同。320 原型 80×80⇒真实鱼眼一格≈20 px 台阶。闭集代价 `clock 74`⇒`--seg-classes` 默认 `person`。真实 frame31 检测器 `person 0.30` 落图而分割头没看见⇒**它是形状提供者，不是更好的人检测器**。`--seg-fuse` 默认关⇒pcd 逐位不变；**未证明更好**。⚠️ `w=int(round(x1-x0))≠int(round(x1))-int(round(x0))`⇒真实一帧即崩、仿真不复现⇒先裁整数像素再 resize。
18. **仿真“探索退化”＝不是代码回归**：同码两跑 `traj_xmin` **−6.60 vs −12.00**，`final` 其余指标全在 44 跑中游⇒唯一离群量是 `xmin`。⚠️ **原记的“t=249.6 航向单帧跳变 −104.9°”在数据里不存在**（复算最大单帧变化 17.8°、零帧 >60°；跳 151.8° 的是**目标方位**，东西换点）——已更正，见 doc §7.5。真形态＝t≈262 起在 x≈−6.5 的**极限环**（`drive`→`backoff`→**yaw 符号翻转两次**→`arc`→…），末 25 s 净西向位移≈0。三根因：A 航向（见上）B **A\* 擦墙路径** C **三套可达判据不一致**。反例 `b3l_OFF_r2` 末段 90 s 净位移 0.02 m 却 `xmin −12.66`⇒**“卡住”与“探得少”无因果**。

## 现役（19–23，2026-10-10 接手回合）

**★ 本轮决策（boss）：§1.2/§1.3 不采纳** —— 两项出厂即默认关 ⇒ 不采纳＝**什么都没变**（零成本零风险）；**§1.1 保留 `robot_radius=0.25`** —— 保守裕度不是修复（杠杆仅 1–2%）。见文档 §7.12。

19. **★★★ 三修已落地（`5cd8bbe`＋`82ed503`）**：① `robot_radius` **默认 0.19→0.25**；单一真相源 `eff_robot_radius`（explorer 与 `summary["reach"]` 同源）；`--robot-radius 0.19` 仍可逐位复现旧规划器。② `ahead_clear_use_obstacle_now`（默认 True）③ `backoff_realign`（默认 False）——后两个用 `--explore-set` 开。**真实侧价格**：可达 401.8 m²@0.19 → **392.7@0.25（−2.3%）** → 355.8@0.30（−11.5%）。**真实侧全跑两次 `map.pcd` md5 都是 `76025d9e519c4995e8a5fb5a5275758b`、`summary["map"]` 全字段复现** ⇒ 真实决策未被动。⚠️ **`summary["map"]["reachability"]` 随半径变 ⇒ 交付包需重建才与已验收 summary 一致（pcd/TF/视频不变）。**
20. **★★ r=0.25 时 `disc` 与 `square` 两核完全重合**（角偏移 0.212 < 0.25）⇒ **`--footprint` 在新默认下是空操作**，只在 r≤0.22 或 r≥0.28 才移动掩码。不是 bug（离散化表达不出），但会误导记得“方核→圆核”那一仗的人。已写进 `nav.py` 与 help。
21. **★ n≥3 纪律**：同码两跑 `backoffs` 就 17 vs 5 ⇒ **n=1 的比较是抽签**（`backoffs` 19→6 那类“看着变好”不算数）。
22. **★ 更正**：交接 §5.3(a) 的 Gate −1 配方两处错——`r2slam/out/*.npz` 是 `--dump-map` 的**地图栅格**（回放 `frames: 0` 且**退出码 0** 静默空跑），且 `replay_explore.py` **不认** `R2SLAM_PKG`（只 `test_nav`/`test_seg` 认）。正解：`dump_frames.py`＋旧包做**物理目录**；开关入口是 `--set`。
23. **★★★ 探索退化的真病＝贴墙时里程计坍缩（5%，不在 §1 覆盖内）**：脚本把 `all_3` 报成“停 24 s”，但**同窗口里程计净位移 0.032 m / 真值 5.308 m**、真值**穿过北墙到房外** ⇒ ★ **`frames.csv` 同时有估计列 `pose_*` 与真值列 `truth_*`，报“卡住”前先并排打出来**。根因：`registered` 是裁过 `[0.3,40]` 的**在距点数**（不是配准成功数），贴墙回波被裁 ⇒ ICP 沿行进方向退化 ⇒ 位姿冻结 ⇒ 从旧位姿/旧地图重选点 ⇒ 正反馈；**选点全程健康，是执行端瞎了**。★ **新尺子 `min(reg/points)` 40 跑零重叠**：0.53/0.59 两跑必冻结，≥0.90 的 33 跑必不冻结 ⇒ **2/40＝5%**；**归档 `xmin` 中位 −11.26 是正常天气**。⇒ **`park_n`/停滞是症状量不是故障量，报它必须带 `max err_xy` 与 `min reg/points`**。★ 反向：**只有 `a2`+`a3` 复合会贴墙**（2/4 vs 0/26，p≈0.014）⇒ 已决定不采纳。**n=3：`r19`/`r25` 都 PASS ⇒ 对照臂健康 ⇒ 本轮“未复现”≠“修好了”。** 全在 `docs/仿真探索恢复_判据与实跑_20261010.md` §7.7–§7.12。

## 挂起的问题（boss 说"后面再讨论"）
- **M5 ATE 旋钮**：`runtime.voxel_size` 0.50→0.25 ⇒ ATE p50 0.276→0.140 m，代价 `ms_odom` 15.3→53.9、p90 26→131。⚠️ ATE 主要是坏段计数⇒跨序列稳的是 RPE 与路径比。**默认仍 0.50**，见 `docs/呈现层修复与M3复核_20261010.md` §3。

## 家底/硬约束
- 唯一不可再得：`D://E-Downd//data.bag`（初赛 2532 MB，81 分）。
- 交付：实时＋整集视频＋自建 pcd；官方 pcd 只作提示；仿真 LiDAR yaw **不修**；输入只允许点云/图像/IMU；输出 pcd＋TF；建图 50%/感知 50%。
- 边缘算力是规则要求；类别表组委会不给；**真实侧无真值⇒任何 mIoU/识别率都是编的（要敢说“不知道”）**。
- **成果已归档 GitHub**：`solidsnailz36-cloud/openloong-semantic-slam`（**public**；根 README＝对外版）。手机号已脱敏（`156****6887`／11 处）＋**全史 `filter-branch` 重写** ⇒ ⚠️ **此前短 hash 引用全失效**（内容未变，按标题找）。
- ⚠️ **OpenLoong 赛题二（＝本项目）复赛未晋级**（10-10 22:19 群内出名单 / 官方公布日 10-13）：把“赛题一延期”外推成“赛题二也延期”⇒交了未修订版；**赛题一延期至 ~10-12 仍活着**。申诉口径：`docs/OpenLoong复赛_赛题二结果与申诉_20261011.md`。

## 环境
- 仿真＝容器 `r2sim`；`/work`=`D:/E-Downd/sim_official`；代码 `/work/r2slam_ws`。改完必须 `sync_to_container.sh`（手工白名单，已含 `frames.py`/`sources/*`/`viz.py`/`seg.py`/`replay_explore.py`/`yolo26n-seg_320.onnx`）。**漏文件的后果是静默的。** ⚠️ **`sync_to_container.sh` 不要加 `MSYS_NO_PATHCONV=1`**（`CreateFile E:\e:` 失败）——与只读复算相反。
- 容器：py3.8.10 / **ORT 1.16.3（只到 opset 19）** / cv2 5.0.0 / 无 `onnx` ⇒ **导出一律 `--opset 19`**。
- **杀仿真**：`docker exec r2sim bash /work/killbot.sh`。⚠️ 宿主跑 `kill_sim.sh`＝没跑。
- 只读复算：`export MSYS_NO_PATHCONV=1 && docker exec -i r2sim python3 - <<'EOF'`（漏 `-i` 静默无输出）；需 cwd 用 `-w /work/r2slam_ws`。
- 真实侧本机 `.venv`（py3.13.14/ORT 1.30.0/torch 2.14.1+cpu/ultralytics 8.4.172/onnx 1.23.1）；`wsl.exe`/`wmic.exe` 被拦（用 `tasklist //FO CSV`）。
- `Xvfb :99` 被 kill 不清 `/tmp/.X99-lock`⇒相机不发、LiDAR 照常⇒**就绪门必须查两个话题**。
- 短跑机理：ROS master 一断 `sim_ros1.py` 就 return（退出码 0、summary 完整）。**有效性门：`exit==0 && frames.csv 跨度 ≥0.97×目标`**（`--duration` 墙钟；**A/B 配平用 `--sim-duration`（传感器时间）**）。
- Python 的 `/tmp/x` 落到**当前盘根**⇒跨工具传文件用项目内路径。

## 官方仿真器三坑
1. **LiDAR 不随车体旋转**（yaw=0）⇒航向不可观测。
2. **相机—LiDAR 外参只在 yaw≈0 成立**：`p_cam = Rz(-yaw) @ Tcl @ p_lidar`。
3. **顶墙时 LiDAR 原点落进墙体碰撞体⇒射线穿出打到房外地面**（66 跑里 22 跑有）⇒"雷达探到墙外"与"卡墙"同源。
- 有效判据＝比对 `/gazebo/model_states`。

## 真值通道
`/root/.gazebo/models/house.world` 烘焙 model/link/collision 世界位姿⇒`out/_truth_geo.py`（**必须传 `--map`**，否则静默 `return 0`）。COLLADA `up_axis` 不可信。**地图系＝世界系。**只解析 `<box>` 会漏全部家具。

## 感知/地图（定论）
- `labels.py`＝类别表唯一源；`OnnxYoloDetector` 自加载 sidecar。
- 语义地图：绿＝`ground_ever & ~obstacle`；**格子有回波才算 seen**。
- 障碍层：`MIN_VOTES=2`|`LO/HI=0.35/2.00`|`RELEASE 2/0.30`|`OPEN_PX=3`|`STRONG_VOTES=20`|`HOLD=reset`|`GROUND_RATIO=1.0`。⚠️ 别再退回 `votes<12`。`add()` 末句 `ground = ground_ever & ~obstacle`。
- 格索引 `floor((xy+extent)/resolution)`；真实 res 0.10/extent 60/n 1200，仿真 extent 30/n 600。**必须 floor**。**行＝y、列＝x**：`row=floor((y+ext)/res)`（验证：正确映射 100% 命中、镜像 81.9%）。读 `grids_*` dump 别写 `n-1-...`。

## 算法栈不变量
- `config.py`+`sources/` 是唯一知道"仿真/真实"处；**下游出现 `if sim`＝设计失败**。
- `--explore` 默认关。`online_pipeline.py`＝主程序；`package_submission.py`＝终审（**FAIL 绝不写 zip**）。**包里无代码**。真实视频必须 1.4× 加速。
- 一律用 `header.stamp`；`--npz` 复算须存 `control_yaw`；npz 是未去斜原始点云。改完两源各 5 帧冒烟；改签名必须真跑；比 md5。
- ⚠️ **可达判据有三套（已确认缺陷，待统一）**：A\* `free_space` 用 `ground`（未知＝不可走）；`_ahead_clear` 用 `blocked()`＝`obstacle‖obstacle_now`（未知＝可走、**含本帧回波**）；`_lookahead` 的 LOS 用 `wmap.ground`；`_clear` 用 `blocked()`（未知＝可走⇒对远处目标几乎恒 True）。违反"同一规则只能有一份实现"⇒车朝"地图上可走"的点走而油门判死 ⇒ backoff ⇒ 摇摆。

## 探索算法/地图层封板
A\*（8 连通、半径腐蚀+障碍膨胀、**圆核**）；跟随＝三档油门+视线可达前视点；探索＝前沿簇+角扇区+`score=size/(1+d_start)`+`sweep`+`island`+`shadow`。目标＝观测位姿（`target_standoff_m=0.80`）。**`n_frozen_frames` 永远 0。**四态 `drive`/`wedge`/`arc`/`turn`；**`robot_radius` 默认 0.25**（旧 0.19 用 `--robot-radius 0.19` 复现）。
⚠️ 状态机对地图细节极敏感⇒单跑得相反结论。
⚠️ replay 无墙钟预算⇒**验收必须闭环＋报 `ms_*` p90/max**。**不要调参。**
已封板：D1/D3/D1+D3/`island_trigger_m2`/`barren_blacklist`/sweep·frontier 参数/价值函数。

## 已判死/登记不落地
- **不再重试**：`local_u`/`region_lambda`/`rfirst_gate`/`fb_relax`/`OBSTACLE_HOLD=total`/`debt_gain_m2`/黑名单/进度期限。`rfirst` 重开须有可见性论证。
- B3 停投。不值做：`V/(1+d)→V·exp(-λd)`。
- **登记不落地**：G7 形态学清理器/`fresh_split_m`/点云层判别器（在线 18 维留一 AUC 0.638<0.65⇒别再堆特征）。
- **K=2 已封板**。
- 已装开关（默认关，grep 找）：`OBSTACLE_GROUND_RATIO`/`RENDER_*`/`disc_lambda`/`fresh_split_m`/`probe`/`sweep_value_mode`/`island_probe`/`strand_dump`/`--cf-hold-cells`/`--footprint`/`--sync-render`/`--seg-fuse`/`--seg-classes`/`--robot-radius`/`--explore-set ahead_clear_use_obstacle_now=False`/`--explore-set backoff_realign=True`（**后两个已判不采纳**，见第 23 条）。**出厂只有 numpy+cv2+onnxruntime⇒诊断不许 `import scipy`。**

## 指标规则
- 权威口径＝`out/_room_cover2.py`（分母＝单调的 `ground_ever ∩ 框`）。**报百分比必须报分母。**
- ①count 型必须同时报 normalized rate；②"切换/放弃/重规划减少"类必须同时查完成量/覆盖⇒**`switches` 永不单独出现**。
- **两个 n=1 的比较是抽签**；查参考集是否被挑过。**n=3v3 置换 p 下限 0.10⇒只能说"没分离"。**
- 算"行为不变"的 md5 **必须剔除 `traversable_m2` 类统计列**。跨源先 `repr()` 打原始值。
- **★ `ms_*` 只能比背靠背同一会话的一对**；同对里无关阶段也大幅变化⇒整对被污染。
- **★ 量之前先问"这是不是同一路输入"**。
- **★ 归档跑 `frames.csv` 有真值列**（`truth_x/y`、`truth_yaw_deg`、`err_xy_m`）⇒无逐帧点云也能归因。

## 工程硬规则
- 会话开头补 PATH：`export PATH="/c/Users/ZhuanZ/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:$PATH"`。
- 含中文路径：存在性判断用 Python `os.path.exists`。
- **Gate −1：默认值/不传必须 md5 逐位相同**；哈希"决策列"不是整个文件。r3 基线决策 md5 `94df03cd8a087fc9b15d2ff4086a5d7f`。
- **同一规则只能有一份实现**（`mask_cell` 委派 `mask_cells`）；渲染"会动"＝有第二真相源的指纹。帧尺寸必须钉死（bag 有 2 帧宽度不同，编码器静默压缩）。
- **自检通过≠正确**；判据落在具体量上。
- `pgrep -c -f` 算进调用者；`pkill` 不跳过父进程⇒清理写脚本。⚠️ `pkill -x gz` 是错的。
- **有重试的驱动必须"每次尝试一份日志"**。
