# SLAM 大师赛 — 项目长期记忆

> 只放跨会话必须记住的结论 / 判死 / 硬约束。推导在 `docs/`，流程细则与踩过的坑在技能 `robot-policy-replay-ab`（新会话先看它）。

## 文档地图
- **先读**：`docs/交接_20261008_当前状态与后续计划.md`（当前状态 + P0/P1/P2）。
- **当前主线**：`docs/语义地图朝向与灰斑修复_20261009.md`（§6 渲染↔规划 / §7 靠墙灰带 / §10 OBSTACLE_GROUND_RATIO A/B / §11 3D 检测调研 / §12 边缘算力重排）。
- 开放词汇：`docs/开放词汇边缘化_第一轮实测_20261009.md`。
- 收线/归因：`docs/B3成本剖析_20261008.md` §11–§13、`docs/漂移归因_20261008.md`。
- 历史：`docs/HANDOFF_项目交接.md`（初赛；§9 已过时，§5.2 类别表 / §8 限制必读）、`docs/复盘_踩坑手册_小白向.md`、`docs/探索算法_v2_设计与参考.md` §47–§71。
- ⚠️ 技能 `robot-policy-replay-ab/SKILL.md` 已 187 KB，该按主题拆（未做）。

## 结论索引
1. **漂移**：真实侧不显著（ATE p50 0.276 m/102 m）；仿真侧＝少数帧被打偏且不恢复，根因＝**官方仿真器航向不可观测（LiDAR 不随车体转）**，非 `heading.py` bug。
2. **B3 已收线停投**：成本门 PASS（`ms_map` p90 OFF ≤31.4 / ON ≥94.9 ms，**只看 p90**）；效果门 FAIL 撤回（里程 128.90→82.43 m p=0.0047；`plan_failures` ×12.7 ⇒ 规划器在走不通的地图上反复失败）。归因＝成本上界 `max-eval` 语义缺陷；无上界消融跑不完。`OBSTACLE_TOPO_FUSE_M2` 默认 0.0。
3. **渲染↔规划一致性**（`b0bd756`）：曾有空类（`seen & ~ground & ~obstacle`，6466 格）被画成障碍 ⇒ 一格两个答案。定论：`GROUND_COVER_INFORMATIVE` + 删渲染补丁，**`render` 只画 `ground`（= `free_space` 读的同一数组）**。**画面可以有颜色，不能有第二个判据。** 默认仍 False；指标 `summary["map"]["seen_not_classified_cells"]` 归零即一致。
4. **靠墙灰带**（`fe9e0a3`）：`ground = ground_ever & ~obstacle` 非单调 ⇒ 障碍 178.4 m² 里 120.9 曾为地面；`OBSTACLE_RELEASE` 永不触发。修法 **`OBSTACLE_GROUND_RATIO=1.0`**（ground 489→549、obstacle 178→118.5 m²）。安全：放掉的 60 m² 里 99.8% 是矛盾格，纯墙 votes≥20 一格没碰。闭环 A/B 仿真 n=3v3 过；**默认已改 1.0**（boss 批准）。Gate −1 已验 md5 `b664e5cfb6f76076`。
5. **类别表自定，39 类 / 51 条 prompt**：组委会不给列表 ⇒ `labels.py::_SPECS` 是我们的选择。已追加 13 COCO 类（零推理成本）+ 几何类（`unclassified object` / `railing`）+ 3 个**不在 COCO 80** 的类（38 `fire extinguisher` / 39 `gym equipment` / 40 `humanoid robot`）＋ 15 条别名 prompt。旧 id 全稳定。**"没标出来"= ①词表缺类 ②远处点云稀被 LiDAR 门拒 ③无实例层 ④YOLO 漏小目标**；换 3D 检测器只治 ③④。★ **措辞救不了模型看不见的类**（见索引 8 第五轮）。
6. **边缘算力是规则要求** ⇒ 顺序重排：① 先拿回预算（render 下关键路径 → 地图层平方律，都要 A/B 门；算数代价 p50 170–264 ms vs 5 Hz 预算 200 ms；render 占 24–35% 且非算法）② 零新模型的免费覆盖（实例层 + 扩词表 + 几何类）③ 需任意类别 → 2D YOLO-World **替换** yolo26n（在线成本为零）④ 3D 大模型判死、IGLOSS 降级。**先量逐阶段预算，别加模型。**
7. **实例层**（`r2slam/instances.py`，默认关）——能力有、**判据被真值证伪**：BEV 栅格 + `connectedComponents` + `reduceat`，3.7 ms/帧（合成）、p50 9.5 ms（全量）。行为中性已证（决策 md5 `b664e5cfb6f76076`）。真值：家具 16.2% / 真人 19.3% / 只落墙 63.4% / 空地 0.0% ⇒ **精度 35.9%**；召回 12/22=55%。错误单向（on nothing=0%）⇒ 判据目标＝提高"落家具上"占比、保持"落空地"≈0。**在此之前不再改几何阈值。**
8. **⑤ 开放词汇**（doc + `823b559`、A 方案 `9cb03bc`）：
   - 词表真相：烘焙 36 条 prompt 全是 COCO 类；`yolo26n` 本就是 COCO 80 ⇒ **换 YOLO-World 词表 80→36 是负收益**。q320 多检出真因是 backbone 更强。
   - 接线缺陷已修：`online_pipeline.py` 从未传 `class_map` ⇒ idx 1–35 全 → `None` 静默丢弃。Gate −1 只差 10 个 `ms_*`。
   - 顺带修 `detect.py` 硬编码 640：新增 `_declared_side()`，**图的形状是权威**。
   - INT8 零检出根因：输出端 `Concat` 共用尺度 ⇒ 类别通道压成 0；修法 `quantize_onnx.py --exclude-tail 2`。**量化只值 1.35–1.42×，分辨率才是主杠杆。**
   - 全量 1427 帧：q320 **45.3 ms / 13.31 MB**；拒收率 22.3% vs yolo26n 23.7%；`seen_cells` 均 68215。320 丢"小且静止"物体（丢了 `suitcase`）。
   - **A 方案结果**：加 38 `fire extinguisher` / 39 `gym equipment` / 40 `humanoid robot` ⇒ 39 类。**`fire extinguisher` 6/6 检出（人眼确认），`yolo26n` 3 帧瞎、1 帧误判 person**；零回归。**`gym equipment` / `humanoid robot` 松门也是 0 ⇒ CLIP 无响应。** id 41/42 = `unclassified object` / `railing`（全库符号引用，安全）。
   - `floor mat` 故意没加（厚度 < 0.2 m 门）。⚠️ 全部 CPU 本机数；**Orin 逐阶段预算仍未做**。
   - **★ 第五轮（boss 选 3）—— 措辞无效、量化在 51 类上失效、但都不需要**（doc §21–§27）：
     `labels.PROMPT_VARIANTS`（按**名字**索引）＋ `prompt_list()` 支持**一个 label 烤多条 prompt**（`index_to_label` 多对一，canonical 保留 ⇒ 前 37 索引逐位不变、两代可比）⇒ `yoloworld_alias_320.onnx`（51 类）。
     **① 措辞无效**：15 条别名只有 `stationary bicycle` 响应（`max` 0.187 < 0.30 门）；强开门唯一框是**画面边缘假框**。
     `humanoid robot` 四条措辞 **max ≤ 0.011**，**而目标确实在帧里**（帧 196 白色圆头人形机器人）⇒ 上一轮"被 person 压住"**证伪**，是**模型能力**问题。
     ⚠️ **仪器教训**：`detect_mapped` 只留 **top-1 类** ⇒ 只看 label 计数会把"有响应但没赢"读成"没响应"；必须报 **`max`/`any≥x`/`top1`/`top1@conf`** 四列（新工具 `tools/probe_openvocab.py`）。
     **② 量化在 51 类上失效**：fp32 0.62 → int8 0.43 框/帧（−31%）；同套量化在 36 类上无害（0.68）；`--exclude-tail 4/6` 无效、`--per-channel` 反而多出 `potted plant` 假框 ⇒ **51 类只能 fp32**；而 fp32 detect p50 **62.2 ms 比现役 64.6 快** ⇒ 不需要量化。
     **③ 全量 1427 帧**：`fire extinguisher` **0 → 25 观测 / 5 实例 / 112 点**（首次进地图）；`clock`/`handbag` 也有；代价＝丢 `suitcase`/`bottle`（320 小目标，与 §15 一致，非别名造成）。
     ★ **位姿/几何列 1427 帧逐位相同**，差异只在检测派生列 ⇒ **换检测器只改标签**。★ **确定性门**：yolo26n 重跑 vs 昨日 **决策 md5 逐位相同 `9b2b284631d9acb4fb96a22acad1d851`**。★ 顺带纠正：yolo26n 的 `suitcase`(-16.25,0.73) 其实是**红色消防器材箱**，新模型标 `fire extinguisher` 更对。
     ⇒ **账"正但小"，换不换进交付＝"多一个类 vs 丢 suitcase"，等 boss 拍。** 新工具 `out/_alias_ab.py`；新技能 `openvocab-detector-attribution`。
9. **§73**：报覆盖必须同时报 `wall_s` 与 `sim_span`（RTF）。

## 渲染（定论）
- `render()` 只画 `ground`。**画面可以有颜色，不能有第二个真相源。**
- "会动"是渲染 bug 的指纹。**面板俯视图：+x 右、+y 上**（曾上下镜像 ⇒ 左右互换）。**灰斑 = 画面在说谎。**
- 写渲染规则按语义写不按形状写。帧尺寸必须钉死（bag 有 2 帧宽度不同，编码器静默压缩）。
- ⚠️ 复述实现的测试抓不到实现里的约定错误。

## 家底 / 硬约束
- 唯一不可再得：`D://E-Downd//data.bag`（初赛 2532 MB，81 分）。
- 交付：实时 + 整集视频 + 自建 pcd；官方 pcd 只作提示；仿真 LiDAR yaw **不修**；输入只允许点云/图像/IMU；要输出 pcd + TF；建图 50% / 感知 50%。
- 边缘算力是规则要求；类别表组委会不给；**真实侧无真值 ⇒ 任何 mIoU / 识别率都是编的（要敢说"不知道"）。**

## 环境
- 仿真＝Docker 容器 `r2sim`；`/work` = `D:/E-Downd/sim_official`；容器内代码 `/work/r2slam_ws`。产物落宿主盘，不必 `docker cp`。**改完必须 `sync_to_container.sh`**。`PKG_PATHS` 是手工白名单，`replay_explore.py` 不在 ⇒ 单独 cp。`out/*.py` 跑法 `cd /work/r2slam_ws && python3 out/xxx.py`。
- 只读复算：`export MSYS_NO_PATHCONV=1 && docker exec -i r2sim python3 - <<'EOF'`（漏 `-i` 静默无输出）。
- 真实侧本机 `.venv`；`wsl.exe` / `wmic.exe` 被拦（用 `tasklist //FO CSV`）。
- `Xvfb :99` 被 kill 不清理 `/tmp/.X99-lock` ⇒ 相机不发、LiDAR 照常 ⇒ **就绪门必须查两个话题**。
- "Docker 死机"不是容器 OOM；`/work` 是 bind mount ⇒ 产物全在。
- 短跑机理：`sim_ros1.py:122` 循环条件 `not rospy.is_shutdown()` ⇒ ROS master 一断就 `return`（退出码 0、summary 完整）。**有效性门：`exit==0 && frames.csv 跨度 ≥ 0.97×目标`。**

## 官方仿真器两坑
1. **LiDAR 不随车体旋转**（yaw=0）⇒ 航向不可观测。判航向先看 `yaw_odom_deg`（不转）/ `yaw_deg` / `truth_yaw_deg` 各属哪条通道。
2. **相机—LiDAR 外参只在 yaw≈0 成立**：`p_cam = Rz(-yaw) @ Tcl @ p_lidar`。
- "反投影回原框"自检抓不到；有效判据＝与 `/gazebo/model_states` 比对。**无 IMU / 里程计 ⇒ 单跑间不能比分数。**

## 真值通道
`/root/.gazebo/models/house.world` 烘焙 model/link/collision 世界位姿 ⇒ `out/_truth_geo.py`（**必须传 `--map`**，否则静默 `return 0`）。COLLADA 是 XML，`up_axis` 不可信（按底部贴地自适应）。**地图系＝世界系。**

## 感知 / 地图（定论）
- `labels.py` ＝类别表唯一源；检测器 `yolo26n.onnx`（9.43 MB，回退 `yolo11n`），换模型**必须**跑 `tools/check_onnx_parity.py`。
- 语义地图：绿 = `ground_ever & ~obstacle`；**格子有回波才算 seen**。
- 障碍层（`worldmap.py`）：`MIN_VOTES=2`（数帧）｜`LO/HI=0.35/2.00`｜`RELEASE_FRAMES/RATIO=2/0.30`｜`OPEN_PX=3`｜`STRONG_VOTES=20`｜`HOLD="reset"`｜`MIN_POINTS=1`｜**`GROUND_RATIO=1.0`**｜`GROUND_COVER_INFORMATIVE=False`｜`blocked()=obstacle|obstacle_now`。⚠️ **别再退回 `votes<12`**。`obstacle_now` 写入 bug 已修（行为零变化）。`free_space` 读 `wmap.ground` 不是 `blocked()`。`add()` 末句 `ground = ground_ever & ~obstacle` ⇒ **ground 是 obstacle 的纯函数**。
- 格索引：`floor((xy+extent)/resolution)`；真实 res 0.10 / extent 60.0 / n 1200，仿真 extent 30.0 / n 600。`cell_of` 返回 `(cx,cy)`。**必须 floor。**

## 算法栈不变量
- `config.py` + `sources/` 是唯一知道"仿真/真实"处；**下游出现 `if sim` = 设计失败**。
- `--explore` 默认关（仿真不传机器人不动但日志健康 ⇒ 判据落在"真的动过"）。
- `online_pipeline.py` ＝主程序；`package_submission.py` ＝终审（**FAIL 绝不写 zip**）。
- 当前提交包 md5 `b8cb38957c64f379277f4fa6148b1a31` / 37,481,494 B（2026-10-08 16:16，0 FAIL/0 WARN）。此后改动未进包。**包里无代码。** 选跑依据：权威 `out/_room_cover2.py` 下 subr2=32% / subr3=50%（覆盖 vs 漂移，两边 n=1 待 boss 拍）。真实视频必须 1.4× 加速；`vendor_check.json` 只存于包内。
- 一律用 `header.stamp`；`--npz` 复算须存 `control_yaw`；npz 是未去斜原始点云。改完两源各 5 帧冒烟；改签名必须真跑；比 md5。

## 探索算法
规划＝A\*（8 连通、半径腐蚀 + 障碍膨胀）；跟随＝三档油门 + 视线可达前视点；探索＝前沿簇 + 角扇区 + `score=size/(1+d_start)` + `sweep` + `island` + `shadow`。目标＝**观测位姿**（`target_standoff_m=0.80`）。**`n_frozen_frames` 永远 0。** 四态：`drive`/`wedge`/`arc`/`turn`。**状态机对地图细节极敏感 ⇒ 单跑会得相反结论。**

## 地图层封板
已封板：D1 / D3 / D1+D3 / `island_trigger_m2` / `barren_blacklist` / sweep·frontier 参数 / 价值函数。机制 A（门口 3 格第 2 票同帧升格）；机制 B（开运算在 `STRONG_VOTES=20` 阶跃）。⚠️ replay 无墙钟预算 ⇒ **验收必须闭环 + 报 `ms_*` 的 p90/max**。⚠️ **不要调参。**

## 已判死 / 登记不落地
- **不再重试**：`local_u` / `region_lambda` / `rfirst_gate` / `fb_relax` / `OBSTACLE_HOLD=total` / `debt_gain_m2` / 黑名单 / 进度期限。`rfirst` 重开**须有可见性论证**。
- B3 停投。不值做（非判死）：`V/(1+d) → V·exp(-λd)`。
- **登记不落地**：G7 地图形态学清理器 / `fresh_split_m` / 点云层判别器（在线 18 维留一 AUC **0.638 < 0.65** ⇒ 观测层信息不足，**别再堆特征**）。
- **K=2 已封板**（分的是回波密度不是真实性：墙丢 4.7% / 家具丢 58.6%）。
- 已装开关（默认关，grep 找）：`GROUND_COVER_INFORMATIVE` / `OBSTACLE_GROUND_RATIO` / `RENDER_*` / `disc_lambda` / `fresh_split_m` / `patch_probe` / `sweep_value_mode` / `island_probe` / `strand_dump` / `--cf-hold-cells`。`_fold` 统一三处折叠。**出厂运行时只有 numpy+cv2+onnxruntime ⇒ 诊断不许 `import scipy`。**

## 指标规则
- 权威口径＝`out/_room_cover2.py`（分母＝单调的 `ground_ever ∩ 框`）。**报百分比必须报分母。**
- **项目级硬规则**：① count 型指标必须同时报 normalized rate；② "切换/放弃/重规划减少"类必须同时查完成量/覆盖 ⇒ **`switches` 永不单独出现**。
- **两个 n=1 的比较是抽签**；查参考集是否被挑过（`subr*` 有选择偏置）。**n=3v3 置换 p 下限 0.10 ⇒ 只能说"没分离"。**
- 算"行为不变"的 md5 **必须剔除 `traversable_m2` 类统计列**。
- 两源量纲：replay `yaw` **弧度** / online `yaw_deg` **度** ⇒ 跨源先 `repr()` 打原始值。
- 跨仪器对帧只用 `stamp`。报"掩码变化量"必须同时报格坐标。影子重放测决策不测覆盖。

## 工程硬规则
- 会话开头补 PATH：`export PATH="/c/Users/ZhuanZ/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:$PATH"`。
- 含中文路径：存在性判断用 Python `os.path.exists`，不用 Git Bash 的 `ls/cd/find`。
- Gate −1：默认值与不传**必须 md5 逐位相同**，且与历史 md5 相同；**哈希"决策列"不是整个文件**。r3 基线决策 md5 = `94df03cd8a087fc9b15d2ff4086a5d7f`。Gate 0/1/2/3 门槛、闭环口径、反事实守恒 —— 见技能。
- **自检通过 ≠ 正确**；判据落在**具体量**上。⚠️ 坏在输入的开关：span/帧率/summary 照样全绿。
- `pgrep -c -f` 算进调用者自己；`pkill` 跳过自己但不跳过父进程 ⇒ `docker exec bash -lc '...pkill...'` **会杀掉执行它的 shell**。清理写成脚本文件 `out/_cleansim.py`。⚠️ `pkill -x gz` 是错的（`gz` 是真实文件）。
- **有重试的驱动必须"每次尝试一份日志"**（`run_${TAG}_a${attempt}.log`）。
- 其余（docker / 后台 / Edit 纪律 / 守恒量 / 事件 vs 帧 / 参数扫描 / commit）见技能同名小节。
