# SLAM 大师赛 — 项目长期记忆

> 只放**跨会话必须记住的决策与坑**；数字与推导在文档里，按索引去找。目标 ~6 KB。

## 索引
- **★★ `docs/交接_20261004_复赛感知精修.md` = 当前进度交接，新会话第一份读它**（历史版本 `docs/_bak/`）。
- `docs/复赛实施方案.md`（主文档 §10–§18）；`docs/复赛自评与决赛差距.md`（状态/差距）；
  `docs/HANDOFF_项目交接.md`、`docs/评审意见归因与修正方案.md`、`docs/灾备与恢复清单.md`、`docs/复盘_踩坑手册_小白向.md`（**改前备份到 `docs/_bak/`**）。
- 技能：`gazebo-classic-headless-container` / `remote-job-and-snapshot` / `camera-lidar-person-fusion` / `headless-rviz-recording` / `lidar-map-forensics` / `competition-dataset-triage` / `competition-submission-final-audit`。

## 家底 / 赛事硬约束
- 唯一"丢了不可再得"的只有 `D:\E-Downd\data.bag`（初赛 2,532 MB）。
- 初赛 **81 分**，根因＝ RTK 降级段水平位置回退到里程计。邮箱 `open@openloong.org.cn` 抄送 `luqinghua@openloong.net`。
- **★ 官方答疑＝交付形态硬约束**：实时运行 + **整个数据集的视频** + 自建图 pcd；官方 pcd 只作提示；仿真 LiDAR yaw 按"**不修**"；输入只允许 **点云/图像/IMU**（`/state_estimation`·`/registered_scan` 只能事后评估）；**要输出建图 pcd + TF**；**建图 50% / 感知 50%**。

## 环境
- **仿真 = 本机 Docker 容器 `r2sim`**（`devrt/simulator-robot`：ROS Noetic + Gazebo 11.15.1）；挂载 `D:/E-Downd/sim_official -> /work`；容器内代码 `/work/r2slam_ws`（**改完本机必须 `docker cp` 过去**）。`docker start r2sim`，`sim_up.sh` 起仿真（`Xvfb :99`）。**Docker Desktop 必须开着**。
- **真实侧在本机 `.venv` 跑**（1427 帧 ≈5.7 min）。云端 T4 已废弃（校园网到不了）。`wsl.exe`/`wmic.exe` 被安全策略拦截。

## ★★ 官方仿真器两条最贵的坑
1. **LiDAR 不随车体旋转**（`vehicleSimulator.cpp` 把 LiDAR yaw 写死 0）⇒ **航向不可观测**，`heading.py` 用上一帧指令角速度×实测 dt 死推（p50 1.85°）；**转圈不产生新信息**。
2. **相机—LiDAR 外参只在 yaw≈0 成立** ⇒ 正解 `p_cam = Rz(-yaw) @ Tcl @ p_lidar`，按 `LidarSpec.rotates_with_base` 配置化（仿真 false，真实 true）。
- ⚠️ **"反投影回原框"自检抓不到这两条**（往返自消，实测 98.9% 通过）。有效判据是**与 `/gazebo/model_states` 比对**。
- **★ 仿真硬精度上界**：无 IMU、无里程计话题（只有 `/state_estimation`=真值），航向只能死推，`err_yaw` p50 **0.46°–12.05°** ⇒ **单跑之间不能比分数**，要比就**同一份采集重放**。

## 感知（定论）
- 检测器 = **YOLO11n ONNX**（`vendor/detector/yolo11n.onnx`，640，COCO80；运行时只用 onnxruntime/cv2，**无 torch**）。旧 tiny-yolov3 退役（把 youbot/椅子认成人）。
- YOLO11 输出裸 `[1,84,8400]`（cxcywh+letterbox，**无自带 NMS**）⇒ 解码自写 ⇒ **必须 `tools/check_onnx_parity.py` 对 ultralytics 逐帧比对**（偏差 ≤1 px、0 漏、0 类错）。
- `labels.py` = 类别表唯一定义源（`min_points`/`min_box_frac`/`min_height`）。
- `fusion.py`：深度用**相机中心距离**（原 `p[:,0]` 是 LiDAR 前向分量，斜视时全错）→ 按**深度间隙切近组**→ **体素连通域取大簇**；返回 `_idx` 供 `obj_mask`。
- **A/B（同 332 帧只换感知）**：person 实例 25→5（3 真人全中，误差 0.20–0.25 m）；残差 median 0.43→0.33 m。

## 交付文件本身"不报错的错"（第三轮，全部已修）
① pcd **从没跨帧去重** ⇒ 同体素每帧各追加，实测重复 **28–59 倍**；② 标签"先到先得" ⇒ 单帧误检**永久污染整面墙** ⇒ 改**按体素投票 ≥2 票**（⚠️ **一帧一体素只投一票**，按点计则单帧自己能凑够）；③ 加 **`max_height`**（墙本来就高）；④ **跨标签实例合并** `CROSS_LABEL_MERGE_M=0.8`；⑤ `cell_of` 必须 `np.floor`（`astype(int)` 朝零截断⇒出界点贴到边界=墙）；体素键空间**必须独立于 extent**。
- **渲染层两坑**：折线**掩掉窗口外点再拼接**⇒连出没走过的直线（按连续段分段画）；面板固定 20 m 窗⇒远处目标画不出（改自动取景）；只画"已确认实例"会让"检出未定位"看着像"没检出"⇒加**当帧细环形标记**。

## 根因史（v4–v6，压缩）
**v4**：① "标到墙上"**不是深度错**（LiDAR 深度对真人准到 0.97–1.01× 真距），错的是 **2D 框比实物高约 1.6 倍**，框吞墙点 ⇒ 加跨传感器门限 `LABEL_BOX_HEIGHT`；② 房间进不去**不是探索问题**（绿 230.7 m²，机器人只够得着 115.1 m²），**别再改探索策略**；③ 真实侧"检出人没标记"不是绘制问题 ⇒ 是**框没融合成物体**；真实数据集是**健身房**，"被认成人的机器人"其实是**白色训练器械**。
**v5**：① 真实侧把点云**按里程计航向反向旋转**再投影（`yaw` 补偿只该给仿真用）⇒ `yaw_proj = 0 if rotates_with_base else yaw`，`localised` 46.9%→95.3%（**藏一轮因 `frames.csv` yaw 列被门控成空列** ⇒ 教训：**空列让 bug 隐身**）；② 融合把**平地板**当目标 ⇒ 融合前用 `traversable.is_floor` 剔地面点；③ `min_height=1.00` 加错了（虚多是墙本来就高）⇒ 改 **0.40**；④ "标在墙上"靠**相对**判据 `LABEL_MAX_BOX_RATIO` 压住（真人 p50 1.65 vs 墙上虚多 4.57）⚠️ 一次跑的分隔度不能当跨跑阈值；残留：同墙被标 5 个 `chair`（每类要各自标定）；⑤ **规划层绝不能用"当帧回波"**（规划图逐帧闪变⇒原地打转）；⑥ 障碍层"一帧定死"+膨胀吃掉一半绿 ⇒ `OBSTACLE_MIN_VOTES=2`，但**安全单独走 `blocked()=obstacle|obstacle_now`**；`seen` 不再被天花板污染（`seen_mask`，`h<=OBSTACLE_HI`）。
**v6**：① **`box_h_px` 在 ~32 px 触底**（检测器 stride 下限）；`box_height_m = depth*box_h/f` 在远处失效（2.42m→74px→1.74m ✓；9.93m→32px→3.09m ✗）⇒ 旧门限在 5–9 m 外拒掉真人（258 真人框杀 118）；② 修法：**两条判据用 `or`**（各为充分条件）：`pass_width = width_lat ≤ 0.55`（任何距离成立）`or pass_box = (box_h_px ≥ 48) and (0<框推高 ≤2.40)`；`and` 会让真实侧 79% 真人框误杀。⚠️ `width_lat` **必须是相机系横向跨度**（`xc[:,0]` 极差）。效果：仿真真人被标 **35%→90%**，真实侧 **100% 保留**；③ 灰/绿/黑定义见下；④ 门口是**实体回波**（0→2.53 m 铺满），`diagnose_bottleneck.py` 量出客厅到各房间**瓶颈净宽 0.00 m**（拓扑连续，非膨胀参数）——机器人贴墙开过但墙在相机 **±57° 视场外** ⇒ 该改的是**去看**；⑤ 真实侧灰斑机制=**针孔 283/300 是障碍格**（28% 障碍块仅 1 格）；⑥ 三个"看起来在跑"的坑：同步脚本自检索引被清空表⇒`KeyError`；`| tail` 吞退出码 + `pipefail`⇒脚本打 SIM_READY 后中止（症状只有输出目录没变）；`sim_up.sh >/dev/null` 藏了 `gzserver` 僵死。**绝不静音一个允许失败的步骤；链条里别用管道接脚本。**

**v7（2026-10-04 第七轮，见 `docs/交接_20261004_复赛感知精修.md`）**：
① **"人标到墙上"真根因＝检测器把墙面帧当成人**（交付那一跑 13 个人实例、真值只有 3 个；多出来的在 y≈−7.8 一排，真值里什么都没有）。逐框实测：**人框 1227 个只有 214 个落在真人身上**；**置信度与横向宽度都不分离**（扫 conf 0.35→0.75 精度恒 44–46%；`sel_width_lat` 虚多反而更窄），**只有框推高分离**（真人 p10 1.74 / 虚多 p10 0.87）⇒ **v6 的 `or` 门限把 `LABEL_BOX_HEIGHT` 架空了**。
修法：新增 **`LABEL_BOX_HEIGHT_MUST={PERSON:(1.40,3.50)}` 作为必要条件 AND**（充分条件表保留，新计数 `reject_box_height_must`）。同采集 A/B：仿真 person 实例 **16→7**、确认虚多 **5→1**、真人观测 141/62/88→132/52/82；真实侧 localised 793→757、**绿 382.2→382.1 不变**，掉的 36 框经帧距核对是**一个激光实测 0.61 m 高的假人**（被拒 max 1.29 / 接受 min 1.42，间隔干净）。
② **换检测器实测无用**：yolo11s 落空处人框 880（11n 818）、精度一样，延迟 109→**269 ms**（预算 100）；11m 767 ms。**病是域差不是容量**；真正的路＝仿真自动标注微调 / 提高 320×180 渲染分辨率。
③④ ★**仪器是本轮最值钱的发现**：`diagnose_ground.py` **没去斜**（采集存的是**未去斜**原始点云）⇒ 补上后绿 429.6→422.2；补完仍差 ⇒ 与交付栅格逐格比 **seen IoU 0.981 / obstacle 0.946 / ground 0.858**（差 51 m²）⇒ **重放用 (x,y,yaw) 平面位姿、那一跑用完整 SE(3)，差 ≈0.07 m 与容差 ≈0.20 m 同量级 ⇒ 重放不可能复刻 `ground`**。⇒ **"针孔"是不稳定统计量（格子差 10%、读数差 4 倍）**，上一版"300/1388"作废；新增拓扑口径 **`pockets`**（被绿完全包住的灰连通块）。`dump_grids` 补写 **`ground_ever`/`obstacle_votes`/`obstacle_now`** ⇒ **交付产物自己就是仪器**。
⑤ ⚠️ `real_grey_before_after.png`／`_grey_ab.py` **是错的**（右图是被否决的实验当作 AFTER、用了已废弃的 2×2 偶数核、两边都是重放）；已用 `out/_grey_ab3.py` 重画成 `real_grey_abc.png`（旧规则重放/新规则重放/交付产物三联）。那圈"外围灰"＝**只看得到天花板（>2 m）的格子**，旧规则任何回波都算看过；新规则 92.4→5.7 m²。
⑥ **本轮改动已过 81 项测试 + 两源同采集 A/B，但尚未进提交包**（包仍是 v6 的 `ec8260d2…`）。下一步＝重出包 + 探索策略"薄障碍线＝待确认目标"。

## 语义地图三色定义（boss 问过）
```
绿 ground     = ground_ever & ~obstacle   这格确认是地板（有回波落在 |h|<容差）
灰 seen&~ground = 看过这里但没找到地板
黑 其它        = 从没有回波落在这里
```
**关键：格子只有"有回波落进去"才算 seen，射线穿过不算** ⇒ 真开着的门应画黑。

## 真实数据集
- 新 `openloong_r2.zip`(6.34 G) → `D:\E-Downd\openloong_r2_ex\`（`config/real_r2.yaml`）。
- 相机 FishPoly 1600×1296, A11=730.56；`LiDAR2IMU_0`≈单位旋转 ⇒ **IMU 与 LiDAR 共姿态**；图像 topic `/camera/rgb`。
- 点云 49,152 点/帧带 `offset_time` 0–0.094 s ⇒ **必须去斜**；IMU 成串到达、基本不可用。标定用**等距 `r=f·θ`（f=730.6）**已验证。

## 算法栈不变量
- `config.py` + `sources/` 是全项目**唯一**允许知道"仿真/真实"的地方；下游只认 `Frame`。**下游出现 `if sim` = 设计失败**。
- `tools/online_pipeline.py` = 产出交付物主程序；`tools/package_submission.py` = 终审（任一 FAIL 非零退出）。**终审 FAIL 绝不许写 zip**。
- 交付物 `r2slam/out/{online_sim,online_real}/`：run.mp4 + map.pcd + label_table.yaml + summary.json + frames.csv + tf.csv + tf/。pcd 字段 **`x y z label rgb`（17 B/点）**。
- **★ 当前包** = `提交包/…【固态蜗牛队】…zip` = **36,204,882 B / md5 `ec8260d28de6590033be36daca247b6c`**（终审 0 FAIL / 0 WARN）。上一版 `129d3764…`(37,357,792 B) 在 `_backup/20261004_2100_v5_129d3764.zip`。**MAP 里没有 torch**。
- **bag 记录时刻 ≠ 传感器时间** ⇒ 一律 `header.stamp`；`offset_time` 从帧戳倒推 ⇒ `time_offsets_from: end`。
- **`--npz` 复算没有驱动器** ⇒ 上一帧指令恒 0 ⇒ 航向恒 0 ⇒ **每个物体被绕机器人转过错误角度而地图看着正常** ⇒ `dump_frame(extra={"control_yaw": yaw})` 必须存。
- **改完管线必须两源各跑 5 帧冒烟**；改函数签名后**必须真跑一次**。
- **传输/比对：尺寸不算证据**（曾三次 md5 不同而 size 恒定）⇒ 比 md5。
- `voxel_size=0.5`；`pcd_voxel 0.15` 与配准 voxel 解耦。
- ⚠️ `tools/diagnose_ground.py` 读数**不可信（差 4 倍）**，嫌疑：不应用 object mask（管线 `obstacle_mask = ... & ~obj_mask`）。**修好前不下真实侧灰斑结论**。

## 探索（5 条硬规则；详见方案 §17/§18）
1. 候选只按距离排⇒**极限环** ⇒ 前沿**簇连通分量** + `score=簇大小/(1+距离)` + `min_target_range`。
2. **可达性是"连通分量"不是直线射线**（症状=看得见门口、进不去）。
3. **地面掩码是"一层点"** ⇒ 先闭运算，但**不能穿已知障碍** ⇒ `close(free) & ~dilate(obstacle)`。
4. **航向差不是停车理由——该画弧**（三档油门：原地转 / ≤25° 直行 / 走弧）。
5. 大视场传感器上 frontier 不够 ⇒ 加 `sweep`；目标指向**块的深处**。
- **★ 每个"什么都没发生"的判据都要说清盯的是哪个量**（帧数/直线/看到的地面 都曾盯错）。
- **★ 探索停条件改对过两次**：第一错盯"看到的地面"→改"**没走过的地面**"；第二错**赶路时误判停滞** ⇒ 加 `_closing_on_target()`，效果 帧 452→1452、路程 37.75→126 m。
- ⚠️ 探索常被 `--duration` 截断（`done_reason: null`），**房子北半边从未靠近** ⇒ 9/19 漏检主要是**没走到**。
- ⚠️ **仿真单跑方差极大**（路程 112/47/94/96/44/59 m，一次 300 帧没走动）⇒ **最该修的东西之一**。

## 硬规则
- **每个会话开头先补 PATH**：`export PATH="/c/Users/ZhuanZ/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:$PATH"`。
- **含中文路径**：存在性判断一律用 Python `os.path.exists`，不要用 Git Bash 的 ls/cd/find。
- 同一文件多处修改**串行**发 Edit；`Write` 覆盖前先 `Read`。
- **★ 最普适：自检通过 ≠ 正确**。判据要落在**具体量**上。
- 其余见技能：`remote-job-and-snapshot`、`gazebo-classic-headless-container`、`competition-submission-final-audit`、`lidar-map-forensics`（RTK 只取 `status==2`）。
