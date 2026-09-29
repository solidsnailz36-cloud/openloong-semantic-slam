# 大师赛 第一期赛题2 · 项目交接文档

> **给谁读**：接手本项目的 AI / 工程师。读完这一份 + `AUDIT.md`，应当能独立复现、修改、重跑整条链。
> **最后更新**：2026-09-13（初赛提交后锁定）
> **姊妹文档**：`docs/复盘_踩坑手册_小白向.md`（本项目踩过的每一个坑，写给未来的自己）
> **配套交付物文档**：`out/run4_v2/README.md`（对外说明，进提交包）、`out/run4_v2/AUDIT.md`（语义审计，进提交包）

---

## 0. 一句话状态

**初赛已完成并提交。** 现行提交包 `提交包/大师赛第一期赛题2—【固态蜗牛队】—【张煜晨】—【156****6887】.zip`
（132,544,618 B，md5 `044801db2993fffe4b532a48da1b4882`），内含 8 个成员，
产自 `out/run4_v2/`，其源包为 `out/run4_v2/submission.zip`（同一 md5）。

赛程已进入复赛窗口准备期（复赛 9/18–10/8），复赛是 **Gazebo 仿真**，
与初赛的**室外校园 + RTK** 是两套完全不同的题面 —— 见 §9。

---

## 1. 赛事与交付范围

### 1.1 赛题

**具身感知与导航 —— 非结构化场景下的在线语义地图构建与目标关联。**
三阶段：

| 阶段 | 内容 | 评审重点 |
|---|---|---|
| 初赛 | 处理给定 bag → 建语义地图 | 地图构建精度、语义标注准确率 |
| 复赛 | **组委会给定 Gazebo 仿真环境** 内探索 + 语义标注 | 自主感知算法的**泛化性**、算法在**不同环境下的部署与迁移能力** |
| 总决赛 | 人形机器人真机，语音指令 → 导航到目标物 | 指令理解正确率、导航时间/成功率；准确率 50% / 效率 30% / 鲁棒性 20% |

### 1.2 初赛提交要求（**两份官方材料互相冲突，按更严的执行**）

| 来源 | 录屏上限 |
|---|---|
| 赛题主页【作品提交规则】 | 视频 **5 min** 以内，可加速 |
| 随数据集下发的 `dataset-guide.md`「三、提交要求」 | **可加速至 2 分钟以内** |

⇒ **处置：压到 2 分钟以内，两份同时满足。** 现行成片 1′52.08″。

三项提交内容（两份材料一致）：
1. 整个采集场景的 PCD 点云 **与语义标注信息**
2. 车辆行驶的**全部轨迹**（格式参考官方 `path.yaml`）
3. SLAM 算法运行**全过程录屏**（含 RViz 等可视化界面）

> `dataset-guide.md` 另有一句关键说明：**「无需提交源码。」** —— 所以交付包里只有数据 + 文档，没有脚本。

### 1.3 提交方式

- 邮箱 `open@openloong.org.cn`，**抄送** `luqinghua@openloong.net`
- 压缩包与邮件主题**同名**：`大师赛第一期赛题2——队伍名称——队长姓名——队长手机号`
  - 本项目实际使用：`大师赛第一期赛题2—【固态蜗牛队】—【张煜晨】—【156****6887】`
- **>50 MB 必须走「超大附件」或上传云盘并在邮件正文给链接 + 密码**（本包 126.4 MB）
- 压缩包原则上 ≤500 MB

### 1.4 赛程（官网实时页，2026-09-12 抓取）

| 阶段 | 日期 |
|---|---|
| 报名组队 & 作品提交 | 7月29日 — **9月13日** |
| 初赛评审 | 9月14日 — 9月16日 |
| 公布晋级名单（前 30%） | 9月17日 |
| **复赛阶段** | **9月18日 — 10月8日**（3 周） |
| 复赛评审 | 10月9日 — 10月12日 |
| 公布决赛名单 | 10月13日 |
| 线下总决赛 | 暂定 1 月初 |

> ⚠️ 搜索引擎快照里存在旧版日期（提交截止 9/3、9/11 公布晋级），与官网差 8–10 天。**以官网为准**：
> <https://www.openloong.org.cn/cn/contests/master/j2daue>

---

## 2. 资产地图

### 2.1 本地（Windows 工作站）

```
E:\SLAM-大师赛\                          ← 工作区根
├── docs\                                ← 本次新增：交接 + 复盘
├── out\
│   ├── run4_v2\                         ★ 现行交付目录（提交包源头）
│   │   ├── semantic_map.pcd             225,210,718 B / 9,383,771 点
│   │   ├── labels.npy                   9,383,899 B（与 PCD 的 label 列逐点相同）
│   │   ├── path.yaml                    737,543 B / 6,522 点
│   │   ├── slam_run.mp4                 7,859,441 B / 1′52.08″ / 2.43×
│   │   ├── traj_enu.csv / traj_local.csv
│   │   ├── README.md                    42,900 B（进包）
│   │   ├── AUDIT.md                     27,710 B（进包）
│   │   ├── submission.zip               ★ 132,544,618 B / 044801db2993fffe4b532a48da1b4882
│   │   ├── _bak\                        ← 回退点（4 个历史包/视频）
│   │   └── shots\                       录屏抽帧（刷新过）
│   ├── run4\                            上一版（跑火车的中间态 + RB3_REPORT.txt）
│   ├── run2\、cmp\、viz_out\、viz_out_before\
│   └── _bak\ 等
├── 提交包\                               ★ 已按规范命名的最终压缩包
├── sn... （历史产物：semantic_map*.pcd / submission*.zip）
└── .workbuddy\
    ├── memory\                           ★ 记忆：MEMORY.md + 2026-09-11/12/13.md
    ├── tmp\                              ★ 349 个脚本（管线脚本 + 诊断脚本 + 探针）
    └── out\                              报告：model_options.md / osm_crossvalidation.md /
                                          semantic_audit.md / 各版本截图
```

### 2.2 远端云机（真正干重活的地方）

- 地址 `183.239.162.232:30032`，root，工作目录 `/root/gpufree-data/slam_comp`
- 规格：Tesla T4 / Ubuntu 20.04 / ROS Noetic / **容器 cgroup 内存上限 42 GiB**（`free` 显示的 376 GB 是宿主的，别信）
- 通道（本地调用方式）：
  ```bash
  cd E:/SLAM-大师赛/.workbuddy/tmp
  bash rr.sh "<远端命令>"                 # 执行命令并回显输出
  "E:/software/python.exe" put_lf.py a.sh b.py   # 以 LF 行尾上传到远端工作目录
  ```
  - `rr.sh` → `ssh_env.sh`（凭据）+ `remote_ssh.py`（paramiko，跑在 `E:/software/python.exe`）
  - **注意**：本机 py3.13 的 venv 里没有 paramiko，必须用 `E:/software/python.exe`

- 远端目录：
  ```
  slam_comp/
  ├── venv/                     Python 3.8.10（venv/bin/python3，注意没有 venv/bin/python）
  ├── data.bag                  2.4 GB，6522 帧 / 652 s
  ├── 根目录 ~90 个 .py/.sh     管线 + 诊断脚本（见 §4）
  ├── sfpnet/                   SFPNet 代码 + ckpt/model_kit.pth + sfpnet_sk.yaml
  ├── Cylinder3D/               旧语义后端（已弃用，保留对照）
  ├── run/                      ★ 初赛全量产物（scans / labels_per_scan / poses* / rtk* / loops.npz）
  ├── run2/                     Cylinder3D 时代的交付（对照用）
  ├── run3/sf/                  SFPNet 全量逐帧标签（6522 个 npy）
  ├── run4/                     ★ 现行交付的中转目录（scans 软链 + ENU 位姿 + 地图 + 录屏）
  ├── run4_v2 的源都在这里
  └── snapshots/20260912_final/ ★ 硬链接冻结快照（含 venv_frozen_20260912，兜底）
  ```

### 2.3 关键指纹（复现时用来确认"你手上的是哪一版"）

| 对象 | 指纹 |
|---|---|
| `data.bag` | 2.4 GB / 6522 帧 LiDAR / 6521 帧 RTK / bag `start_time = 399.889 s` |
| Cylinder3D 权重 | `cylinder3d_mmdet3d.pth` 229,619,485 B md5 `125b5af9578dc50016a3adf1ad6e5275` |
| SFPNet 权重 | `sfpnet/ckpt/model_kit.pth` 430,361,294 B md5 `f681220c85e63e7669c8dd22ad261b08` |
| 环境基线 | py3.8.10 / torch 2.4.1+cu121 / **spconv 2.3.6（源码编译，唯一没法 pip 复原）** / numpy 1.24.4 / mmdet3d 1.4.0 / mmcv 2.2.0 / T4 driver 535.183.06 |

---

## 3. 系统架构（数据流）

```
data.bag (2.4 GB)
   │
   ├─ /livox/lidar   6522 × CustomMsg ─┐
   ├─ /livox/imu    130435 × Imu       │  decode_bag.py
   ├─ /qx/evk        6521 × NavSatFix  │  ──────────────► run/scans/*.pcd   (6522 帧, x y z intensity)
   ├─ /qx/evk/heading                  │                  run/stamps.npy
   └─ /qx/evk/velocity                 ┘                  run/rtk.csv  (6518 有效)
                                                          run/imu.csv
   │
   ├─[A] 里程计  kissicp_map.py  ──────► run/poses.npy    (KISS-ICP 局部系, 无回环)
   │
   ├─[B] 回环    loop_lsq.py      ──────► run/poses_opt.npy + loops.npz
   │
   ├─[C] RTK 解码 decode_rtk_full.py ───► run/rtk_full.csv (9 元 position_covariance!)
   │
   ├─[D] 融合    rtk_fuse4.py      ─────► run/poses_fused3.npy  ★ 已是 ENU
   │        （假固定解高度门限 + 3D 相似 + 一步落 ENU）
   │
   ├─[E] 语义    sfpnet_infer.py (run3/sf/labels_per_scan/*.npy, 6522 个)
   │             semantic_label_v3.py
   │        （标签多数投票体素化 + 动态类整类剔除 + SOR + 地面五类归并）
   │                                     ──► run4/semantic_map.pcd + labels.npy
   │
   ├─[F] 轨迹    write_traj_local.py ───► traj_local.csv
   │             geo_trajectory.py   ───► traj_enu.csv + path.yaml
   │
   ├─[G] 录屏    make_final_cache.py（地图抽稀缓存）
   │             make_online_video.sh → slam_online.py
   │             （Xvfb + roscore + RViz + Tk 叠加 + ffmpeg x11grab）
   │             → slam_run_online_full.mp4（~4′32″ 母带）
   │             speed_calc.py + ffmpeg setpts/drawtext → slam_run.mp4
   │
   └─[H] 打包    fx2_pack.py ──────────► submission.zip（8 成员 + in-zip==source 断言）
                 final_accept.py ─────► 从 zip 内部独立复算（不读打包脚本自报告）
```

**关键设计决策（为什么长这样）：**

1. **不用 ROS 跑 SLAM。** KISS-ICP 走**纯 Python**，绕开 `livox_ros_driver2/CustomMsg` 的 bag 类型坑（容器没装该驱动，`rosbag play` 直接报缺消息定义）。ROS 只在**录屏**环节出现。
2. **RTK 是逐帧绝对约束，不是事后配准。** 这是本项目最大的一次架构纠错 —— 详见 §5 与复盘文档第 3 章。
3. **位姿一步落 ENU。** 全链只有一份位姿文件 `poses_fused3.npy`，它本身就是 ENU；地图 / 三条轨迹 / 录屏都读它。**下游不存在任何坐标系补丁。**
4. **数据只有 LiDAR + IMU + RTK，没有相机。** 这一刀砍掉所有 2D-基础模型 / 图-点配对路线，语义**只能走纯 3D**。

---

## 4. 标准链路（产出现行提交包的链路）

### 4.0 总览

| # | 阶段 | 脚本 | 输入 → 输出 | 是否需 GPU | 耗时量级 |
|---|---|---|---|---|---|
| 0 | 环境 | — | 远端 venv + spconv + SFPNet | — | 一次性 |
| 1 | 解码 | `decode_bag.py` | `data.bag` → `run/scans/*.pcd` + `rtk.csv` + `imu.csv` + `stamps.npy` | 否 | ~20 min |
| 2 | 里程计 | `kissicp_map.py` | scans → `poses.npy` + `map.pcd` | 否 | ~40 min |
| 3 | 回环 | `loop_lsq.py` | `poses.npy` → `poses_opt.npy` + `loops.npz` | 否 | ~10 min（ICP 6 min，`REUSE_LOOPS=1` 可跳过） |
| 4 | RTK 解码 | `decode_rtk_full.py` | bag → `rtk_full.csv`（含 `position_covariance` 9 元）+ `heading.csv` + `velocity.csv` | 否 | ~5 min |
| 5 | **融合 + 落 ENU** | `rtk_fuse4.py` | `poses_opt` + `rtk_full` → **`poses_fused3.npy`（ENU）** | 否 | ~5 min |
| 6 | 语义逐帧推理 | `sf_full.sh` → `sfpnet_infer.py` | `run/scans` → `run3/sf/labels_per_scan/*.npy` | **是** | **~11 min** |
| 7 | 语义建图 | `semantic_label_v3.py` | poses + labels_per_scan + scans → `semantic_map.pcd` + `labels.npy` | 否 | ~15 min |
| 8 | 轨迹三件套 | `write_traj_local.py` / `geo_trajectory.py` | poses → `traj_local.csv` / `traj_enu.csv` / `path.yaml` | 否 | 1 min |
| 9 | 录屏缓存 | `make_final_cache.py` | `semantic_map.pcd` → `final_map_sub.npz`（抽稀到 ~72 万点） | 否 | 1 min |
| 10 | 录屏 | `make_online_video.sh` → `slam_online.py` | poses + scans + 缓存 → `slam_run_online_full.mp4` | **是** | **~4.5 min**（544 ticks） |
| 11 | 编码压缩 | `speed_calc.py` + ffmpeg | 母带 → `slam_run.mp4`（1′52″） | 否 | 1 min |
| 12 | 打包 | `fx2_pack.py` | 8 成员 → `submission.zip` | 否 | 1 min |
| 13 | 终审 | `final_accept.py` | zip → PASS/FAIL | 否 | 1 min |

**编排脚本**（远端，串成一条受跟踪后台任务）：

- `rebuild2.sh` — 【编译/重算最重的一段】altitude-gated ENU 位姿 → ENU 地图 → ENU 轨迹 → 重录 → 打包
  （**不重跑 KISS-ICP 与 SFPNet 推理**，复用 `run/scans` 与 `run3/sf/labels_per_scan`，
  这就是"30 min 而不是 6 h"的原因）
- `rebuild3.sh` — 只做位姿之后的一切：**poses → traj → cache → smoke → record → encode → report**（6 阶段）
- `reramp.sh` — **只重录视频**（相机换机位）：cam → cache → smoke → record(SHOT_AT=250) → encode → report
- `run_night.sh` — 无人值守夜间链（首次全量：SFPNet 全量标签 → 建图 → 在线录屏 → 打包）
- `sf_full.sh` — 单跑 SFPNet 全量逐帧推理

### 4.1 各阶段详解（含"改这里会炸什么"）

#### 阶段 1 — 解码（`decode_bag.py`）

- 输出：`scans/<时间戳>.pcd`（**文件名补零**，避免字典序 ≠ 数值序）、`stamps.npy`、`rtk.csv`、`imu.csv`
- **两个必知细节**：
  1. Livox `CustomMsg` 头部在 `rsvd[3]` 之后**还有 4 字节 `data_type`**。少读这 4 字节 → `x` 恒为 0.0、`offset_time` 变成 2²⁰ 的倍数。正确读法：`off += 1+3; narr=u32; off += 4`。
  2. 每点布局 `[u32 offset_time][f32 x][f32 y][f32 z][u8 refl][u8 tag][u8 line]`；`header.stamp == timebase`，单位 **ns**，每帧跨度 **97–100 ms**。
- **`rtk.csv` 未保存 `position_covariance`** → 这是后来不得不写 `decode_rtk_full.py` 回炉的原因。
- **`/qx/evk/heading` 是 8 字节裸 float64，没有 std_msgs/Header** —— 用标准消息解析会全错。

#### 阶段 2 — 里程计（`kissicp_map.py`）

- KISS-ICP v1.3.0，**API 与旧版不同**：需 `from kiss_icp.pipeline import KissICP`；`register_frame` 参数变化；`poses` 属性被移除 → 必须手动累积 `kiss.last_pose`。
- 建图侧 `MAX_RANGE = 40`（**记住这个数**：语义回投口径曾因没有这道门控而与建图不一致）。
- 产物 `poses.npy` 是 **KISS 局部系**，**纯里程计、无回环** → 2544 m 累积约 24.7 m（0.97%）漂移。这不是参数问题。

#### 阶段 3 — 回环（`loop_lsq.py`）

- 关键帧子图 ICP 检测回环 → `loops.npz`（存盘，`REUSE_LOOPS=1` 跳过 6 min ICP）。
- **只解位置、不碰旋转**；未知量是"对里程计的修正 `dx`"的**增量形式**：
  - 里程计约束 `dx_{i+1} − dx_i = 0`（w=1，保持形状）
  - 回环约束 `dx_b − dx_a = E_ab`（w=100）
  - 极小 Tikhonov 去规范零空间；`dx_0 = 0` 后处理归一
  - **直接稀疏求解** `A.T@A` + `scipy.sparse.linalg.spsolve` —— 一次成功
- 结果：回环残差 **105.97 → 0.00 cm**；位置修正均值 118 cm / max 252 cm。
- **为什么只解位置**：回环误差 = 平移 106 cm + 旋转 2.5–3.3°，但**面内 yaw 误差极小**（均值 0.23°）→ 那 3° 是俯仰/横滚，在植被杂波主导的小子图上 ICP 不可信。
- ⚠️ **回环没有"治好"地图**。全量评估 ATE(2D) 只从 608.3 cm 降到 599.5 cm。真正解决问题的是阶段 5。

#### 阶段 4 — RTK 完整解码（`decode_rtk_full.py`）

产出 `rtk_full.csv`（**含 9 元 `position_covariance` + `cov_type`**）、`heading.csv`、`velocity.csv`。
**没有这个文件，阶段 5 的两道质量门限无法实现。**

#### 阶段 5 — ★ 核心：融合 + 一步落 ENU（`rtk_fuse4.py`）

这是全项目技术含量最高的一段，**改它之前请读完本节的四条**。

1. **3D 相似拟合**：每帧 RTK 位置（E/N/U）与 KISS 局部轨迹做 Umeyama（scale + R∈SO(3) + t），迭代剔除最差 25% 残差。
   实测 `scale = 0.98761`（里程计测距低估 1.24%）、解出倾角 **0.90°** → **KISS 局部系的 z 轴本来就近竖直**。
2. **在局部系内只修正位置**，旋转保留 KISS 原样（倾角真实且全帧共享 → 地图地面是一张一致的斜平面，搬正反而掀翻）。
3. **两道质量门限**：
   - **(a) 协方差门限** `σ_ee ≥ 0.5 m` → 剔除 **786 帧（12.0%）**
   - **(b) 高度一致性门限**（本项目独有）`/qx/evk` 有 **7 段 / 431 帧 / 6.61% 的"假固定解"**：`status = 2`、协方差健康，但海拔偏离 40 s 邻域中值最多 **5.624 m**。其中 **149 帧 `σ_ee ≤ 0.30 m`（门限 a 完全看不见）**，最极端 **2 帧 `σ_ee = 0.02 m` 却偏 +3.04 m**。
     判据：**40 s 中值窗 + 0.5 m 容差 + 5 帧最小段长**。
     **处置：只丢高度、不丢水平** —— 异常帧只插值**竖直改正量**，`lat/lon` 原样保留；插值沿**累计弧长**（非时间）。
4. **位姿一步到位落 ENU**：先解 local→ENU 相似变换，再把**每一帧位姿矩阵整体搬过去**，写出的 `poses_fused3.npy` **本身就是 ENU**。

| 参数 | 值（以进包 README 为准） |
|---|---|
| `s` | **1.01255**（重录链自述）/ 1.01243118（离线拟合）——同一变换的两次独立解 |
| yaw | **+23.936°** |
| `t` | **(−6.576, +6.639, +1.448) m** |
| 可用拟合帧 | **5587 / 6522** |
| 位姿 ↔ RTK ENU 残差（内点） | 中位 **0.00 mm** |
| 交叉验证 | `1/s = 0.98761` == 第 1 步独立解出的 `scale`，一致到 5 位有效数字 |

> 参数存在 `run/local2enu.npy`（s, R 四元素, t），`run4/local2enu.npy`。录屏脚本通过环境变量 `LOCAL2ENU=` 读它。

#### 阶段 6 — 语义逐帧推理（`sfpnet_infer.py`）

- `sfpnet/ckpt/model_kit.pth`（107.55 M 参数 / 410 MiB）+ `sfpnet_sk.yaml`（SemanticKITTI 配置，**自己推的**，仓库只发了 S.MID 的 demo config）。
- **intensity 必须 ÷255**（SK 训练用 0–1 量纲）。不除 → 56% 点被判成 traffic-sign。
- 推理 API：用 `load_aligned()` 加载 ckpt（**spconv 权重布局取决于 algo**，一个 checkpoint 里 KRSC/RSCK 两种布局共存，见复盘文档）；输出点数 == 输入点数，无损。
- 速度：T4 上 **93 ms/帧**（离线）/ **~193 ms/帧**（在线端含前后处理）。
- 全量 6522 帧 → `run3/sf/labels_per_scan/*.npy`（uint8 19 类）。

#### 阶段 7 — 语义建图（`semantic_label_v3.py`）

四步，顺序不能变：

1. **动态类整类剔除**：SK 0–7（car/bicycle/motorcycle/truck/other-vehicle/person/bicyclist/motorcyclist）。
   本数据里这一类**实测就是自车**（车顶 MID-360 扫到自身）：152/152 帧出现，水平距离中位 1.42 m、垂直偏移 +1.07 m，全程稳定。
2. **逐帧 SOR**（k=20，σ=2.0），点数 <20 的帧跳过。
3. **0.15 m 体素内直接对标签做多数投票** → 逐点 `label`；`rgb` 由 id 官方色表导出，严格自洽；另存同序 `labels.npy`。
   > **不要**回到"按标签上色 → 体素内颜色平均 → 最近色反推标签"这条链 —— 混色会反推出**第三个类**（实测 sidewalk 19.4% → 8.24%）。
4. **地面五类归并**（road/parking/sidewalk/other-ground/terrain → `road`）。理由见 §5"地面细分不可恢复"。

点数账：45.2 M 原始点 − 动态 6.35 M − SOR 1.06 M → 0.15 m 体素 → **9,383,771 点**。

#### 阶段 8 — 三条轨迹

- `write_traj_local.py` → `traj_local.csv`（`stamp,x,y,z,yaw`）
- `geo_trajectory.py` → `traj_enu.csv`（`stamp,east,north,up`）+ `path.yaml`
- ⚠️ **`traj_local.csv` 的名字是历史遗留，它的数值就是 ENU**（与 `traj_enu.csv` 最大差 **0.5 mm**，只是 stamp 单位不同：秒 vs 纳秒）。**旧的"它是 KISS 局部系"记录已作废。**
- `path.yaml` schema 必须对齐官方样例：`path:` 列表（`altitude/latitude/longitude/stamp`，缩进 4/4/6）+ 结尾 **`total_points: 6522`**。

#### 阶段 9–11 — 录屏（`make_online_video.sh` + `slam_online.py`）

**这是"真·在线"，不是回放。** `slam_online.py` 约 460 行，每帧现场跑：
① KISS-ICP 逐帧配准 → ② SFPNet 逐帧推理 → ③ 用融合位姿把该帧搬进世界系做增量建图 → ④ 逐帧类别直方图/耗时推到 Tk 屏幕叠加面板。

- 链路：Xvfb `:99`(1280×720) + roscore + RViz + Tk 叠加 + `ffmpeg x11grab`
- 关键环境变量：`RUN_DIR` / `MODEL` / `LIVE=1` / `STRIDE=12`（每 12 帧取 1）/ `VOXEL=0.6` / `HOLD=10` / `SHOT_AT=250` / `OUT=` / `LOCAL2ENU=` / `REC` / `OVERLAY` / `MAX_TICKS`
- **烟雾测试**：`MAX_TICKS=4 STRIDE=48 REC=0` → 看日志里有没有 `RUN_COMPLETE`。**永远先 smoke 再 record。**
- 编码：`speed_calc.py` 算倍率 → `setpts=PTS/$F` + `drawtext`（右下角烧入 `PLAYED AT Nx`）+ libx264。
- 产物：母带 `slam_run_online_full.mp4`（~4′32″，**含启动与收尾全图发布**）→ 成片 `slam_run.mp4`。

#### 阶段 12 — 打包（`fx2_pack.py`）

8 个成员，固定顺序：
`semantic_map.pcd` / `labels.npy` / `path.yaml` / `slam_run.mp4` / `traj_local.csv` / `traj_enu.csv` / `README.md` / `AUDIT.md`
逐成员断言 `in-zip == source`，并 `assert z2.namelist() == ORDER`。

#### 阶段 13 — 终审（`final_accept.py`）

**解 zip → 逐成员对拍 → 解出视频 ffprobe + 抽末帧重量构图 → 解出 path.yaml 用 YAML 解析器 + 与 traj_enu 对拍 → 解出 PCD/npy 逐点比对 + 类别表核对 → README 文案新值/旧值扫描。**
**原则：一律不读打包脚本的自报告。**

---

## 5. 关键不变量（改任何东西之前先读这一节）

### 5.1 坐标系

**交付物全部五份几何成果（PCD / traj_enu / traj_local / path.yaml / 录屏）统一在 ENU**，
原点 = 首个 RTK 固定解帧：`lat 31.121152840 / lon 121.603270630 / alt 6.2988`。
`x` 东 / `y` 北 / `z` 天，单位米。

- 局部→ENU 变换：`s = 1.01255`、yaw `+23.936°`、`t = (−6.576, +6.639, +1.448) m`
- **改位姿必须整体重出地图 + 三条轨迹 + 录屏**（依赖顺序不可颠倒）。

### 5.2 类别编号

采用 **mmdet3d / SemanticKITTI 19 类 0 基约定**（= 官方 20 类表去掉 `unlabeled` 后左移一位）：

| label | 类别 | 占比 | rgb |
|---|---|---|---|
| 8 | `road` | 44.35% | (128, 64, 128) |
| 12 | `building` | 11.59% | (70, 70, 70) |
| 13 | `fence` | 2.25% | (190, 153, 153) |
| 14 | `vegetation` | 41.05% | (107, 142, 35) |
| 15 | `trunk` | 0.24% | (152, 251, 152) |
| 17 | `pole` | 0.50% | (153, 153, 153) |
| 18 | `traffic-sign` | 0.02% | (220, 220, 0) |

> ⚠️ 官方表是 `road = 9 / building = 13 / pole = 18`，**差 1**。别按官方表读。
> 组委会 FAQ 明确「不提供类别列表」→ **交付物必须自带 id↔名称↔色表**（README §2.1(a)）。

### 5.3 `stamp` 口径

用 **bag 原生秒**（400.206 → 1052.304，跨度 652.098 s），**不是 Unix 绝对秒**。
已从原始 bag 实测坐实：`bag start_time = 399.889 s`，`/qx/evk` 的 `header.stamp` 也是相对秒（`sec=400, nsec=234002144`），**包里根本不存在绝对时间**。
官方样例 `path.yaml` 的 1.77e9 是**另一个采集会话**（经纬度差 12 km），只作格式样板。
`geo_trajectory.py` 留有 `STAMP_EPOCH_OFFSET` 常量可平移。

### 5.4 录屏时长

**压到 2 分钟以内**（两份官方材料冲突，按更严的执行）。现行 2.43× → 1′52.08″。
原始母带 4′32.16″ 在远端，任何倍率可在 10 min 内重出。

### 5.5 真值识别

**本数据集的真值就是交付包里的 `path.yaml`**（737,543 B，stamp 400.206–1052.30，与 6522 帧对齐）。
`D:\E-Downd\path.yaml`（540,151 B，stamp 1773xxx）**是另一个数据集的**。
→ **任何 path.yaml 在用之前，先核对 stamp 范围与点数。**

### 5.6 已证伪、别重复劳动

| 结论 | 依据 |
|---|---|
| **地面三级细分不可恢复** | 输出的是"到本帧传感器的距离"而非材质：同一 0.5 m 格内 sidewalk 点均距比 terrain 近 **2.8–3.8 m**，**95.3%** 的格子同向。分界半径 6 m(cyl) / 7.5 m(sfp) = 目视看到的"圆更大"。**归并是唯一诚实做法。** |
| **car = 自车** | 中位水平 1.42 m / 垂直 +1.07 m，全程稳定，应剔除 |
| **交付点云 road 类"厚"不是缺陷** | 纯 road 1 m 格内 z 跨度中位只有 **0.068 m**；归并后 road 里 **52.5% 是原 terrain** |
| **回环不是解药** | ATE(2D) 608.3 → 599.5 cm。真正解决的是 RTK 融合 |
| **去斜（运动补偿）不值得做** | 剪切真实存在（一帧 40 cm），但这是**街区走廊型数据集**，绝大多数面平行于航向，而平行面在沿轨剪切下**数学不变**（实测 0.2→0.2 cm）。只有正对来向的面 4.5 → 0.9 cm |
| **`/qx/evk/heading` 无独立价值** | 20 个重访对 ±6° 穷举最优 dYaw：`corr(最优dYaw, 航向dYaw) = +0.034`，只降 1% |
| **`fence` 与 `building` 不要合并** | 两者是真实存在的两类几何（p50 高度 3.03 m vs 1.52 m），只是边界模糊。合并 = 自创标签体系，净亏 |
| **两条判据已被自己否掉** | ① 里程计 z 对比 RTK：噪声底中位 1467 mm，被漂移淹没 ② 6×6 分块平面拟合：无分辨力 |

---

## 6. 回退链与快照

```
现行  out/run4_v2/submission.zip                     132,544,618  md5 044801db2993fffe4b532a48da1b4882
  │   （= 提交包/大师赛第一期赛题2—【固态蜗牛队】—【张煜晨】—【156****6887】.zip）
  │
  ├─ out/run4_v2/_bak/submission_before_reframe.zip   146,861,313  1fccf443b345485b34a00eb5d34a3318
  │      （取景重录前；假固定解已修、ENU 已统一）
  ├─ out/run4_v2/_bak/slam_run_before_reframe.mp4      22,193,678  a5ae868e…（"只拍一角"版）
  ├─ out/run4_v2/_bak/slam_run_170s_2m50.mp4           10,321,666  c6478c03f5bbc0c5753fce6d31be2068（1.60× 版）
  ├─ out/run4_v2/_bak/path_before_schemafix.yaml          633,164
  │
  ├─ out/run4/submission.zip                          148,607,591  d1775123c827e3766f240ca0a247f1f5
  │      （ENU 已统一、假固定解未修、视频未重录）
  ├─ out/run4/_bak/submission_before_framefix.zip     144,137,650  7464fc671f8f744affb0a1e6f2abe203
  │      （坐标系统一之前）
  ├─ out/run2/submission.zip                          226,165,065  1e4c16e7ddeddefb53a1b01835bb4bf9
  │      （Cylinder3D 时代）
  └─ run2/submission_prevideo2.zip                    216,500,509  1ed2fcfc3807e249a16256c655eef343

远端兜底：snapshots/20260912_final/  ← 硬链接冻结（含 venv_frozen_20260912，7.7 G 表观 / 实占 ~0）
          小包 snapshots/20260912_final_code.tar.gz 3,404,244 B  md5 f29a6730109222d5744a385b312948b2
          回退手册见远端 MANIFEST.md（4 个场景：环境坏了 / 代码改坏 / 标签误删 / 全崩先保交付）
```

**硬链接快照的边界**：挡得住 `rm` / `rmtree`（最常见的破坏方式），**挡不住就地写入**
（`open(path,'w')` 改同一 inode 时快照会跟着变）。
⇒ **装任何新环境一律另起 venv，绝不在 `venv/` 里 `pip install`。**

---

## 7. 现行提交包指纹（逐成员）

| 成员 | 大小 | 说明 |
|---|---|---|
| `semantic_map.pcd` | 225,210,718 | `FIELDS x y z intensity label rgb` / `POINTS 9383771` |
| `labels.npy` | 9,383,899 | uint8，取值 {8,12,13,14,15,17,18}，与 PCD `label` 列逐点相同 |
| `path.yaml` | 737,543 | 6,522 点，含 `total_points: 6522` |
| `slam_run.mp4` | 7,859,441 | h264 1280×720 25 fps **1′52.08″**，md5 `c0f2d2760dc7…` |
| `traj_local.csv` | 586,558 | `stamp,x,y,z,yaw`（ENU 值） |
| `traj_enu.csv` | 227,679 | `stamp,east,north,up` |
| `README.md` | 42,900 | 对外说明；文档内**无 `<待填>` 残留** |
| `AUDIT.md` | 27,710 | 语义审计（9 节，含"没有解决的问题"） |
| **`submission.zip`** | **132,544,618 = 126.4 MB** | md5 `044801db2993fffe4b532a48da1b4882` |

---

## 8. 已知限制与未解决项（**交付文档里已如实披露，不要偷偷改掉**）

1. **语义后端未做边缘适配**：SFPNet 107.55 M 参数 / 410 MiB，未量化剪枝；
   `193.7 ms/帧` 是 **T4** 数字，不是边缘平台数字。FAQ 要求"模型大小需适合边缘计算设备"——
   **§4 标记为已知限制，未声称已适配。复赛要还这笔账。**
2. **绝对语义准确率给不出**：无真值，任何 mIoU 都是编的。能证明的只到
   "表面 vs 垂直结构"这一层（F1 66.2% vs 61.4%）与"是不是地面类"（89.1%）。
3. **`/map`（终态全图）在 RViz 里渲染不出来**，`/map_live` 通道正常 —— **原因未查明**，已放弃调试。
   影响仅限末帧，已在 README §4 写明。
4. **A 段 RTK 无解区（帧 3521–4071，224.8 m）的形状**：用 RTK 两端切向反解出
   "204 m 直线 + 末段 93.3° 转角"，R=6 m 时总长与实测吻合到 0.1 m。
   残余不确定 = 约 21 m 转角区的曲率细节，量级亚米级。**外证：OSM 路网交叉验证，
   263 m 直行段航向差 0.05°、转角差 0.86°，87.3% 帧落在道路 5 m 内。**
5. **未解疑点**：`/qx/evk/velocity` 分量中位数 0.019/0.336/−0.091 但范数 p50 = 15.18，量级自相矛盾。
   **该字段未用于任何环节**，lat/lon/alt/cov 已交叉验证可信 → 不影响地图。
6. **KISS 旋转未动**：位置用 RTK 逐帧校正（垂直 RMSE 1 mm），**旋转保持 KISS 原样**
   （作者判断倾斜真实且全帧共享）。位置与姿态不同源 —— **这是后续追小尺度锐度时的第一个下手点。**

---

## 9. 复赛工作计划（9/18 – 10/8，3 周）

### 9.1 题面变化（**必须先认清这个断裂**）

| | 初赛 | 复赛 |
|---|---|---|
| 环境 | 上海张江，室外校园，坡地，2540 m | **Gazebo 仿真，室内 `house.world`** |
| 传感器 | Livox Mid-360 + IMU + **双天线 RTK** | 车辆仿真器（`vehicle_simulator`），**大概率没有 RTK** |
| 场景内容 | 道路 / 植被 / 建筑 | **大物件（沙发、凳子）+ 小物件（可口可乐、花瓶）+ 行人**，物件位置**有高有低** |
| 语义目标 | 地面材质等"东西类"（stuff） | **可数的**物体实例（thing）→ 需要"目标关联" |
| 评审 | 地图精度、语义准确率 | **泛化性 + 跨环境部署与迁移能力** |

**⇒ 三个直接推论：**

1. **RTK 融合这条护城河在复赛失效**（仿真里没有 RTK 绝对定位）。复赛的位姿可靠性回到
   **里程计 + 回环**，而这正是初赛暴露出的弱项（2544 m 累积 0.97%）。
   → 复赛第一优先级是把 `loop_lsq.py` 那条链做扎实，或换成更强的 LiDAR 惯性里程计。
   **反过来说：Gazebo 会免费给出完美 GT 位姿 → 可以直接算真 ATE，不用再靠"物理不可能测试"这类间接判据。**
2. **语义从"stuff"转向"thing"** → 19 类 SemanticKITTI 体系**不再适配**。
   家具/瓶罐/行人不在 SK 的 19 类里。开放词汇（open-vocabulary）从"备选"变成"刚需"。
3. **"部署与迁移能力"是明写进评审的** → 模型体积/量化/推理时延这一笔账必须还（§8 第 1 条）。

### 9.2 环境搭建（第一步，有坑）

组委会文档给的步骤：

```bash
# 1. world
unzip gazebo_models.zip && cd gazebo_models && mv ./* ~/.gazebo/models/
gazebo ~/.gazebo/models/house.world
# 2. 车辆模型
unzip src.zip            # 放到 catkin_ws/src
cd catkin_ws && catkin_make
# 3. 运行
roslaunch vehicle_simulator house_vehicle.launch
```

- **坑预判**：本机 Windows **无 ROS**；WSL 被沙箱黑名单禁。⇒ **在远端 T4 云机（Ubuntu 20.04 + ROS Noetic）上做**，
  Gazebo 需要渲染 → 复用我们已跑通的 **Xvfb + `LIBGL_ALWAYS_SOFTWARE=1`** 配方（见技能 `headless-rviz-recording`）。
- Gazebo 是**物理仿真**，CPU 负载高（我们只有 42 GiB cgroup 上限，但 CPU 核多）；渲染无 GPU 时可软件光栅。

### 9.3 复用度评估（现有链哪些能直接搬）

| 环节 | 复赛可复用度 | 说明 |
|---|---|---|
| `decode_bag.py` | ★★★★ | 只需改话题名/消息类型（Gazebo 通常是 `sensor_msgs/PointCloud2`，比 Livox CustomMsg **更简单**） |
| `kissicp_map.py` | ★★★★ | 直接可用（KISS-ICP 只吃点云） |
| `loop_lsq.py` | ★★★★★ | 复赛**更重要**，需加强（初赛只修 0.7%，因为被 RTK 掩盖了） |
| `rtk_fuse4.py` | ☆ | **不可用**（无 RTK） |
| `sfpnet_infer.py` | ★★ | 架构可复用，但权重/类别体系要换 |
| `semantic_label_v3.py` | ★★★ | 建图/投票/滤波逻辑可复用，类别表要换 |
| 录屏链 | ★★★★★ | `slam_online.py` + Xvfb + Tk 叠加 + ffmpeg **原样可用**（最省事的一块） |
| 打包/终审 | ★★★★★ | `fx2_pack.py` / `final_accept.py` 改成员清单即可 |

### 9.4 三周排期（建议）

| 周 | 任务 | 产出 / 判据 |
|---|---|---|
| **W1（9/18–9/24）** | ① 远端拉起 Gazebo + `house.world` + 车辆仿真器，确认能出点云/里程计/GT 位姿 ② 打通"仿真点云 → KISS-ICP → 回环 → 地图"最小闭环 ③ **建立 ATE 评估脚本**（用 GT） | 一条能在仿真里跑通并量化 ATE 的链 |
| **W2（9/25–10/1）** | ① **开放词汇语义**：IGLOSS（唯一"只用 3D 点云、类别用文本定义"的方案）落地，目标是把"沙发/凳子/可乐/花瓶/行人"用自然语言 prompt 分出来 ② 与 SK 19 类后端做 A/B ③ 目标关联（thing 聚类 + 实例 id） | 仿真环境内的物体级语义 + 实例关联 |
| **W3（10/2–10/8）** | ① 部署适配：模型量化（FP16/INT8）、ONNX/TensorRT、实测时延与体积 ② 泛化性实验：改光照 / 移动物件 / 换场景，量指标变化 ③ 录屏 + 打包 + 终审 | 复赛提交包 |

### 9.5 复赛技术路线候选（来自 `out/model_options.md`，按 收益÷风险 排序）

| 顺序 | 方案 | 为什么 | 工作量 |
|---|---|---|---|
| 1 | **SFPNet 权重替换**（**已完成**，就是 run4 用的） | 只动一个变量；已把"柱面划分在非旋转雷达上只 +1.2%"变成自己的实测 | 已完成 |
| 2 | **IGLOSS 零样本开放词汇** ← **复赛首选** | 唯一能在"不知道评委类别体系"时适配的方案；**推理只用 3D 点云**；单帧即可；类别用自由文本 prompt 定义 | 2–3 天 |
| 3 | 3DLabelProp（ICCV 2023） | 零样本域泛化证据最硬（SK 训练 → SemanticPOSS 50.4 / PandaSet 固态 55.2），且**只需要几何 + 序列 + 位姿**；但老栈 + C++ 编译 + 220 GB 预处理 | 4–6 天，风险最高 |
| 4 | S.MID 同传感器先验 | 唯一公开的、与我们传感器**相同**的带标注数据（CC BY-NC-SA 4.0，**非商业**，比赛用途需确认） | 2 天 |
| 5 | ScaLR+ 骨干替换 | 蒸馏数据含 PandarGT（MEMS 固态非旋转）→ 见过"无光束环"的输入 | 2 天 |
| — | 自监督域适配 / 伪标签自训 | 治本但三周做不完 | 备选 |

**必须记住的不确定性**：**没有任何模型在"中国城市 + 360° Livox Mid-360 + 19 类 SK 体系"这个组合上被评测过。**
最接近的两条各对一半（S.MID 对了传感器错了场景；SemanticPOSS 对了国家错了传感器）。**谁胜谁负只能自己量。**

### 9.6 复赛的验收判据（沿用初赛建立的方法论）

初赛最重要的方法论资产是**"无真值也能评"**这一套（详见 `AUDIT.md`）：
1. **几何伪真值**（不依赖标注）：每 1 m 格 z 的低 5% 分位当局部地面，`dz<0.5 m` ⇒ 表面、`dz>2.0 m` ⇒ 空中结构
2. **分歧点几何裁决**：两版判断相反的点，用几何定胜负
3. **物理不可能测试**：地图点到轨迹距离 > 传感器量程 ⇒ 坐标系错了
4. **外部仲裁**：OSM 路网（初赛用过）→ 复赛可换成 Gazebo GT / 室内平面图
5. **体素纯度**：区分"噪声"与"成片系统性错位"

**复赛新增优势：Gazebo 给完美 GT ⇒ 判据可以升级为直接算 ATE / mIoU。**

---

## 10. 常见故障速查

| 症状 | 最可能的原因 | 处置 |
|---|---|---|
| `$'\r': command not found` | 上传的 `.sh` 带 CRLF | 用 `put_lf.py`（二进制读 + `replace(b"\r\n", b"\n")`） |
| `ModuleNotFoundError: No module named 'rospkg'` | venv 隔离了系统 site-packages | 脚本里 `sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')` 与 `/usr/lib/python3/dist-packages` **加到末尾**（不要放 PYTHONPATH 前面） |
| `numpy.ndarray size changed, Expected 88 from C header, got 80` | PYTHONPATH 里系统 numpy 1.17.4 顶掉了 venv 的 1.24.4 | 同上，`sys.path.append` 到**末尾** |
| 进程莫名消失、日志停在某一行不动 | cgroup OOM（上限 42 GiB） | `cat /sys/fs/cgroup/memory.max` 确认；降规模（如位姿图用关键帧） |
| SSH 会话被自己杀掉（`remote_exit=-1`） | `pkill -f <pattern>` 匹配到了自己所在命令行 | 用 `pkill -f "[s]fpnet_infer"` 形式；别把待匹配串写进同一条命令的别处 |
| 点云 z 飞到几百米、不报错 | PCD `COUNT 1` 时加了子数组维度 → 静默广播 | `COUNT==1` 不要加维度；加 `z 跨度 > 60 m 则 abort` 断言 |
| 解出的栅格坐标飞到 20 万米 | 栅格键编码负坐标时 `//` 与 `%` 分解错 | 加偏移：`(cx+10000)*100000 + (cy+10000)`，解完再减 |
| `.npy` 读出长度差 128 | 用 `np.frombuffer(整个文件)` 把 128 B 头也算进去了 | 一律 `np.load` |
| 变速后视频时长没变 | `setpts` 只压缩 PTS | 时长要按 PTS 算；用 `speed_calc.py` 定倍率 |
| 视频"只拍了一角" | RViz 相机 `Focal Point` 没放在地图重心上 | 机位 `Distance 520 / Focal 地图重心 / Yaw 0 / Pitch 1.05`（见 §11） |
| 中文路径下 ffmpeg 报 `No such file` | Git Bash 对中文路径的假阴性 | 先 `shutil.copyfile` 到 ASCII 路径再跑 |
| `python -c` 里正则 0 匹配 | bash 吃掉反斜杠 | **带正则的脚本一律写成文件再跑** |
| zip 整体 md5 与远端不同 | zip 本地文件头存了 mtime | 比较**逐成员 md5 / 字节数**，相同即载荷一致 |

---

## 11. 相关文档与技能索引

| 文档 | 位置 | 内容 |
|---|---|---|
| 对外交付说明 | `out/run4_v2/README.md` | §0 修订摘要 / §1 方法链路（1.1–1.7）/ §2 提交文件 + 两处口径 / §3 定量结果 / §4 已知限制 |
| 语义审计 | `out/run4_v2/AUDIT.md` | 9 节：几何伪真值 / 分歧裁决 / 地面细分否证 / 自车 / 坐标系一致性 / RTK 假固定解四条判据合议 / 复算方法 / **没有解决的问题** |
| 模型选型调研 | `.workbuddy/out/model_options.md` | 5 条候选路线 + 硬约束 + 不确定性 |
| OSM 交叉验证 | `.workbuddy/out/osm_crossvalidation.md` | 第三方独立证据 |
| 语义审计（早期） | `.workbuddy/out/semantic_audit.md` | 第一版三判据 |
| **踩坑复盘** | `docs/复盘_踩坑手册_小白向.md` | 本项目所有坑的成因与解法 |
| 记忆 | `.workbuddy/memory/MEMORY.md` + `2026-09-11/12/13.md` | 决策、数字、坑（**写新记录前先读这里**） |

**已沉淀的技能**（`~/.workbuddy/skills/`）：

| 技能 | 用途 |
|---|---|
| `headless-rviz-recording` | 无显示、无 GPU 容器里录 RViz/OpenGL 成交付视频 |
| `lidar-map-forensics` | 审计 LiDAR+RTK 建图精度、判"剩余误差还能不能降"、审计语义标注质量 |
| `competition-submission-final-audit` | 提交前按赛题原文逐条核验交付包 |
| `groundtruth-free-semseg-audit` | 无人工标注时比较两套 3D 语义标注谁更准 |

---

## 12. 交接检查清单（接手第一天照做）

- [ ] 读本文档 §3 §4 §5（架构 / 链路 / 不变量）
- [ ] 读 `out/run4_v2/README.md` §0（知道这一版改了什么）
- [ ] 读 `docs/复盘_踩坑手册_小白向.md` 第 9 章（方法论）
- [ ] 验证远端可达：`cd .workbuddy/tmp && bash rr.sh "hostname; ls /root/gpufree-data/slam_comp | head"`
- [ ] 验证本地交付完整：`E:/software/python.exe .workbuddy/tmp/final_accept.py` → 应 PASS
- [ ] **不要动 `out/run4_v2/`**（它是提交基线）；要试新东西请复制到新目录
- [ ] **不要动 `snapshots/20260912_final/`**
- [ ] 改任何东西之前：先建回退点（`_bak/` 下 `cp`），再动手
