# `obstacle_now` 写入 bug 的修复（2026-10-08）—— §67

> Boss 拍板：**只改那一处**，先红后绿，然后重建基准、复跑同一条逐帧链。**不做 B3。**

---

## 0. 结论先行

| # | 结果 |
|---|---|
| **1** | **单元测试：修前 16 checks / 2 failed，修后 0 failed。** 两个失败点正是"混合格"，而且 **C 与 C2 是同一批点、只换顺序 —— 结果相反**（`False` vs `True`），这就是"后写覆盖"的判决性证据。 |
| **2** | **★ 改前 vs 改后：r2 与 r3 的决策列 md5 完全相同**（`4e73b0d9086e05e6` / `05b323896edebc9a`）。⇒ **这个 bug 在这两个采集上改变的行为是 0** —— 累积层 `obstacle` 兜住了。 |
| **3** | **第 20 票链完全不变**：守恒式仍 **1489/1489**；`removable` **1→0**、`obstacle` **0→1**、`reach` **169.82→33.53** 仍在**同一帧 f1091**；**reach 序列逐帧完全相同（0 个不同帧）**。 |
| **4** | **`obstacle_now` 的读数恢复正常**：23 个投票帧里 `now == 0` 的从 **13 → 0**。 |
| **5** | ⇒ **B4 与机制 B 严格分层，互不解释。** B4 是独立的感知层写入 bug；机制 B 是独立的地图拓扑机制。**"发现一个 bug 就把所有现象归咎于它"被数据直接排除。** |

---

## 1. 改动（一行）

```python
# r2slam/r2slam/worldmap.py, WorldMap.add()
-            self.obstacle_now[cyo, cxo] |= o[ok]
+            self.obstacle_now[cyo[o[ok]], cxo[o[ok]]] = True
```

**指纹**：`worldmap.py **606701c17b6e → 06087bc5596f**`（宿主与容器已同步、逐位一致）。

**没有同时改**：`20` / `votes >= 2` / `removable` / `release` / `on_path` / `reach` / 任何阈值。
其余指纹与 §66 相同：`explore.py aa2f11f2d4fc`｜`nav.py 994444576714`｜
`tools/replay_explore.py df69977490d4`｜`tools/online_pipeline.py 1308406f78a1`。

---

## 2. 单元测试（`out/_now_write_test.py`）

直接对着 **numpy 语义**写，而不是对着管线写 —— 因为这是个 numpy 语义 bug。
用真实的 `WorldMap` + `add()`，只读 `obstacle_now` / `_obs_votes`。

### 2.1 修前（红）

```
16 checks, 2 failed
   FAIL  obstacle_now[cell]                    (got False, want True)   <- 场景 C
   FAIL  cell1 (obs first, two floors after)   (got False, want True)   <- 场景 E
```

### 2.2 ★ 判决性对照（修前）

| 场景 | 输入 | `obstacle_now` | |
|---|---|---|---|
| **C** | 同格两点：**障碍在前**、非障碍在后 | **False** | **FAIL** |
| **C2** | 同格两点：**非障碍在前**、障碍在后 | **True** | PASS |

**同一批点、只换顺序、结果相反** —— 排除了"过滤器不同""口径不同"等解释，只剩写入语义。

### 2.3 修后（绿）

```
16 checks, 0 failed
```

场景覆盖：单障碍点 / 同格两障碍点（票只 +1）/ **障碍在前+非障碍在后** / 反序 /
全非障碍 / 一帧多格混合 / 跨帧清零（`obstacle_now` 清、`_obs_votes` 保持单调）。

---

## 3. Gate −1

| 对比 | 结果 |
|---|---|
| **仪器 on vs off**（改后，r3） | `05b323896edebc9a` = `05b323896edebc9a` ✅ |
| **★ 改前 vs 改后**（r3，不带仪器） | `05b323896edebc9a` = `05b323896edebc9a` ✅ **行为零变化** |
| **★ 改前 vs 改后**（r2，不带仪器） | `4e73b0d9086e05e6` = `4e73b0d9086e05e6` ✅ **行为零变化** |

⇒ **决策列 md5 与修复前完全相同**，所以**基准不用换**（但 `worldmap.py` 的指纹必须更新）。

**为什么行为不变**：`blocked() = obstacle | obstacle_now`。丢掉的格在**同一帧**还没进 `obstacle`
（票 < 2 或刚被摘），但在**本采集里这些格下一帧/下一段就有票**，累积层补上了。
⇒ **这个 bug 要显形，需要"某格第一次出现障碍回波、且该帧同时有非障碍回波、且累积层还没覆盖"**
—— r2/r3 里没满足这三条同时成立的场合。

---

## 4. 复跑同一条逐帧链（读数恢复，机制不变）

### 4.1 守恒式仍成立

```
obstacle == after_rel_onp & ~removable      checked 1489 frames ; mismatches: 0
```

### 4.2 `obstacle_now` 读数恢复正常

| 指标 | 修前（§66） | 修后 |
|---|---|---|
| 23 个投票帧里 `obstacle_now == 0` 的 | **13** | **0** |

### 4.3 ★ 第 20 票链完全不变

| frame | votes | **REM** | **OBST** | **reach m²** |
|---|---|---|---|---|
| 1086–1090 | 19 | **1** | **0** | 169.82 |
| **1091** | **20** | **0** | **1** | **33.53** |
| 1131 | 21 | 0 | 1 | 33.26 |
| 1140 | 22 | 0 | 1 | 33.17 |
| 1173 | 23 | 0 | 1 | 33.22 |

**`removable` 1→0、`obstacle` 0→1、`reach` 169.82→33.53 —— 与修前逐格相同，仍在同一帧。**
同帧反事实也一样：`votes < 20` ⇒ `obstacle=1`；`< 30` ⇒ `obstacle=0`。

### 4.4 reach 序列逐帧不变

```
rows 1489 vs 1489 ;  identical: True ;  differing frames: 0
```

⇒ §61/§62/§65/§66 里所有 `reach` 与塌陷时刻的读数**不受这次修复影响**。

---

## 5. 一个附带的确认：`reach` 本来就不该受这个 bug 影响

`PathPlanner.free_space`（`nav.py` 457–483）读的是 **`wmap.ground`**（`ground_ever & ~obstacle`），
**不是 `blocked()`**。注释里还留着一段实测记录：

> "It briefly also removed **this frame's** returns. That was a mistake with a clear signature …
> the robot **spins instead of driving**. Measured on the simulator house, same duration:
> `turn` 182 → 468 frames (12% → 39%), forward motion 83% → 58%, distance 109 → 52 m."

⇒ **当帧回波层只喂 `_ahead_clear`（安全层）**，所以这次修复的**影响面被限制在局部避障**，
而它在 r2/r3 上没有改变任何决策（§3）。

---

## 6. 没做的（按 Boss 的边界）

- **没碰 `20`**、没动 `votes>=2`、没动 `removable`、没动 release、没动 `on_path`、没动 `reach`、没动任何阈值。
- **没做 B3**（拓扑保险丝）。按 Boss 的顺序：**B4 修完并复跑同一条链之后**才讨论。
- **提交包未动**：仍是 `ec8260d28de6590033be36daca247b6c` / 36,204,882 B
  ⚠️ **包里的 `worldmap.py` 仍是修复前的版本** —— 下次打包会带上这一行。
- 没跑其余采集。

## 7. 复现

```bash
# 单元测试（红/绿）
python3 out/_now_write_test.py

# 三组验证
bash /work/ab/_fix_run.sh          # r3 plain / r2 plain / r3 with instruments
python out/_fix_cmp.py --a out/island2/_oft_r3.csv --b out/island2/_fix_r3.csv --label "..."
python3 out/_vote_chain.py --probe /work/ab/pc4_fix.csv --strand /work/ab/fix_chain_strand.csv --cell 0
```

**产物**：`out/island2/_fix_r3.csv`、`_fix_r2.csv`、`_fix_chain.csv`、`_pc4_fix.csv`、`_fix_chain_strand.csv`；
工具 `out/_now_write_test.py`、`out/_fix_cmp.py`、`out/_fix_run.sh`。

## 8. 状态与本轮之后

| 问题 | 状态 |
|---|---|
| **B4**：`obstacle_now` 写入 bug | **已修、已验证、行为零变化** |
| **机制 B**：`20 → 不可移除 → reach collapse` | **逐帧锁死、未修**（且已证明与 B4 无关） |
| **机制 A**：低票假障碍切断窄口 | `ratio=1` 机制成立、**未封板** |
| **B3**：拓扑保险丝 | **未做**（等 Boss 拍板） |
| **真值工具** | 已修（§64） |

**下一步（等 Boss 拍）**：现在 `obstacle_now` 干净了，可以谈 B3 —— 但不加新判据，
只问"**这一帧新增的障碍是否把机器人当前可达世界砍掉 > X m²**"，若是则该格进"待确认"
而不是立刻成为不可移除障碍。
