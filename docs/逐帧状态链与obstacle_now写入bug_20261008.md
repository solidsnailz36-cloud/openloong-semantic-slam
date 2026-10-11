# 逐帧状态链 + `obstacle_now` 写入 bug（2026-10-08）—— §66

> Boss 拍板：**先做 B4**（`obstacle_now` 10 帧 vs `obs_votes` 23 次的口径矛盾），
> 再做「锁死 `20 → 不可移除 → reach collapse` 的逐帧因果链」。
> **不碰 `20`、不调阈值、不跑大规模 A/B。**
>
> **本轮没有改动被测代码的任何行为**：只在 `--probe-cells` 的只读记录里加了列。
> 代价是 `replay_explore.py` 改了（`c679685be736` → `df69977490d4`），`worldmap.py` / `explore.py` 一字未动。

---

## 0. 三个结论

| # | 结论 |
|---|---|
| **1** | **逐帧因果链锁死。** `obstacle == after_rel_onp & ~removable` 在 **1489/1489 帧**成立（0 mismatch）。第 20 票落下的那一帧，`removable` **1→0**、`obstacle` **0→1**、`reach` **169.82 → 33.53 m²** —— 三件事**同一帧**。 |
| **2** | **★ B4 是真 bug，机制已定位且可一行证明。** `obstacle_now` 在**混合格**（本帧既有障碍点又有非障碍点）上**恒为 0**：13/13 帧。原因是 `self.obstacle_now[cyo, cxo] \|= o[ok]` 对**重复索引**不保证 OR 语义 —— 后写的覆盖先写的。 |
| **3** | **它不是"票来自别处"。** 那 13 帧该格**确实有 1 个障碍点**（`pts_obs_here = 1`），而且**全图 `obstacle_now` 非空**（和 214–332）。⇒ **是写入 bug，不是读数 bug，也不是时钟 bug。** |

---

## 1. 仪器：把过滤链复现出来（只读）

`worldmap.obstacle` 是一条五步链的**末端**，某格读 0 可能是三条规则里任意一条干的，
而 `removable` 只是 `WorldMap.add` 里的一个局部变量、**从不落盘**。所以探针处按**同一批数组**重算：

```
obstacle = votes >= 2
obstacle &= ~outvoted        # OBSTACLE_GROUND_RATIO，今天 = 0 ⇒ 跳过
obstacle &= ~released        # OBSTACLE_RELEASE_FRAMES / _RATIO
obstacle &= ~on_path
opened   = morph_open(obstacle, OPEN_PX)
unknown  = dilate(~seen, 3×3)
removable = obstacle & ~opened & ~unknown & (votes < OBSTACLE_STRONG_VOTES)
obstacle &= ~removable
```

新增 10 个只读列（`--probe-cells`）：
`votes_ge_min` / `after_outvoted` / `released` / `after_rel_onp` / `opened_keeps` /
`all_nb_seen` / `removable` / **`now_total`**（全图 `obstacle_now.sum()`）/
**`pts_here`** / **`pts_obs_here`**（该格本帧的落点 / 障碍点数，用与 `add` 相同的 `finite`+`_in` 口径）。

---

## 2. ★ 结论一：逐帧因果链锁死（守恒式 1489/1489）

```
--- A. conservation:  obstacle == after_rel_onp & ~removable ---
     checked 1489 frames ; mismatches: 0
```

⇒ **这条链就是完整故事**，没有"另一条路径"把该格写成障碍。

第 20 票前后的逐帧（`REM` = removable，`OBST` = 最终 `obstacle` 层）：

| frame | t(s) | votes | now | aft | opn | unk | **REM** | **OBST** | **reach m²** |
|---|---|---|---|---|---|---|---|---|---|
| 1086 | 234.225 | 19 | 0 | 1 | 0 | 0 | 1 | 0 | 169.82 |
| 1087–1090 | … | 19 | 0 | 1 | 0 | 0 | 1 | 0 | 169.82 |
| **1091** | **235.224** | **20** | 1 | 1 | 0 | 0 | **0** | **1** | **33.53** |
| 1092–1096 | … | 20 | 0 | 1 | 0 | 0 | 0 | 1 | 33.53 → 33.41 |

**`removable` 1→0、`obstacle` 0→1、`reach` 169.82→33.53 —— 全部发生在 f1091 这一帧。**

**同帧反事实**（不动代码，只改判据的读法）：

| 若切在 | `removable` | `obstacle` |
|---|---|---|
| `votes < 2` | False | 1 |
| `votes < 5` | False | 1 |
| **`votes < 20`（今天）** | **False** | **1** ← 塌陷 |
| `votes < 30` | True | 0 |
| `votes < 50/100/200/600` | True | 0 |

⚠️ **这张表只说明"这一格在这一帧"对该切点敏感，不构成"把 20 改大"的建议** ——
§65 已经量过 20 是 `nframes` 分布的 p32–35、65–68% 的格都过线，往上调会连带压掉真结构。
**列在这里只是为了说明"该帧的命运由这一个比较决定"。**

---

## 3. ★★ 结论二：B4 的答案 —— `obstacle_now` 的写入有 bug

### 3.1 现象面（13 帧）

23 个投票帧里，13 帧 `obstacle_now == 0`。而 `now_total`（全图）在这些帧上是 **214–332**，
⇒ **不是"整帧被清空"**。

### 3.2 决定性判据：把"点数"和"障碍点数"分开看

按 `(pts_here, pts_obs_here)` 分组，看 `obstacle_now`：

| `pts_here` | `pts_obs_here` | 帧数 | `now=1` | `now=0` | 应该是什么 |
|---|---|---|---|---|---|
| 0 | 0 | 1366 | 0 | 1366 | 0 ✅ |
| 1 | 0 | 55 | 0 | 55 | 0 ✅ |
| **1** | **1** | **9** | **9** | **0** | **1 ✅** |
| 2 / 3 / 4 | 0 | 45 | 0 | 45 | 0 ✅ |
| **2** | **1** | **12** | **0** | **12** | ⚠️ |
| **3** | **1** | **1** | **0** | **1** | ⚠️ |
| **2** | **2** | **1** | **1** | **0** | **1 ✅** |

- **纯障碍格**（`0 < pts_obs == pts`）：**10 / 10 帧 `now=1`**
- **混合格**（`0 < pts_obs < pts`）：**13 / 13 帧 `now=0`**

### 3.3 一行证明

```python
arr = np.zeros(4, bool)
arr[np.array([1, 1])] |= np.array([True, False])    # two points in the SAME cell
arr[1]   ->  False      # OR semantics would give True
arr[np.zeros(4, bool)][0]      # --
arr2 = np.zeros(4, bool)
arr2[np.array([1, 1])] |= np.array([False, True])
arr2[1]  ->  True
```

**`arr[idx] |= mask` 在 `idx` 有重复时等价于 `arr[idx] = arr[idx] | mask`，而 `__setitem__`
对重复索引是"后写覆盖"——不保证 OR。** 本项目的实际数据里，障碍点**总是排在非障碍点之前**，
所以混合格 13/13 全被写成 `False`。

### 3.4 波及范围（**都要复核**）

1. **`blocked() = obstacle | obstacle_now`** —— 当帧分量在这些格上丢失。
   累积层通常兜得住，但**首次出现的物体**（票还没到 2）在这些格上等于"本帧看不见"。
   这直接影响 `explore.py` 的 `_ahead_clear`（它读 `blocked()`）。
2. **`--dump-obs-features` 的 `nb8`**（逐帧 8 邻域障碍计数）建立在 `obstacle_now` 上 ⇒ 同样被削弱。
   **这解释了 §65 的一个观察**：累积量 `nbm`（AUC 0.650/0.733）比逐帧的 `nb8` 有信息得多。
3. 凡是用 `obstacle_now` 做过的**逐帧**读数（本项目 §56 起的若干处），要按"混合格会漏"复核。
   **用 `obs_votes` 的结论不受影响**（它走 `np.unique`，不经过这个写入）。

### 3.5 一行修法（**本轮没改，等拍板**）

```python
# today
self.obstacle_now[cyo, cxo] |= o[ok]
# one-line fix: write only the obstacle points, always True -> no repeat-index writeback
self.obstacle_now[cyo[o[ok]], cxo[o[ok]]] = True
```

⚠️ **它是"修 bug"，不是"调参"**（不涉及任何阈值、判据、形态学），但它**会改变行为**：
`blocked()` 变完整 ⇒ 局部避障更保守 ⇒ **所有 A/B 的基准都会变**。
按纪律**先报不改**；要动请单独拍板，并重跑 Gate −1 建立新基准。

---

## 4. 复现

```bash
bash /work/ab/_chain_run.sh        # r3：--probe-cells + --probe-points + --strand-dump
python3 out/_vote_chain.py --probe /work/ab/pc4_chain66.csv \
        --strand /work/ab/chain66_strand.csv --cell 0
```

- **Gate −1**：带 `--probe-cells/--probe-points/--strand-dump` 的决策列 md5 = **`05b323896edebc9a`**
  = 不带这些仪器的同一采集（§65）⇒ 仪器逐位 no-op。
- **指纹**：`tools/replay_explore.py **df69977490d4**`（只加只读列）｜
  `r2slam/worldmap.py 606701c17b6e`（**未动**）｜`r2slam/explore.py aa2f11f2d4fc`（**未动**）。
- **产物**：`out/island2/_pc4_chain66.csv`（22 列）、`_chain65_strand.csv`、`_chain65_dec.csv`；
  读数器 `out/_vote_chain.py`；驱动 `out/_chain_run.sh`。

---

## 5. 没做的

- **没改 `obstacle_now`**（§3.5 只是提案）；**没碰 `20`**；没动 release/开运算；没跑 A/B。
- §65 的两个候选（B1 `nbm`、B2 连通性豁免）**按 Boss 的拍板搁置**；
  且 §65 已证明它们各有反例（r2 门口的机制 A 假障碍 `nbm = 226/226/337`）。
- **提交包未动**：`ec8260d28de6590033be36daca247b6c` / 36,204,882 B。

## 6. 下一步（等 Boss 拍）

1. **修 §3.5 那一行**（纯 bug 修复，重建基准）—— 我建议**先做它**，因为它让后续所有读数干净。
2. 再做 **B3 拓扑保险丝**的最小反事实实验：不加新判据，只问
   "**这一帧新增的障碍是否把机器人当前可达世界砍掉 > X m²**"，
   若是则让该格进入"待确认"而不是立刻成为不可移除障碍。
   —— 与目标一致：**宁可暂时不信一个可疑障碍，也不要让一个格把整个房间从地图上删掉。**
