# Tank AI：26×26 坦克大战

## 新入口：Rust 批量学习与真实 ROM 周期验收

`rust_train.py study` 用实验性 Rust 引擎训练 CNN、空间 Transformer 和16步历史
Transformer，并定期回真实 ROM 做完整关卡和技能评测。默认三种模型各两个种子，
每次最多262,144步，每65,536步验收，每16,384步保存 checkpoint，最多两组并行。
完整关卡验收保留12,000帧上限；不会把性能测试用的600帧时长当作正式效果评测。

```powershell
uv run --no-sync python rust_train.py study --output runs\rust-monitored-study `
  --models cnn,spatial,history --seeds 0,1 --device cuda --codec cuda --parallel 2
```

此路线保持10个原始动作和独立 seeded PRNG，**不宣称与 ROM 随机流等效**；
原有15动作路线及旧实验输出不改。所有权重从零训练，不静默转换旧模型。
有验收改善才考虑额外预算，无改善按门槛停止。
模型、历史、内存、评测、停训与签名细则见 [Rust 后端说明](docs/rust_backend.md)。

## 当前工作方式：先验收，再训练

约 1046 万步的长实验已暂停，保留在 `runs/20260928-boundary-continue347136-g099/`。
固定评测长期没有稳定通关，因此不会自动接着跑到 2000 万步。

当前恢复流程分三部分：

1. v7 修正地图中临时坦克占位 bit7，保留真实材料与残砖形状，避免误造“幽灵墙”。
2. 六类**真实 ROM 技能场景**单独测射击、侧向躲弹、对轰、残砖清障、拾取、单敌守家。
   技能成功不是整关胜利，测试场景也不是普通游戏的随机开局。
3. 有限预算对照：从零基准 / 旧权重初始化 / 更长折扣 / 精简奖励；
   每组两个训练种子、每个约 5 万步，最多两组并发，并设置停训门槛。

所有组使用同一修正后的环境和评测协议。短预算没有通关只能报告“尚无证据”，
不能自动加时，也不能把混合奖励上升称为获胜能力提升。
命令、预算、指标和结果目录见 [有限预算恢复实验](docs/recovery_experiments.md)。

原版 NES 模拟器仍是生产训练与验收基准。ROM 主线保留两种打法：

- **MaskablePPO**：47×26×26 语义观测，10 个原始动作＋5 个启发式宏。
- **启发式基线**：与学习策略共享状态、子弹威胁、射线和 26×26 寻路。

不再保留像素学习、13×13 地图、旧版机器人或静默加载旧 checkpoint 的兼容层。
坦克位置是像素中心；寻路检查完整的 16px 坦克占地，不允许穿过 8px 窄缝。
残砖保留四个 4px 小块：坦克仍按原版 8px 单元阻挡，射击和拆砖使用细分形状。
追加的子弹危险图考虑坦克宽度、残砖与墙体遮挡，表示未来 24 个游戏帧内的命中紧迫度。
携带道具的红坦有独立标记，首次受击掉落后取消标记，不靠屏幕闪烁颜色识别。
不能发射或当前无法对齐拦截时禁用 COUNTER；有砖实际挡住的基地方向射击允许，
不会仅因朝向基地或砖属于基地外围就禁止。
已经贴住外边界且面向外侧时，重复的向外移动 / 开火被屏蔽；有用的转向、沿边移动、
朝场内射击和正常拆砖仍保留。

## 实验性 Rust 后端与无损内存优化

独立入口为 [rust_train.py](rust_train.py)，不替换以上 ROM 主线。
[Rust 镜像](native/src/engine.rs) 使用本项目最新的敌军、道具、计时器、碰撞次序与残砖逻辑，
而不是旧性能 worktree 的引擎快照。性能方案只借鉴并适配无损 rollout、批量调用和缓冲区复用。

- 使用版本化独立随机流，同 seed、同配置可复现；**不是 ROM 随机流复刻**。
- 只用 **10 个原始动作**；不能直接加载现有 15 动作模型。
- rollout 保留 float32 的每一位，仅按 minibatch 解压，可选 CUDA 展开；不做 float16 量化。
- `--steps` 必须整除 `envs*n_steps`，拒绝超预算取整，不自动启动长训练。
- 原版 ROM 用于一次性初始化地图和独立迁移评测，不参与 Rust rollout 的逐帧执行。

```powershell
cargo build --release --manifest-path native\Cargo.toml
# 同配置 dense / compact 短对照，要求最终参数哈希一致
uv run --no-sync python rust_train.py compare --output runs\rust-compare `
  --device cuda --codec cuda --envs 32 --n-steps 64 --batch-size 256 --steps 4096
# 在真实 ROM 中分别评测确定性和采样策略；不代表已经学会通关
uv run --no-sync python rust_train.py eval --model runs\rust-compare\cuda\model.zip `
  --output runs\rust-compare\rom-transfer.json --device cuda
```

当前能力边界、完整命令和延期事项见 [Rust 后端说明](docs/rust_backend.md)。
不要把条件随机数一致的轨迹测试、短跑速度或压缩率当作独立 ROM parity 或胜率。

## 安装与测试

Python 3.11，依赖由 `pyproject.toml` / `uv.lock` 固定。项目目录需要本地
`battle_city.nes`；不要替换 ROM 后继续使用原模型。

```powershell
uv sync
cargo build --release --manifest-path native\Cargo.toml
uv run python -m unittest discover -v
```

已有 CUDA PyTorch 环境时可用 `uv run --no-sync` 避免替换已配置的 PyTorch。
使用 `--device cpu` 可以完成所有测试和短训练；正式训练可用 `--device cuda`。
nes-py 依赖的旧 Gym 可能打印维护状态提示，不影响本项目的 Gymnasium 接口。

## 训练

每次实验必须使用**新的输出目录**，避免混合不同环境、参数和权重。

```powershell
# 短验收：只验证管线，不代表模型已经学会获胜
uv run --no-sync python train.py --output-dir runs\smoke `
  --steps 16 --envs 2 --device cpu --n-steps 8 --batch-size 8 --n-epochs 1 `
  --max-frames 120 --eval-every 16 --train-stages 1 --eval-stages 2 `
  --eval-seeds 101 --benchmark-stages 3 --benchmark-seeds 456

# 分阶段训练：击杀目标 → 防守目标 → 完整关卡
uv run --no-sync python train.py --output-dir runs\curriculum-g099 `
  --steps 300000 --combat-steps 50000 --defense-steps 50000 `
  --envs 6 --device cuda --gamma 0.99
```

课程阶段按预算推进，不伪装成能力已达标：

| 阶段 | 回合成功条件 |
| --- | --- |
| `combat` | 真正摧毁至少一个敌人；若有道具在场，留出最多 600 帧拾取时间 |
| `defense` | 基地存活至少 900 帧，且摧毁至少两个敌人 |
| `full` | 原版游戏中本关敌人全部清除，基地与玩家仍有生存条件 |

课程使用真实关卡，不依赖未经验证的 RAM 改写。多个训练 worker 时保留一个完整关卡
worker；完整关卡评测不会使用课程成功指标。预算会向上取整到完整 PPO rollout。
首次击杀时有道具在场，不再立即重置；拾取、道具离场或 600 帧窗口结束后才完成课程。
窗口不因后续掉落而无限延长，基地被毁、命耗尽与总时限仍按原规则处理。

默认训练关卡为 1–3，模型选择评测为 4–5，留出基准为 6–7；三组关卡必须互不重叠。
`--train-stages` / `--eval-stages` / `--benchmark-stages` 可修改。
`--gamma 0.997` 或 `0.999` 可建立对照实验，但没有哪个数值被预先认定更优。

输出包含：

- `manifest.json`：配置、固定评测种子、环境版本、源文件及 ROM 哈希；
- `best.zip`：按确定性完整关卡通关率选择；
- `latest.json`：原子发布的最新权重指针；`checkpoints/` 只保留最近三份；
- `final.zip`、各阶段 checkpoint：每个都有 `.zip.manifest.json`；
- 定期/阶段结束评测 JSON，以及 `training-result.json`。

续训指定 `.zip`，提供新的输出目录，并保持原有配置；`--steps` 是新的累计目标。
环境语义、源码、ROM、观测或关键配置不兼容时会明确拒绝加载，不做静默迁移。
最新模型默认每约 10000 步发布（`--checkpoint-every`），独立于 `--eval-every`；
仅在优化完成后发布。启动时也发布步数为 0 的初始化模型，窗口会明确显示 0 步。

### 外边界修正版接续训练

```powershell
uv run --no-sync python train.py --output-dir runs\boundary-continue `
  --continue-from runs\20260928-tactics-warm53760-g099\checkpoints\step-000000347136.zip `
  --steps 20000000 --combat-steps 500000 --defense-steps 1500000 `
  --envs 6 --device cuda --gamma 0.99
```

`--continue-from` 是明确审核过的 v5→v6 同布局接续，保留权重、优化器、
训练步数和课程进度；`--steps` 是累计目标，不是额外增加的步数。
三种加载方式互斥：`--resume` 要求同版本；`--continue-from` 只接受指定的兼容修正；
`--warm-start` 是此前 v3→v4 / v4→v5 扩展观测时的只迁移权重方案，会重置优化器和进度。
当前 v6 不会静默绕过版本检查加载任意历史模型。
所有迁移 / 接续都记录来源和上游记录，但不保证策略效果更好。

## 固定评测

```powershell
uv run --no-sync python eval.py --model runs\curriculum-g099\best.zip `
  --output runs\curriculum-g099\benchmark.json --device cuda

uv run --no-sync python eval.py --heuristic --stages 6,7 --seeds 1001,2002 `
  --output runs\heuristic-benchmark.json
```

模型评测分别报告确定性和采样动作；启发式也使用相同模式标签与开局。
报告包含逐关胜率、基地被毁/命耗尽/超时、真实击杀/发射/掉命/拾取、停滞和奖励分项。
另有道具掉落次数、出现机会和拾取率；没有道具机会时拾取率为 `null`，不误报为 0%。
外边界无效动作单独统计累计帧数、最长连续时长和占比，不能只看“视频动起来了”。
课程 `objective` 成功不计入通关率。自定义 `--stages` / `--seeds` 标为自定义评测，
不要当作未见测试集。短测试时长必须在训练和模型评测间保持一致。

重置仍是同一关卡存档加 0–23 帧等待；种子数量不等于独立局面数量。
跨关卡留出评测用于补充这一局限，不能只凭训练回报判断能力。

## 观战

```powershell
uv run --no-sync python watch.py --heuristic --stage 1
uv run --no-sync python watch.py --model runs\curriculum-g099\best.zip --stage 1
uv run --no-sync python watch.py --run-dir runs\curriculum-g099 --stage 1 --device cpu
uv run --no-sync python watch.py --run-dir runs\boundary-continue --stage 1 --device cpu --sampled
```

`--model` 固定权重；`--run-dir` 只跟随指定实验，在回合结束后加载最新完整发布的权重，
窗口显示正在使用的训练步数。仍严格检查环境和源码签名，不跨实验热切换。
训练发布第一份 checkpoint 后即可启动跟随模式。回合结束显示真实结果后再重置。
`--sampled` 展示策略的采样动作；不加时取最高概率动作。两种模式都需评测，
采样减少僵硬循环不等于已经学会通关。

watch 默认录制无声 MP4，按“通关优先 → 击杀多 → 掉命少”保留最好的 10 局完整实战。
同分时通关更快者优先，失败局存活更久者优先；完全相同时保留较早的录像。
游戏画面按真实游戏时间编码（默认 15fps），邻近像素放大到 768×720。

- 跟随模式默认保存到该实验的 `recordings/`，`rankings.json` 记录成绩、种子、
  模型步数、checkpoint 哈希和环境版本，重启 watch 后继续保留榜单。
- 每局流式写入临时文件，结束后才入榜；不够前十或退出时未完成的录像会丢弃。
- `--record-dir` 可指定目录；`--no-record` 可关闭录像。编码需要系统 FFmpeg。
- 只记录游戏画面，不录桌面或麦克风；保存录像不会额外保留对应的旧 checkpoint。

## 代码导航

| 文件 | 职责 |
| --- | --- |
| `game_state.py` | 唯一 RAM 解码、子弹身份和逐帧事件 |
| `battle_city_env.py` | 47 通道观测、动作掩码、课程终点、奖励和计数 |
| `heuristic.py` | 完整占地寻路、瞄准、防守、躲弹、对轰、道具 |
| `policy.py` | 26×26 CNN |
| `migration.py` | 明确版本与布局的权重迁移，非静默兼容 |
| `experiments.py` | 版本校验、模型加载、固定评测、指标 |
| `recordings.py` | MP4 流式录制、前十排名与持久化 |
| `skills.py` | 六类真实模拟器技能场景与独立成功率 |
| `ablation.py` | 有限预算单变量对照、并发上限与停训门槛 |
| `train.py` / `eval.py` / `watch.py` | 三个入口 |
| `test_*.py` | 状态、真实模拟器事件、寻路与工作流回归 |

详见 [RAM 校准记录](docs/ram.md) 和 [奖励定义](docs/reward_design.md)。
新增实验优先沿这套状态和评测接口扩展，不再复制 `v2` / `v3` 训练入口。
