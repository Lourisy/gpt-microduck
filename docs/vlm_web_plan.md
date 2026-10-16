# Microduck VLM Web 控制方案

## 目标

在不修改官方 Microduck 运控代码和官方 ONNX 权重的前提下，新增一个网页入口，让 GPT-6-Astra 根据用户文本、机器人视觉和本体状态，逐个调用 Microduck 技能。

第一版只运行在 MuJoCo 仿真环境中，使用 `official_policies/` 下的不带轮滑鞋权重：

- `alpha_walking.onnx`
- `alpha_stand.onnx`
- `alpha_sitstand.onnx`
- `alpha_ground_pick.onnx`
- `ball_kick_left.onnx`
- `ball_kick_right.onnx`
- `roulade.onnx`

官方无 VLM 的运行方式必须继续保持可用。因此 `scripts/infer_policy.py`、官方权重和原有命令行路径保持不变，VLM 只通过新增入口复用底层能力。

## 已确认的底层接口

官方 ONNX 不接收视觉或深度信息。

输入是 61 维 observation：

```text
3   base angular velocity
3   projected gravity 或 raw accelerometer
14  relative joint position
14  joint velocity
14  last action
13  command
```

输出是 14 维 action。目标关节角由底层执行器计算：

```python
target_position = DEFAULT_POSE + action * action_scale
```

VLM 的视觉信息只用于高层决策和技能完成判断，不进入官方 ONNX。

MuJoCo 模型已有 `head_camera`。服务端同时渲染一个跟踪 `trunk_base` 的第三视角相机供人查看，并从 `head_camera` 获取 RGB 图像低频发送给 VLM；底层 ONNX 继续以 50 Hz 运行。`head_camera` 按官方 camera 实现先修正 180° 朝向，再在网页输出中顺时针旋转 90°，与真实相机消费方的方向一致。第三视角只用于网页显示，不发送给 GPT。

## 总体架构

```text
浏览器
  ├─ 输入用户文本
  ├─ 显示第三视角仿真画面和 head_camera 图像
  ├─ 显示当前技能和状态
  └─ 发送停止命令
          │ HTTP polling + POST
          ▼
scripts/vlm_web.py
  ├─ MuJoCo 仿真循环
  ├─ 官方 ONNX 技能执行
  ├─ 第三视角和 head_camera 渲染
  ├─ GPT-6-Astra 调用
  └─ 浏览器 HTTP 服务
          │ HTTP Responses API
          ▼
https://tokenhub.ilab.beer/v1/responses
```

第一版尽量少引入依赖：

- Python 标准库 `urllib`：调用中转站的 Responses API；
- 标准库 HTTP：浏览器和仿真服务之间的连接；
- `http.server`：提供静态 HTML；
- MuJoCo renderer：生成相机图像。

不引入 FastAPI、Uvicorn 或前端构建系统。网页先使用一个静态 HTML 文件和少量 JavaScript。

建议新增的文件只有：

```text
scripts/vlm_web.py
scripts/vlm_web.html
docs/vlm_web_plan.md
```

第一版不新增 `src/mjlab_microduck/` 内的模块，避免和训练环境、任务注册及官方推理路径产生依赖。

## VLM 调用时机

采用 HomeBody 的“一次调用决定一个 skill”模式，而不是固定频率调用 GPT：

1. 用户提交一条文本任务。
2. 服务端采集当前相机图像和状态，调用 GPT-6-Astra。
3. GPT 返回一个技能及其参数。
4. 本地执行这个技能。
5. 技能完成或超时后，服务端采集新的视觉和状态。
6. 服务端再次调用 GPT，让 GPT 判断上一个技能是否完成、是否异常，并决定下一个技能或 `done`。

因此，GPT 调用频率由技能生命周期决定：

- 不是 50 Hz；
- 不是固定每秒轮询；
- 每个 skill 至少对应一次执行前决策和一次执行后判断；
- 网络请求不能阻塞 50 Hz 的 MuJoCo/ONNX 控制循环。

服务端不把跌倒或一般碰撞作为技能结束条件。技能只由本地完成条件结束；如果完成条件一直没有到达，则在统一的安全时间上限后返回 `timeout`，再进入下一次 GPT 调用。

## walking 设计

walking 不在执行器内部自动变速，也不隐藏启动加速或反向补偿。VLM 给出的 command 直接传给官方 walking policy：

```text
vx -> command[0]
vy -> command[1]
wz -> command[2]
```

walking 的时间由 `duration_s` 控制：

```json
{
  "skill": "walking",
  "arguments": {
    "vx": 0.3,
    "vy": 0.0,
    "wz": 0.0,
    "duration_s": 2.0
  }
}
```

在相同速度 command 下，较短 duration 表示较短移动距离，较长 duration 表示较长移动距离。VLM 在每个 walking skill 结束后读取新的相机图像和状态，再决定是否继续 walking、停止或执行其他技能。

`duration_s` 的范围是 `0.1 <= duration_s <= SKILL_TIMEOUT_S`，当前 `SKILL_TIMEOUT_S=10.0`。

官方权重目前存在低速起步现象：静止时 `0.2 m/s` 可能不动，而已经行走时可以降低速度。第一版不在底层修复这一行为，而是在系统提示中明确：

- walking command 必须使用官方训练范围；
- 低速 command 可能无法从静止起步；
- walking 的移动精度优先通过 `duration_s` 控制；
- VLM 应根据执行后的图像和状态判断是否真的发生了移动。

这样可以保持执行器语义简单，也不会改变官方策略的输入分布。

## 可用技能

### `walking`

使用 `alpha_walking.onnx`。

参数：

```json
{
  "vx": "m/s",
  "vy": "m/s",
  "wz": "rad/s",
  "duration_s": "seconds"
}
```

### `stand`

使用 `alpha_stand.onnx`，保持站立或恢复站立。

参数：`{}`。本地等待躯干回到站立高度并稳定 `0.4 s`。

### `sitstand`

使用 `alpha_sitstand.onnx`。

参数：

```json
{
  "mode": "sit | stand"
}
```

本地固定运行 `3.0 s`；`sit` 完成后保持坐姿，不会自动恢复站立。下一次 GPT 调用通过视觉判断动作是否成功。

### `ground_pick`

使用 `alpha_ground_pick.onnx`，沿用官方 phase command 编码，由本地执行器负责 phase 推进。

参数：`{}`。本地推进官方 phase，phase 到达结束值后才算完成。第一版不让 VLM 直接设置 phase。

### `kick_left` / `kick_right`

分别使用对应的 ball kick ONNX。执行器加载带球场景，并使用官方的球位置初始化逻辑。

参数：`{}`。本地使用官方 kick controller 的内部计时器等待结束。

### `roulade`

使用 `roulade.onnx`。

参数：`{}`。本地使用 roulade controller 的内部计时器等待结束。

### `done`

表示用户任务已经完成，或 GPT 判断当前任务无法继续完成。执行器停止动作并保持安全状态。

参数：

```json
{
  "summary": "short completion summary"
}
```

## VLM 输入

每次调用发送：

```text
system/context:
  - Microduck 机器人和可用技能说明
  - 61D ONNX observation 与技能 command 约定
  - 每个技能的参数范围和本地完成条件
  - walking 低速起步限制
  - JSON 输出格式

user/current turn:
  - 用户原始文本任务
  - 当前 head_camera RGB 图像
  - 当前技能
  - 当前本体状态摘要
  - 上一个技能的参数
  - 上一个技能的执行结果
  - 最近几步的历史摘要
```

本体状态只发送高层摘要，不发送完整 61D observation。例如：

```json
{
  "upright_score": 0.99,
  "command": {"vx": 0.3, "vy": 0.0, "wz": 0.0},
  "skill": "walking",
  "elapsed_s": 1.8
}
```

不保存或依赖模型内部思考，只保存 GPT 输出中的短 `text` 字段和结构化执行结果。

## VLM 输出

统一使用一个 JSON 对象：

```json
{
  "text": "The robot is upright and should walk toward the object.",
  "skill": "walking",
  "arguments": {
    "vx": 0.3,
    "vy": 0.0,
    "wz": 0.0,
    "duration_s": 2.0
  }
}
```

任务结束时：

```json
{
  "text": "The requested action is complete.",
  "skill": "done",
  "arguments": {
    "summary": "..."
  }
}
```

本地必须做 JSON 解析和参数校验。非法输出、未知技能、越界参数或 API 错误都不能直接进入控制器。

## GPT 对技能完成的判断

本地执行器只报告可测事件：

- walking 的 `duration_s` 到达；
- phase 到达结束值；
- kick/roulade 的内部计时器到达结束值；
- sitstand/stand 的固定 `3.0 s` 等待时间；
- MuJoCo 仿真继续运行；
- 是否发生数值异常；
- 当前 command 和 skill 名称。

GPT 在执行结束后结合新的视觉图像判断：

- 上一个技能是否完成；
- 是否出现异常姿态或明显失败；
- 是否需要重复技能；
- 是否需要先站立恢复；
- 是否应该结束任务。

例如，`ground_pick` 本地 phase 已结束，但 GPT 从相机中判断没有成功拿起目标，则下一次输出可以继续 `ground_pick` 或先 `stand`。第一版不额外实现物体检测器，把物体是否拿起的判断留给视觉模型。

## 超时保护

第一版不实现跌倒或一般碰撞触发，也不把脚部接触地面作为技能事件。

每个技能都在本地等待自己的完成条件：walking 等待 `duration_s`，ground_pick 等待 phase 结束，kick/roulade 等待内部计时器结束，sitstand/stand 固定等待 `3.0 s`。如果条件在 `SKILL_TIMEOUT_S`（当前 10 秒墙钟时间）内没有到达，服务端返回 `timeout`，记录当前观测并调用下一次 GPT。`upright_score` 仍作为状态摘要提供给网页和 GPT，但不触发任务停止。

## 网页通信

浏览器到服务端使用标准 HTTP 接口。第一版使用 POST 发送任务，浏览器定期轮询状态和相机帧；这样不需要额外的 WebSocket 服务依赖，代码更短。

浏览器发送：

```http
POST /api/task
Content-Type: application/json

{"text":"向前走到箱子旁边"}
```

```http
POST /api/stop
```

浏览器轮询状态：

```http
GET /api/state
```

响应：

```json
{
  "type": "state",
  "skill": "walking",
  "status": "running",
  "arguments": {"vx": 0.3, "vy": 0.0, "wz": 0.0},
  "elapsed_s": 1.2
}
```

浏览器轮询相机帧：

```http
GET /api/frame.jpg
```

```http
GET /api/third_frame.jpg
```

`/api/frame` 是机器人头部相机图像，作为 VLM 的视觉输入；`/api/third_frame` 是跟踪机器人躯干的第三视角，仅供网页观察。网页使用 `/api/frame_stream` 和 `/api/third_stream` 的 MJPEG 流持续显示两路画面，拖动第三视角画面可旋转相机，滚轮可缩放。

技能事件包含在 `/api/state` 的 `current` 和 `history` 字段中：

```json
{
  "type": "skill_result",
  "skill": "walking",
  "result": "completed",
  "message": "duration reached"
}
```

相机帧以低频发送，建议先使用 2～5 FPS。浏览器不保存 API key，也不直接调用 GPT。

## API 配置

为了方便本地仿真，服务端从被 git 忽略的 `scripts/vlm_local_config.py` 读取配置：

```python
OPENAI_API_KEY = "..."
OPENAI_BASE_URL = "https://tokenhub.ilab.beer/v1"
OPENAI_MODEL = "gpt-6-astra"
```

该文件只由服务端读取，浏览器不会收到 API key。`scripts/vlm_local_config.py` 已加入 `.gitignore`，不能提交到仓库。

实现时应保留环境变量覆盖能力，便于远程运行或 CI：

```text
OPENAI_API_KEY
OPENAI_BASE_URL
OPENAI_MODEL
```

请求发送到：

```text
POST ${OPENAI_BASE_URL}/responses
```

API key 不写入 HTML、git、日志或 HTTP 响应。此前在对话中暴露的旧 key 必须先撤销并换新，再填入本地配置文件。

由于当前中转站使用 Responses API，API 适配器单独放在 `vlm_web.py` 的小函数中。仿真和技能执行不依赖具体 API SDK；如果中转站的 JSON 字段和官方 Responses API 有差异，只修改该适配器。

## 线程和时间模型

```text
MuJoCo/ONNX loop: 50 Hz
camera publish: 2～5 Hz
browser HTTP polling: asynchronous
GPT request: once per skill boundary
```

GPT 请求期间，当前 skill 不能因为网络阻塞而停止。技能完成后进入 `waiting_for_vlm` 状态，保持安全姿态或最后一个目标，直到下一次 GPT 响应到达；如果请求超时，则停止并向网页报告错误。

第一版只允许一个当前任务和一个仿真机器人，不实现多用户队列和多机器人管理。

## 实现阶段

### 阶段 1：复用官方运控

- 从 `infer_policy.py` 复制或最小化复用必要的加载逻辑；
- 加载默认 MuJoCo scene、BAM 和 walking/standing；
- 在无网页、无 GPT 的情况下运行固定 walking skill；
- 验证 61D observation 和 14D action。

### 阶段 2：技能执行器

- 加载全部七个官方 ONNX；
- 实现技能名到 ONNX 的映射；
- 实现 walking 的 `duration_s`，以及其他技能的本地完成条件；
- 实现 walking command 直通；
- 实现停止和超时事件；
- 不修改 `infer_policy.py`。

### 阶段 3：Fake VLM

- 使用固定 JSON 替代 GPT；
- 验证一个 skill 完成后进入下一次决策；
- 验证 `done`、非法 skill 和参数错误；
- 验证技能完成或超时后再次决策。

### 阶段 4：网页前端

- 提供文本输入和停止按钮；
- 显示相机图像；
- 显示当前 skill、参数和结果；
- 通过 HTTP 接口验证浏览器控制链路。

### 阶段 5：GPT-6-Astra

- 接入中转站 Responses API；
- 发送 RGB 图像、本体状态和历史摘要；
- 解析结构化 JSON；
- 对 API 超时、空响应和非法 JSON 做本地处理；
- 保存请求摘要和结果日志，但不保存 API key 和内部思考。

## 验证标准

实现完成后至少验证：

1. 不启动 VLM 入口时，原有 `infer_policy.py` 行为不变。
2. walking 的 command 直接传入官方策略，没有隐藏速度改写。
3. 同一 walking command 下，改变 duration 会改变执行时间。
4. 七个官方 ONNX 都能加载并产生正确 action 维度。
5. VLM 请求期间 MuJoCo 仍保持 50 Hz 控制。
6. GPT 返回 `done` 后动作停止。
7. 技能完成或超时后触发下一次 VLM 判断。
8. 浏览器可以提交文本并看到实时状态和相机画面。
9. 原有官方文件没有被修改。

## 非目标

第一版不实现：

- 真实机器人控制；
- 深度图输入；
- 任意 geom 碰撞分类；
- 多机器人或多用户；
- 自动速度补偿；
- walking 内部启动加速或反向 ramp；
- 复杂前端 3D 可视化；
- 让 GPT 直接输出 50 Hz action；
- 修改官方训练配置或重新训练策略。
