# Embodied Navigation MVP

一个可运行的具身导航最小原型：用户用自然语言下达目标，系统完成目标解析、视觉驱动导航、到达反馈，并支持持续多轮交互。

本项目重点是“可以跑起来、结构可扩展、Windows 环境可降级”。

## 这个项目用了哪些

### 核心技术栈

- `Python`：整体逻辑与模块组织
- `Gradio`：Web 交互界面（聊天 + 画面 + 状态）
- `AI2-THOR`：真实室内场景模拟与动作执行
- `Pillow` / `NumPy`：图像与数值处理
- `requests`：调用 VLM API

### 感知与控制能力

- `RGB + Depth`：仅依赖视觉和深度信号做导航
- `VLM 检测`：可选，通过 `src/vlm_detector.py` 调 OpenAI 兼容接口
- `启发式视觉伺服`：目标居中与前进控制
- `多级恢复策略`：失目标扫描、绕行、强制脱困、防原地旋转

### 运行兼容策略

- 优先使用 `AI2-THOR`
- 在部分 Windows 环境 AI2-THOR 不可用时自动降级 `MockNavigator`
- 保持统一交互协议，不影响上层 Agent 与 UI

## 模块拆分（导航部分 vs Agent部分）

### 导航部分（Navigation）

负责“看见目标并移动到目标附近”。

- `src/thor_nav.py`
  - `ThorNavigator`：基于 AI2-THOR 的导航执行器
  - 支持目标检测、深度估距、动作决策、失败恢复
- `src/mock_nav.py`
  - `MockNavigator`：轻量回退模拟器
  - 在本地不可用 AI2-THOR 时保证 demo 仍可运行
- `src/vlm_detector.py`
  - `VLMDetector`：调用视觉语言模型返回目标 bbox
- `src/targets.py`
  - `normalize_target()`：目标词归一化（中英指令映射到统一标签）

### Agent部分（Dialogue / Orchestration）

负责“理解指令、组织导航流程、向用户反馈状态”。

- `app.py`
  - 负责整体编排：接收命令 -> 解析目标 -> 调度 navigator -> 流式回传状态
  - 对话管理：支持多轮交互、失败提示、到达确认
  - UI 侧把 Agent 对话、机器人视角、运行状态统一呈现

## 最后实现了哪些效果

- 支持自然语言导航指令（中文/英文）
- 支持目标类别：`sofa`、`bed`、`table`、`apple`
- 支持逐帧状态输出（导航中、扫描中、到达、超时）
- 到达后自动进入下一轮对话（“还需要什么？”）
- 在环境受限时自动切换 Mock 模式，确保端到端可演示

## 快速开始

### 1) 安装依赖

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### 2) 可选配置（启用 VLM 检测）

```bash
set VLM_API_KEY=your_api_key
set VLM_MODEL=gpt-4o-mini
```

可选环境变量：

- `VLM_API_URL`（默认 `https://api.openai.com/v1/chat/completions`）
- `VLM_TIMEOUT_S`（默认 `15`）
- `ALLOW_PRIVILEGED_FALLBACK=1`（允许 AI2-THOR 内部检测作为回退）

### 3) 启动 Web Demo

```bash
.\.venv\Scripts\python .\app.py
```

浏览器打开 Gradio 地址后可直接输入：

- `请到沙发旁边`
- `go to the bed`
- `navigate to table`

## 网站交付与登录说明（怎么登）

### 访问地址

- 本机访问：`http://127.0.0.1:7860` 或 `http://localhost:7860`
- 局域网访问：`http://<部署机器IP>:7860`
- 临时外网访问（Gradio 隧道）：设置 `GRADIO_SHARE=1` 后启动，终端会输出一个 `https://xxxx.gradio.live` 地址

### 登录方式（推荐交付时开启）

项目已支持 Web 端基础登录认证（在 `app.py` 中通过环境变量控制）：

```bash
set WEB_USERNAME=demo
set WEB_PASSWORD=demo123
.\.venv\Scripts\python .\app.py
```

打开网站后会先看到登录框，输入上面的用户名和密码即可进入。

### 不开启登录的情况

- 如果未设置 `WEB_USERNAME` 和 `WEB_PASSWORD`，网站默认不需要登录，打开地址即可使用。

### 一份可直接交付的“网站登录说明”模板

可直接发给验收方：

```text
系统名称：Embodied Navigation MVP
访问地址：http://<服务器IP>:7860
登录账号：demo
登录密码：demo123
备注：登录后在输入框中提交导航指令，例如“请到沙发旁边”。
```

## Demo（命令行版）

新增 `demo_cli.py`，用于不启动 Web 页面也能演示完整链路：

```bash
.\.venv\Scripts\python .\demo_cli.py
```

单次命令模式：

```bash
.\.venv\Scripts\python .\demo_cli.py --command "go to sofa"
```

强制使用 Mock 模式：

```bash
.\.venv\Scripts\python .\demo_cli.py --mock
```

## 项目结构

```text
.
├─ app.py
├─ demo_cli.py
├─ requirements.txt
├─ README.md
└─ src/
   ├─ __init__.py
   ├─ mock_nav.py
   ├─ targets.py
   ├─ thor_nav.py
   └─ vlm_detector.py
```

## 当前边界

- 当前是启发式局部视觉伺服，不是全局语义 SLAM/最优路径规划
- 目标解析目前为规则映射，可继续升级为 LLM parser
- 导航成功率受场景遮挡、初始朝向和目标可见性影响

## 后续可扩展方向

- 引入语义地图与 frontier 探索
- 增加跨场景评测脚本（成功率、步数、耗时）
- 引入更强检测/跟踪与不确定性估计机制
