# PINN Agent V2

一个面向 PINN（Physics-Informed Neural Networks）科研与代码生成场景的多智能体系统。

`PINN Agent V2` 将文献检索、科研方案设计、代码生成、沙盒执行、质量审查、记忆管理与评测体系整合到一个可交互的 TUI 应用中，目标是让 Agent 不只是“聊天”，而是能按照明确 SOP 完成科研代码任务。

---

## 项目亮点

- **Multi-Agent 协作**：基于 LangGraph 将任务拆分为 `Researcher / Coder / Examiner` 三类 Agent。
- **显式 SOP 工作流**：支持 `需求输入 -> 意图识别 -> 记忆读取 -> 研究/编码 -> 审查重试 -> 汇总 -> 记忆写回`。
- **PINN 科研场景聚焦**：围绕 PINN 文献综述、损失函数设计、最小代码示例、训练日志导出等任务构建。
- **安全代码执行**：使用 Docker 沙盒运行模型生成的 Python 代码，支持网络隔离、CPU/内存限制、超时终止与产物导出。
- **长短期记忆系统**：支持 session 级上下文压缩、项目长期记忆与历史经验复用。
- **TUI 可视化交互**：使用 Textual 构建终端界面，展示 Agent Flow、Debug Trace、Artifacts、Session Memory 与模型信息。
- **工程化评测闭环**：支持 mock/live 双模式 eval、规则评分、LLM-as-Judge、latency/token/retry/reliability 指标面板。
- **完整测试体系**：包含 unit、integration、workflow 与 smoke test。

---

## 系统架构

```text
┌─────────────────────────────────────────────┐
│  Textual TUI / CLI                          │
├─────────────────────────────────────────────┤
│  LangGraph Workflow                         │
│  parse_intent -> memory_read -> agents      │
│  -> examiner -> synthesize -> writeback     │
├─────────────────────────────────────────────┤
│  Agents                                     │
│  Researcher | Coder | Examiner              │
├─────────────────────────────────────────────┤
│  Tools                                      │
│  RAG | arXiv/Web Search | Formula | Code    │
├─────────────────────────────────────────────┤
│  Sandbox                                    │
│  Docker execution + artifact export         │
├─────────────────────────────────────────────┤
│  Memory                                     │
│  Session Summary | Project Memory |         │
│  Experience Store                           │
├─────────────────────────────────────────────┤
│  Observability & Eval                       │
│  Trace | Cost | Rubrics | LLM-as-Judge      │
└─────────────────────────────────────────────┘
```

核心工作流由 [orchestrator/graph.py](orchestrator/graph.py) 定义，状态结构由 [orchestrator/state.py](orchestrator/state.py) 统一管理。

---

## Multi-Agent 设计

### Researcher Agent

文件：[agents/researcher.py](agents/researcher.py)

职责：

- 检索本地论文库、arXiv 与 Web 资料
- 生成 PINN 相关文献综述
- 在 full pipeline 任务中给出技术方案设计
- 尽量通过引用与工具检索降低幻觉

### Coder Agent

文件：[agents/coder.py](agents/coder.py)

职责：

- 根据用户需求和研究方案生成 Python / PyTorch 代码
- 调用沙盒执行代码
- 解析 stdout / stderr
- 失败后根据错误信息自动修复
- 保存最终代码和运行产物

### Examiner Agent

文件：[agents/examiner.py](agents/examiner.py)

职责：

- 对文献内容进行学术审查
- 对生成代码进行安全性与可执行性审查
- 输出 `PASS / FAIL`
- 触发 LangGraph 中的自动重试链路

---

## 目录结构

```text
.
├── agents/                 # Researcher / Coder / Examiner
├── eval/                   # 固定评测集、规则评分、LLM-as-Judge、报告生成
├── memory/                 # session memory / project memory / experience memory
├── observability/          # trace 与 token cost 追踪
├── orchestrator/           # LangGraph 状态、路由与工作流图
├── rag/                    # 本地论文知识库构建与 reranker
├── sandbox/                # Docker 沙盒执行器
├── tests/                  # unit / integration / workflow / smoke tests
├── tools/                  # RAG、搜索、公式、代码执行工具
├── tui/                    # Textual TUI
├── config.py               # 统一配置中心
├── main.py                 # 项目入口
├── TECHNICAL_REPORT.md     # 详细技术报告与源码学习路线
└── memory.md               # 项目开发记忆与阶段复盘
```

---

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd PINN_AGENT_PROJECT_V2
```

### 2. 创建 Python 环境

推荐使用 Conda：

```bash
conda create -n mcp_agent python=3.10 -y
conda activate mcp_agent
```

也可以使用 venv：

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

方式一：使用 requirements：

```bash
python -m pip install -r requirements.txt
```

方式二：使用分阶段安装脚本：

```bash
bash install.sh
```

安装后可运行验证脚本：

```bash
python verify_install.py
```

### 4. 配置环境变量

复制示例配置：

```bash
cp .env.example .env
```

如果使用本地 Ollama：

```env
PINN_AGENT_LLM_PROVIDER=ollama
PINN_AGENT_OLLAMA_BASE_URL=http://localhost:11434/v1
PINN_AGENT_OLLAMA_API_KEY=ollama-local
PINN_AGENT_LOCAL_MODEL_CODER=qwen2.5:7b
```

如果使用 Qwen / DashScope OpenAI-Compatible API：

```env
PINN_AGENT_LLM_PROVIDER=qwen
PINN_AGENT_QWEN_API_KEY=your_api_key
PINN_AGENT_QWEN_MODEL_RESEARCHER=qwen-plus
PINN_AGENT_QWEN_MODEL_CODER=qwen3-coder-plus
PINN_AGENT_QWEN_MODEL_EXAMINER=qwen-plus
```

注意：`.env` 已被 `.gitignore` 忽略，请不要把 API Key 提交到 GitHub。

### 5. 构建 Docker 沙盒镜像

代码执行依赖 Docker 沙盒：

```bash
docker build -t pinn_agent_sandbox:latest -f sandbox/Dockerfile.sandbox .
```

如果你暂时不需要真实执行代码，可以先运行 mock eval 和单元测试。

### 6. 构建本地 RAG 知识库

如果你希望 Researcher Agent 使用本地论文库：

```bash
python -m rag.build_memory
```

论文 PDF 默认放在 `papers/` 目录，向量库默认写入 `rag/chroma_db/`。

---

## 运行方式

### 启动 TUI

```bash
python main.py
```

启动 Debug 模式：

```bash
python main.py --debug
```

### CLI 交互模式

```bash
python main.py --cli
```

### 单次查询模式

```bash
python main.py --query "写一个最小 PINN Python 示例并运行，损失项包含物理损失"
```

---

## 评测系统

项目内置一套固定评测体系，适合做回归测试、demo 验证和工程指标分析。

### Mock 模式评测

不依赖真实 LLM / Docker，适合快速回归：

```bash
python -m eval.runner --mode mock --judge-mode heuristic
```

### Live 模式评测

调用真实模型与真实链路：

```bash
python -m eval.runner --mode live --judge-mode llm --case-id code_minimal_pinn
```

### 评测产物

每次评测会生成：

```text
eval/runs/run_*/results.jsonl
eval/runs/run_*/metrics.json
eval/runs/run_*/summary.md
```

`summary.md` 会展示：

- Rule score / Judge score / Overall score
- Judge coverage rate
- Avg / P95 latency
- Token cost
- Retry rate
- Execution success rate
- Memory writeback rate
- Slowest cases
- Most expensive cases

---

## 测试

安装开发依赖：

```bash
python -m pip install -r requirements-dev.txt
```

运行核心测试：

```bash
python -m pytest tests/unit tests/integration tests/workflows -q
```

运行 smoke test：

```bash
python tests/smoke_test_phase2.py
```

当前测试分层：

- `tests/unit/`：纯逻辑单元测试
- `tests/integration/`：跨模块契约测试
- `tests/workflows/`：主工作流回归测试
- `tests/smoke_test_phase2.py`：环境与核心模块健康检查

---

## 记忆系统

项目实现了三类记忆：

### Session Memory

文件：[memory/session_manager.py](memory/session_manager.py)

保存当前会话的：

- 最近 query
- 最近代码摘要
- 最近错误摘要
- 最近产物
- 上下文压缩摘要
- 成功 / 失败代码片段

### Project Memory

文件：[memory/project_store.py](memory/project_store.py)

保存项目长期事实：

- 项目目标
- 架构规则
- 技术栈
- 当前优先级
- 已接受 / 拒绝的设计决策

### Experience Memory

文件：[memory/experience_store.py](memory/experience_store.py)

保存历史任务经验：

- 错误类型
- 症状
- 解决建议
- 成功 / 失败统计
- 可复用 artifact

---

## 安全设计

代码执行链路的核心安全机制：

- Docker 容器隔离
- `--network none` 禁用网络
- CPU / 内存限制
- 运行超时控制
- 非 root 用户执行
- Shell 命令白名单
- 文件读写路径限制
- 运行期产物单独导出

相关文件：

- [tools/code_tools.py](tools/code_tools.py)
- [sandbox/docker_runner.py](sandbox/docker_runner.py)

---


## 常见问题

### 1. 为什么启动时卡在质量审查阶段？

可能原因：

- Examiner 正在调用真实 LLM 做深度审查
- 本地模型响应慢
- `.env` 中模型配置不可用

可以先使用 mock eval 验证工程链路：

```bash
python -m eval.runner --mode mock --judge-mode heuristic
```

或者关闭成功代码的深度审查：

```env
PINN_AGENT_EXAMINER_DEEP_CODE_REVIEW_ON_SUCCESS=false
```

### 2. 为什么 Docker 沙盒不可用？

请确认：

- Docker Desktop / Docker Engine 已启动
- 已构建镜像 `pinn_agent_sandbox:latest`

构建命令：

```bash
docker build -t pinn_agent_sandbox:latest -f sandbox/Dockerfile.sandbox .
```

### 3. 为什么 RAG 构建失败？

常见原因：

- `torch` 环境不可用
- Hugging Face 模型未缓存且当前环境不能联网
- `papers/` 中 PDF 解析失败

可以先运行：

```bash
python verify_install.py
```

再运行：

```bash
python -m rag.build_memory
```

### 4. 我没有真实 LLM API，能体验项目吗？

可以。你可以先运行：

```bash
python -m eval.runner --mode mock --judge-mode heuristic
```

这会验证主要工程链路、评测报告和指标面板，不依赖真实模型。

---

## 路线图

后续可以继续增强：

- 在 TUI 中集成 Eval / Cost / Reliability 实时面板
- 增强 RAG 检索质量评估
- 增加更多 PINN / SciML 固定 benchmark case
- 支持更多模型提供方
- 增强 experience memory 的语义检索能力
- 增加更细粒度的 token budget 和 latency budget 告警

---

## 贡献

欢迎提交 Issue 或 Pull Request。建议贡献方向：

- 新增 PINN / SciML 评测 case
- 改进 Docker 沙盒安全策略
- 优化 TUI 展示
- 扩展 RAG 检索与 rerank 策略
- 增强 LLM-as-Judge 评分维度
- 补充测试覆盖

提交 PR 前建议运行：

```bash
python -m pytest tests/unit tests/integration tests/workflows -q
python -m eval.runner --mode mock --judge-mode heuristic
```

---

## 免责声明

本项目主要用于科研辅助、工程学习与 Agent 应用开发实践。  
模型生成的学术内容与代码结果仍需人工审查，尤其是在正式科研、论文写作或生产环境使用前，应进行独立验证。

---

## License

如果你计划开源，建议在仓库中补充 `LICENSE` 文件。  
常见选择：

- MIT License：宽松，适合社区使用和二次开发
- Apache-2.0：包含更明确的专利授权条款

