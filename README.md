# Elucidator A 股新闻因子研究脚手架

这是配套执行计划的最小可运行环境。目标不是直接给出“荐股结果”，而是先建立一个可审计、无前视泄漏、能做模拟盘验证的研究管线。

> [!IMPORTANT]
> 本项目仅用于研究和工程验证，不构成投资建议。任何进入模拟盘或实盘的策略都必须通过
> 数据授权、点时一致性、交易成本、风险控制和合规评审。

## 研究原则

- **点时一致**：特征只能使用决策时刻已经发布且可获得的数据。
- **严格隔离**：训练、验证、测试和回测按时间切分，二层模型只接收 OOF 预测。
- **数据可追溯**：记录来源、授权、采集时间、版本和处理链路。
- **结果可审计**：保存配置、随机种子、指标、成本假设和模型产物哈希。
- **先模拟后实盘**：没有稳定的样本外结果和模拟盘验收，不进入真实交易。

## 研究管线

```text
行情 / 基本面 / 新闻
        │
        ▼
点时数据合同与质量检查
        │
        ▼
Prophet 状态基线 + 共享 LSTM 时序表征 + LLM 新闻事件抽取
        │
        ▼
严格 OOF 特征与 XGBoost 二层排序
        │
        ▼
含成本回测 → 稳健性审计 → 模拟盘验收
```

核心组合：

- Prophet：只承担指数/行业状态、低频趋势等可解释基线；
- 全市场共享 LSTM：学习跨股票的时序表征，不为每只股票单独过拟合；
- XGBoost：使用严格 OOF 的基模型预测、量价/基本面和新闻事件因子做二层排序或回归；
- DeepSeek V4 Flash：通过服务端 API 把新闻抽取为结构化事件，模型地址和 ID 均来自环境变量；
- Qlib + Parquet/DuckDB：负责因子研究、数据版本和 A 股回测骨架。

完整方案见 [执行计划](docs/EXECUTION_PLAN.zh-CN.md)。

## 项目结构

```text
.
├── configs/research.yaml       # 研究参数与数据源声明
├── data/                       # 本地数据目录（默认不提交）
├── docs/                       # 执行计划与研究文档
├── src/elucidator/             # 配置、契约、LLM 客户端和预检逻辑
├── tests/                      # 契约与预检测试
├── .env.example                # 环境变量模板
├── pyproject.toml              # 包元数据与开发工具配置
└── requirements.lock.txt       # 锁定依赖快照
```

## 环境

项目固定 Python 3.12；当前机器的 Python 3.14 不用于该环境，以避免量化/科学计算包的兼容性问题。

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/elucidator-preflight
.venv/bin/pytest
```

配置 DeepSeek（仅服务端）：

```bash
cp .env.example .env
# 编辑 .env，填写 LLM_API_KEY；不要提交 .env
.venv/bin/elucidator-preflight --check-api
```

未提供行情/新闻授权和 API key 时，预检仍会验证所有本地依赖，但不会伪造或下载受许可约束的数据。生产或实盘之前必须补齐点时数据合同、券商成本参数、模拟盘验收和合规评审。

常用质量检查：

```bash
.venv/bin/ruff check .
.venv/bin/mypy
.venv/bin/pytest
```

## 配置与数据安全

- 复制 `.env.example` 创建本地 `.env`，真实 API key、Token 和数据源 DSN 不进入 Git。
- `data/raw`、`data/interim`、`data/processed`、`artifacts` 和 `mlruns` 默认仅保存在本地。
- 行情和新闻数据必须遵守供应商授权；仓库只保存数据契约、配置和可复现代码。
- LLM 只负责结构化新闻事件，不直接产生交易指令，且其输出需经过模式校验与缓存审计。

## Apple Silicon 运行提示

本机已分别完成 XGBoost、LightGBM、CatBoost、Torch 和 Prophet 的小样本拟合烟测。另复现到 macOS arm64 的本地 OpenMP 运行时冲突：若同一 Python 进程先加载 Torch，再拟合 XGBoost 3.3，进程可能崩溃；反向顺序或各自独立进程可正常运行。研究管线应把 LSTM 与树模型训练放在隔离 worker 中，不使用 `KMP_DUPLICATE_LIB_OK` 一类掩盖运行时冲突的变量。该类 macOS/ARM 多 OpenMP 运行时冲突亦见 [PyTorch 官方 issue](https://github.com/pytorch/pytorch/issues/149201)。
