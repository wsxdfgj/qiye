# 文本摘要工具

本项目是一个基于深度学习的文本摘要工具，提供一个交互式的 Web 界面，支持两种摘要生成模式：

1.  **本地摘要模型**：使用在特定领域数据上微调的 BART 模型，快速生成摘要。
2.  **大语言模型（LLM）**：通过 API 调用外部的大语言模型（如 OpenAI GPT 系列），提供更高质量的摘要。

用户可以通过界面直接输入文本、上传文本文件，并灵活调整本地模型的生成参数。

## ✨ 功能特性

- **双模摘要**：同时集成自训练的本地模型和强大的外部 LLM，满足不同场景的需求。
- **交互式界面**：基于 Streamlit 构建，界面简洁直观，易于使用。
- **灵活的输入方式**：支持直接粘贴文本或上传 `.txt` 文件。
- **参数可调**：允许用户实时调整本地模型的最大/最小输出长度、Beam Size 等核心参数。
- **环境隔离**- 使用 Conda 进行环境管理，确保依赖的稳定性和一致性。
- **GPU 加速**- 自动检测并使用可用的 GPU 资源，加速本地模型推理。

## ⚙️ 环境安装

本项目使用 [Conda](https://docs.conda.io/en/latest/miniconda.html) 进行环境管理。请按照以下步骤安装并配置环境：

1.  **克隆仓库**

    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2.  **创建并激活 Conda 环境**

    使用项目根目录下的 `environment.yml` 文件创建 Conda 环境。该文件包含了所有必需的依赖项。

    ```bash
    conda env create -f environment.yml
    ```

    该命令会自动创建一个名为 `qiye_work` 的 Conda 环境，并安装所有指定的包。

3.  **激活环境**

    ```bash
    conda activate qiye_work
    ```
    *注意：后续所有操作都应在此激活的环境中执行。*

## 🚀 快速开始

1.  **下载本地模型**

    应用依赖一个预训练的 BART 模型。请确保模型文件已放置在 `./results/checkpoint-8500` 目录下。如果该模型未包含在仓库中，请从指定来源下载并解压到该位置。

2.  **配置大语言模型（可选）**

    如果需要使用大语言模型摘要功能，请在项目根目录下创建一个 `.env` 文件，并配置以下环境变量：

    ```env
    # .env 文件示例
    CUSTOM_MODEL="gpt-3.5-turbo"
    API_BASE="https://api.openai.com/v1"
    CUSTOM_API_KEY="sk-your-openai-api-key"
    ```

3.  **启动 Streamlit 应用**

    在项目根目录下，运行以下命令：

    ```bash
    streamlit run app.py
    ```

    应用启动后，浏览器会自动打开一个新的标签页，地址通常为 `http://localhost:8501`。

## 📂 项目结构

```
.
├── app.py                  # Streamlit Web 应用入口
├── environment.yml         # Conda 环境配置文件
├── README.md               # 项目说明文档
├── results/
│   └── checkpoint-8500/    # 存放本地 BART 模型文件
├── src/                    # 源代码目录
│   ├── train.py            # 模型训练脚本
│   ├── infer.py            # 推理脚本
│   ├── evaluate_model.py   # 模型评估脚本
│   └── ...
└── utils/                  # 工具函数目录
    └── invoke_llm.py       # 调用大语言模型的封装函数
```

