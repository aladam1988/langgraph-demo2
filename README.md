# 大规模账号数据处理系统

这是一个使用 LangGraph 构建的智能体系统，用于处理大规模账号数据。系统可以同时处理多个账号，每个账号包含两个不同数据库的数据。

## 功能特点

- 支持多种数据库类型（SQL和MongoDB）
- 使用LangGraph构建的智能体系统
- 并发处理多个账号
- 完善的错误处理和日志记录
- 可配置的处理参数

## 安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd langgraph-demo
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
```bash
cp .env.example .env
# 编辑 .env 文件，填入实际的配置信息
```

## 使用方法

1. 准备账号列表：
创建一个 `account_ids.txt` 文件，每行一个账号ID。

2. 运行程序：
```bash
python src/main.py
```

## 系统架构

- `src/data_access.py`: 数据访问层，处理数据库连接和数据获取
- `src/agents.py`: 智能体系统，使用LangGraph构建的工作流
- `src/main.py`: 主程序，协调整个处理流程

## 配置说明

在 `.env` 文件中配置以下参数：

- 数据库连接信息
- MongoDB配置
- LLM模型配置
- 并发处理参数

## 日志

系统使用 loguru 进行日志记录，日志文件保存在 `processing.log` 中。

## 注意事项

1. 确保有足够的系统资源处理大量账号
2. 根据实际需求调整并发处理数量
3. 定期检查日志文件大小
4. 确保数据库连接信息正确 