# 水用量分析报告系统

这个系统用于分析用户的水用量数据，检测可能的漏水情况，并生成分析报告。

## 主要功能

1. **数据获取**：支持从本地数据库或MCP服务器获取数据
2. **漏水检测**：基于三种条件判断漏水可能性
   - 近期出现两次漏水情况（日均用水>3方）
   - 单次漏水超过5方
   - 最近连续出现超过历史平均用水量一倍的情况
3. **报告生成**：生成Markdown格式的详细分析报告
4. **数据存储**：将分析结果保存到数据库中，方便后续查询

## 数据库存储

系统会将分析结果存储在SQLite数据库中，包含以下信息：

- **账户ID**：水表账户的唯一标识
- **漏水可能性**：无、低、中、高
- **详细数据**：包括最大用水量、平均用水量、日均用水量等
- **异常分析**：记录异常用水情况和漏水风险描述
- **完整报告**：保存完整的分析报告文本

## 使用方法

### 运行分析

```bash
python src/main.py
```

此命令将分析所有账户的用水数据，生成报告并存储到数据库中。

### 查询结果

```bash
# 显示统计信息
python src/query_results.py stats

# 查询特定账户
python src/query_results.py query --account 2226970

# 查询有风险的账户
python src/query_results.py query --risk

# 按漏水可能性查询
python src/query_results.py query --probability 高

# 导出结果为CSV格式
python src/query_results.py query --risk --export results.csv
```

## 配置MCP服务器

如需使用MCP服务器，请在`.env`文件中配置以下参数：

```
USE_MCP_SERVER=true
MCP_SERVER_HOST=您的MCP服务器地址
MCP_SERVER_PORT=5000
MCP_SERVER_PATH=/api/data
MCP_AUTH_TOKEN=您的认证令牌
```

## 数据库表结构

分析结果存储在`data/analysis_results.db`文件中，表结构如下：

| 字段名            | 类型      | 描述                       |
|------------------|-----------|----------------------------|
| id               | INTEGER   | 自增主键                   |
| account_id       | TEXT      | 账户ID                     |
| analysis_date    | TEXT      | 分析日期                   |
| leak_probability | TEXT      | 漏水可能性(无/低/中/高)    |
| risk_warning     | INTEGER   | 是否有风险警告(0/1)        |
| max_usage        | REAL      | 最大用水量                 |
| max_usage_date   | TEXT      | 最大用水量日期             |
| avg_usage        | REAL      | 平均用水量                 |
| max_daily_avg    | REAL      | 最大日均用水量             |
| historical_avg   | REAL      | 历史日均用水量             |
| increase_ratio   | REAL      | 当前与历史用水量比率       |
| abnormal_usage   | TEXT      | 异常用水描述               |
| leak_risk_text   | TEXT      | 漏水风险描述               |
| full_report      | TEXT      | 完整分析报告               |
| data_source      | TEXT      | 数据来源(local_db/mcp_server) |
