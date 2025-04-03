import sqlite3
import os
import json
import uuid
from datetime import datetime

def main():
    """直接生成SQLite数据库文件"""
    # 数据库路径
    db_path = "account_data.db"
    
    # 如果数据库已存在，先删除
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # 创建数据库连接
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建表结构
    print("创建数据库表...")
    
    # 创建账户表
    cursor.execute('''
    CREATE TABLE accounts (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'active',
        metadata TEXT
    )
    ''')
    
    # 创建交易表
    cursor.execute('''
    CREATE TABLE transactions (
        id TEXT PRIMARY KEY,
        account_id TEXT NOT NULL,
        amount REAL NOT NULL,
        type TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        description TEXT,
        metadata TEXT,
        FOREIGN KEY (account_id) REFERENCES accounts(id)
    )
    ''')
    
    # 创建分析结果表
    cursor.execute('''
    CREATE TABLE analysis_results (
        id TEXT PRIMARY KEY,
        account_id TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        result_type TEXT NOT NULL,
        result_data TEXT NOT NULL,
        metadata TEXT,
        FOREIGN KEY (account_id) REFERENCES accounts(id)
    )
    ''')
    
    # 插入账户数据
    account_id = "1"
    print(f"创建账户 (ID: {account_id})...")
    
    cursor.execute('''
    INSERT INTO accounts (id, name, status, metadata, created_at, updated_at)
    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    ''', (
        account_id,
        "时间序列测试账户",
        "active",
        json.dumps({
            "description": "包含时间序列数据的测试账户",
            "created_by": "generate_db.py"
        })
    ))
    
    # 插入交易数据
    print("插入交易数据...")
    
    # 数据格式: 日期, 指标1, 指标2, 指标3, 指标4, 指标5, 指标6
    data = [
        ["2025-01-01", 1, 0, 1.6, 0, 0, 0],
        ["2025-01-11", 1, 0, 1.6, 0, 0, 0],
        ["2025-01-21", 1, 0, 1.6, 0, 0, 0],
        ["2025-02-01", 1, 0, 1.6, 0, 0, 0],
        ["2025-02-11", 1, 0, 2, 0, 0, 0],
        ["2025-02-21", 1, 0, 2.4, 0, 0, 0],
        ["2025-03-01", 1, 0, 3.4, 0, 0, 0],
        ["2025-03-11", 6, 5, 3.4, 5, 5, 5],
        ["2025-03-21", 12, 6, 3.4, 6, 6, 6],
        ["2025-04-01", 24, 12, 3.4, 12, 12, 12]
    ]
    
    for entry in data:
        date_str, metric1, metric2, metric3, metric4, metric5, metric6 = entry
        
        # 将日期字符串转换为时间戳
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        timestamp = date_obj.timestamp()
        
        # 插入交易记录
        cursor.execute('''
        INSERT INTO transactions (id, account_id, amount, type, description, timestamp, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()),
            account_id,
            metric1,  # 使用指标1作为交易金额
            "data_entry",
            f"数据记录 - {date_str}",
            timestamp,
            json.dumps({
                "date": date_str,
                "metric1": metric1,
                "metric2": metric2,
                "metric3": metric3,
                "metric4": metric4,
                "metric5": metric5,
                "metric6": metric6
            })
        ))
    
    # 插入分析结果
    print("插入初始分析结果...")
    
    cursor.execute('''
    INSERT INTO analysis_results (id, account_id, result_type, result_data, metadata)
    VALUES (?, ?, ?, ?, ?)
    ''', (
        str(uuid.uuid4()),
        account_id,
        "initial_assessment",
        "这是账户1的初始数据分析。数据显示在2025年3月开始有显著变化，多个指标从0上升到较高值。",
        json.dumps({
            "created_at": datetime.now().timestamp(),
            "data_points": len(data),
            "date_range": f"{data[0][0]} 至 {data[-1][0]}"
        })
    ))
    
    # 提交更改并关闭连接
    conn.commit()
    conn.close()
    
    print(f"\n✅ 数据库文件 '{db_path}' 已成功创建！")
    print(f"数据库包含:")
    print(f"- 1个账户记录 (ID: {account_id})")
    print(f"- {len(data)}个交易记录")
    print(f"- 1个分析结果")
    print("\n你可以使用SQLite浏览器或其他工具打开此数据库文件。")

if __name__ == "__main__":
    main() 