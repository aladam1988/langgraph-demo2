import os
import sqlite3
import pandas as pd
from loguru import logger

def create_database():
    """创建数据库和表结构"""
    # 获取项目根目录路径
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(root_dir, "data", "account_data.db")
    
    # 确保数据目录存在
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # 连接数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS historical_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_id TEXT NOT NULL,
        reading_time TEXT NOT NULL,
        reading_value INTEGER NOT NULL,
        UNIQUE(account_id, reading_time)
    )
    ''')
    
    conn.commit()
    conn.close()
    
    logger.info(f"数据库已创建: {db_path}")
    return db_path

def import_data(db_path, data_text):
    """导入数据到数据库"""
    # 将文本数据转换为DataFrame
    lines = [line.strip() for line in data_text.strip().split('\n')]
    data = []
    
    for line in lines:
        account_id, reading_time, reading_value = line.split('\t')
        data.append({
            'account_id': account_id,
            'reading_time': reading_time,
            'reading_value': int(reading_value)
        })
    
    df = pd.DataFrame(data)
    
    # 连接数据库
    conn = sqlite3.connect(db_path)
    
    try:
        # 导入数据
        df.to_sql('historical_readings', conn, if_exists='append', index=False)
        
        # 获取导入的记录数
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM historical_readings")
        count = cursor.fetchone()[0]
        
        logger.info(f"成功导入 {count} 条记录到数据库")
        return count
        
    except sqlite3.IntegrityError:
        logger.warning("部分数据已存在，尝试逐条插入...")
        
        # 如果批量导入失败，尝试逐条插入
        cursor = conn.cursor()
        inserted = 0
        
        for _, row in df.iterrows():
            try:
                cursor.execute(
                    "INSERT INTO historical_readings (account_id, reading_time, reading_value) VALUES (?, ?, ?)",
                    (row['account_id'], row['reading_time'], row['reading_value'])
                )
                inserted += 1
            except sqlite3.IntegrityError:
                logger.debug(f"记录已存在: {row['account_id']} - {row['reading_time']}")
                continue
        
        conn.commit()
        logger.info(f"成功导入 {inserted} 条新记录到数据库")
        return inserted
        
    finally:
        conn.close()

def main():
    """主函数"""
    # 创建数据库
    db_path = create_database()
    
    # 历史数据
    data_text = """4091972	2025/3/3	26
4091972	2025/1/15	36
4091972	2024/11/14	43
4091972	2024/9/11	70
4091972	2024/7/11	47
4091972	2024/5/21	0
2226970	2025/4/5	9
2226970	2025/2/15	10
2226970	2024/12/15	13
2226970	2024/10/15	12
2226970	2024/8/15	13
2226970	2024/6/15	16
4139756	2025/3/10	8
4139756	2025/2/6	19
4139756	2024/11/15	15
4139756	2024/9/15	15
4139756	2024/7/15	12
4139756	2024/5/15	12
4139756	2024/3/15	12
2284662	2025/4/8	15
2284662	2025/2/8	7
2284662	2024/12/8	7
2284662	2024/10/8	6
2284662	2024/8/8	6
2284662	2024/6/8	7
4148081	2025/3/16	0
4148081	2025/2/11	1
4148081	2024/12/18	3
4148081	2024/10/4	1
4148081	2024/8/1	2
4148081	2024/6/1	1
4167112	2025/3/11	2
4167112	2025/2/6	5
4167112	2024/11/15	6
4167112	2024/9/15	4
4167112	2024/7/15	4
4167112	2024/5/15	4"""
    
    # 导入数据
    count = import_data(db_path, data_text)
    
    print(f"成功导入 {count} 条历史用水数据记录到数据库: {db_path}")

if __name__ == "__main__":
    main() 