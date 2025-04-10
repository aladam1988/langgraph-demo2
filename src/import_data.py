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
    CREATE TABLE IF NOT EXISTS meter_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_id TEXT NOT NULL,
        reading_time TEXT NOT NULL,
        current_reading REAL NOT NULL,
        current_usage REAL NOT NULL,
        average_usage REAL NOT NULL,
        daily_average REAL NOT NULL,
        daily_max REAL NOT NULL,
        daily_min REAL NOT NULL,
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
    lines = data_text.strip().split('\n')
    headers = lines[0].split('\t')
    data = []
    
    for line in lines[1:]:
        values = line.split('\t')
        if len(values) == len(headers):
            data.append(values)
    
    df = pd.DataFrame(data, columns=headers)
    
    # 连接数据库
    conn = sqlite3.connect(db_path)
    
    # 导入数据
    df.to_sql('meter_readings', conn, if_exists='append', index=False)
    
    # 获取导入的记录数
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM meter_readings")
    count = cursor.fetchone()[0]
    
    conn.close()
    
    logger.info(f"成功导入 {count} 条记录到数据库")
    return count

def main():
    """主函数"""
    # 创建数据库
    db_path = create_database()
    
    # 示例数据
    data_text = """account_id	reading_time	current_reading	current_usage	average_usage	daily_average	daily_max	daily_min
4091972	2025/3/27	74.99	1.56	1.1	1.56	1.56	1.56
4091972	2025/3/28	76.03	1.04	1	1.04	1.04	1.04
4091972	2025/3/29	76.81	0.78	0.9	0.78	0.78	0.78
4091972	2025/3/30	77.42	0.61	1	0.61	0.61	0.61
4091972	2025/3/31	77.9	0.48	1	0.48	0.48	0.48
4091972	2025/4/1	78.26	0.36	1.1	0.36	0.36	0.36
4091972	2025/4/2	79.56	1.3	1.1	1.3	1.3	1.3
4091972	2025/4/3	80.71	1.15	1.1	1.15	1.15	1.15
4091972	2025/4/4	82	1.29	1.4	1.29	1.29	1.29
4091972	2025/4/5	82.8	0.8	1.4	0.8	0.8	0.8
4091972	2025/4/6	83.49	0.69	1.4	0.69	0.69	0.69
4091972	2025/4/7	89.24	5.75	1.4	5.75	5.75	5.75
2226970	2025/3/27	197	1	0.3	1	1	1
2226970	2025/3/28	197	0	0.3	0	0	0
2226970	2025/3/29	197	0	0.3	0	0	0
2226970	2025/3/30	197	0	0.3	0	0	0
2226970	2025/3/31	198	1	0.3	1	1	1
2226970	2025/4/1	198	0	0.3	0	0	0
2226970	2025/4/2	198	0	0.5	0	0	0
2226970	2025/4/3	198	0	0.6	0	0	0
2226970	2025/4/4	199	1	0.7	1	1	1
2226970	2025/4/5	201	2	0.7	2	2	2
2226970	2025/4/6	202	1	0.7	1	1	1
2226970	2025/4/7	204	2	0.7	2	2	2
4139756	2025/3/27	407	1	0.3	1	1	1
4139756	2025/3/28	407	0	0.3	0	0	0
4139756	2025/3/29	407	0	0.3	0	0	0
4139756	2025/3/30	407	0	0.3	0	0	0
4139756	2025/3/31	408	1	0.3	1	1	1
4139756	2025/4/1	408	0	0.3	0	0	0
4139756	2025/4/2	408	0	0.4	0	0	0
4139756	2025/4/3	408	0	0.6	0	0	0
4139756	2025/4/4	409	1	0.7	1	1	1
4139756	2025/4/5	410	1	0.7	1	1	1
4139756	2025/4/6	413	3	0.7	3	3	3
4139756	2025/4/7	414	1	0.7	1	1	1
2284662	2025/3/27	149	0	0.3	0	0	0
2284662	2025/3/28	149	0	0.3	0	0	0
2284662	2025/3/29	149	0	0.3	0	0	0
2284662	2025/3/30	149	0	0.3	0	0	0
2284662	2025/3/31	149	0	0.3	0	0	0
2284662	2025/4/1	149	0	0.3	0	0	0
2284662	2025/4/2	150	1	0.3	1	1	1
2284662	2025/4/3	150	0	0.7	0	0	0
2284662	2025/4/4	150	0	0.9	0	0	0
2284662	2025/4/5	150	0	0.9	0	0	0
2284662	2025/4/6	156	6	0.9	6	6	6
2284662	2025/4/7	158	2	0.9	2	2	2
4148081	2025/3/27	3.24	0.29	0.2	0.29	0.29	0.29
4148081	2025/3/28	3.37	0.13	0.2	0.13	0.13	0.13
4148081	2025/3/29	3.52	0.15	0.2	0.15	0.15	0.15
4148081	2025/3/30	3.65	0.13	0.3	0.13	0.13	0.13
4148081	2025/3/31	3.78	0.13	0.3	0.13	0.13	0.13
4148081	2025/4/1	3.99	0.21	0.4	0.21	0.21	0.21
4148081	2025/4/2	4.68	0.69	0.4	0.69	0.69	0.69
4148081	2025/4/3	5.49	0.81	0.5	0.81	0.81	0.81
4148081	2025/4/4	6.32	0.83	0.5	0.83	0.83	0.83
4148081	2025/4/5	7.35	1.03	0.5	1.03	1.03	1.03
4148081	2025/4/6	8.44	1.09	0.5	1.09	1.09	1.09
4148081	2025/4/7	9.51	1.07	0.5	1.07	1.07	1.07
4167112	2025/3/27	184	0	0.2	0	0	0
4167112	2025/3/28	184	0	0.2	0	0	0
4167112	2025/3/29	184	0	0.2	0	0	0
4167112	2025/3/30	184	0	0.2	0	0	0
4167112	2025/3/31	184	0	0.2	0	0	0
4167112	2025/4/1	184	0	0.3	0	0	0
4167112	2025/4/2	184	0	0.3	0	0	0
4167112	2025/4/3	184	0	0.6	0	0	0
4167112	2025/4/4	185	1	0.8	1	1	1
4167112	2025/4/5	185	0	0.8	0	0	0
4167112	2025/4/6	189	4	0.8	4	4	4
4167112	2025/4/7	192	3	0.8	3	3	3"""
    
    # 导入数据
    count = import_data(db_path, data_text)
    
    print(f"成功导入 {count} 条用水数据记录到数据库: {db_path}")

if __name__ == "__main__":
    main() 