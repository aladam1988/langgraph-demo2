import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

class WaterMeterDataReader:
    """水表数据读取和分析类"""
    
    def __init__(self, db_path="water_meter_data.db"):
        """初始化数据库连接"""
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"数据库文件 '{db_path}' 不存在!")
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # 使查询结果可以通过列名访问
        
        print(f"已连接到数据库: {db_path}")
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            print("数据库连接已关闭")
    
    def get_all_accounts(self):
        """获取所有账户信息"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM accounts ORDER BY id")
        accounts = cursor.fetchall()
        
        print(f"找到 {len(accounts)} 个账户:")
        for account in accounts:
            print(f"ID: {account['id']}, 名称: {account['name']}, 状态: {account['status']}")
        
        return accounts
    
    def get_account_readings(self, account_id):
        """获取指定账户的所有水表读数"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM meter_readings 
            WHERE account_id = ? 
            ORDER BY reading_time
        """, (account_id,))
        
        readings = cursor.fetchall()
        print(f"账户 {account_id} 有 {len(readings)} 条水表读数记录")
        
        return readings
    
    def get_readings_as_dataframe(self, account_id=None):
        """将水表读数转换为pandas DataFrame"""
        cursor = self.conn.cursor()
        
        if account_id:
            cursor.execute("""
                SELECT * FROM meter_readings 
                WHERE account_id = ? 
                ORDER BY reading_time
            """, (account_id,))
        else:
            cursor.execute("""
                SELECT * FROM meter_readings 
                ORDER BY account_id, reading_time
            """)
        
        # 获取列名
        columns = [description[0] for description in cursor.description]
        
        # 获取数据
        data = cursor.fetchall()
        
        # 创建DataFrame
        df = pd.DataFrame([dict(row) for row in data])
        
        # 将reading_time转换为日期类型
        if 'reading_time' in df.columns:
            df['reading_time'] = pd.to_datetime(df['reading_time'])
        
        return df
    
    def plot_water_usage(self, account_id=None, start_date=None, end_date=None):
        """绘制水表用量图表"""
        df = self.get_readings_as_dataframe(account_id)
        
        if df.empty:
            print("没有找到数据!")
            return
        
        # 过滤日期范围
        if start_date:
            df = df[df['reading_time'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['reading_time'] <= pd.to_datetime(end_date)]
        
        # 如果没有指定账户ID，则按账户分组绘图
        if account_id is None:
            # 创建图表
            plt.figure(figsize=(12, 8))
            
            # 获取唯一的账户ID
            account_ids = df['account_id'].unique()
            
            for acc_id in account_ids:
                acc_data = df[df['account_id'] == acc_id]
                plt.plot(acc_data['reading_time'], acc_data['current_usage'], 
                         marker='o', label=f'账户 {acc_id}')
            
            plt.title('所有账户的水表用量')
            plt.xlabel('日期')
            plt.ylabel('用水量')
            plt.grid(True)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 绘制用水量
            ax1.plot(df['reading_time'], df['current_usage'], marker='o', color='blue')
            ax1.set_title(f'账户 {account_id} 的用水量')
            ax1.set_xlabel('日期')
            ax1.set_ylabel('用水量')
            ax1.grid(True)
            
            # 绘制累计读数
            ax2.plot(df['reading_time'], df['current_reading'], marker='o', color='green')
            ax2.set_title(f'账户 {account_id} 的累计读数')
            ax2.set_xlabel('日期')
            ax2.set_ylabel('累计读数')
            ax2.grid(True)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    def analyze_usage_patterns(self, account_id):
        """分析用水模式"""
        df = self.get_readings_as_dataframe(account_id)
        
        if df.empty:
            print("没有找到数据!")
            return
        
        # 计算基本统计数据
        total_usage = df['current_usage'].sum()
        avg_usage = df['current_usage'].mean()
        max_usage = df['current_usage'].max()
        max_usage_date = df.loc[df['current_usage'].idxmax(), 'reading_time']
        zero_usage_days = (df['current_usage'] == 0).sum()
        
        print(f"\n账户 {account_id} 的用水分析:")
        print(f"总用水量: {total_usage:.2f}")
        print(f"平均用水量: {avg_usage:.2f}")
        print(f"最大用水量: {max_usage:.2f} (日期: {max_usage_date.strftime('%Y-%m-%d')})")
        print(f"零用水天数: {zero_usage_days}")
        
        # 检测异常用水
        threshold = avg_usage * 2
        anomalies = df[df['current_usage'] > threshold]
        
        if not anomalies.empty:
            print("\n检测到异常用水:")
            for _, row in anomalies.iterrows():
                print(f"日期: {row['reading_time'].strftime('%Y-%m-%d')}, 用水量: {row['current_usage']:.2f}")
        else:
            print("\n未检测到异常用水")
        
        # 检测可能的漏水
        # 连续多天小量用水可能表示漏水
        consecutive_days = 3
        min_usage = 0.01
        
        for i in range(len(df) - consecutive_days + 1):
            window = df.iloc[i:i+consecutive_days]
            if all(window['current_usage'] >= min_usage) and all(window['current_usage'] < avg_usage * 0.5):
                print(f"\n可能存在漏水 (从 {window.iloc[0]['reading_time'].strftime('%Y-%m-%d')} 到 {window.iloc[-1]['reading_time'].strftime('%Y-%m-%d')})")
                print(f"这段时间的用水量: {window['current_usage'].tolist()}")
                break
    
    def compare_accounts(self, account_ids):
        """比较多个账户的用水情况"""
        if not account_ids:
            print("请提供要比较的账户ID列表")
            return
        
        # 创建一个空的DataFrame来存储所有账户的数据
        all_data = pd.DataFrame()
        
        for account_id in account_ids:
            df = self.get_readings_as_dataframe(account_id)
            if not df.empty:
                # 添加一个标识账户的列
                df['account_label'] = f'账户 {account_id}'
                all_data = pd.concat([all_data, df])
        
        if all_data.empty:
            print("没有找到数据!")
            return
        
        # 计算每个账户的总用水量和平均用水量
        summary = all_data.groupby('account_id').agg({
            'current_usage': ['sum', 'mean', 'max'],
            'current_reading': ['first', 'last']
        })
        
        print("\n账户比较:")
        for account_id in account_ids:
            if account_id in summary.index:
                total = summary.loc[account_id, ('current_usage', 'sum')]
                avg = summary.loc[account_id, ('current_usage', 'mean')]
                max_usage = summary.loc[account_id, ('current_usage', 'max')]
                first_reading = summary.loc[account_id, ('current_reading', 'first')]
                last_reading = summary.loc[account_id, ('current_reading', 'last')]
                
                print(f"账户 {account_id}:")
                print(f"  总用水量: {total:.2f}")
                print(f"  平均用水量: {avg:.2f}")
                print(f"  最大用水量: {max_usage:.2f}")
                print(f"  初始读数: {first_reading:.2f}")
                print(f"  最终读数: {last_reading:.2f}")
                print(f"  总计量表变化: {last_reading - first_reading:.2f}")
                print()
        
        # 绘制比较图表
        plt.figure(figsize=(12, 8))
        
        for account_id in account_ids:
            account_data = all_data[all_data['account_id'] == account_id]
            plt.plot(account_data['reading_time'], account_data['current_usage'], 
                     marker='o', label=f'账户 {account_id}')
        
        plt.title('账户用水量比较')
        plt.xlabel('日期')
        plt.ylabel('用水量')
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    """主函数"""
    reader = WaterMeterDataReader()
    
    try:
        while True:
            print("\n水表数据分析工具")
            print("1. 查看所有账户")
            print("2. 查看特定账户的水表读数")
            print("3. 绘制用水量图表")
            print("4. 分析用水模式")
            print("5. 比较多个账户")
            print("0. 退出")
            
            choice = input("\n请选择操作 (0-5): ")
            
            if choice == '0':
                break
            elif choice == '1':
                reader.get_all_accounts()
            elif choice == '2':
                account_id = input("请输入账户ID: ")
                readings = reader.get_account_readings(account_id)
                
                # 显示前5条记录
                print("\n前5条记录:")
                for i, reading in enumerate(readings[:5]):
                    print(f"{i+1}. 日期: {reading['reading_time']}, 读数: {reading['current_reading']}, 用量: {reading['current_usage']}")
            elif choice == '3':
                account_id = input("请输入账户ID (留空显示所有账户): ")
                start_date = input("请输入开始日期 (YYYY-MM-DD, 留空表示不限): ")
                end_date = input("请输入结束日期 (YYYY-MM-DD, 留空表示不限): ")
                
                account_id = account_id if account_id else None
                start_date = start_date if start_date else None
                end_date = end_date if end_date else None
                
                reader.plot_water_usage(account_id, start_date, end_date)
            elif choice == '4':
                account_id = input("请输入账户ID: ")
                reader.analyze_usage_patterns(account_id)
            elif choice == '5':
                account_ids_input = input("请输入要比较的账户ID (用逗号分隔): ")
                account_ids = [id.strip() for id in account_ids_input.split(',')]
                reader.compare_accounts(account_ids)
            else:
                print("无效的选择，请重试")
    
    finally:
        reader.close()

if __name__ == "__main__":
    main() 