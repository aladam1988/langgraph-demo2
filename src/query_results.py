#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析结果查询工具

此脚本用于查询水用量分析结果数据库，支持多种查询条件，
并可以显示详细的分析结果或导出为CSV/JSON格式
"""

import os
import sys
import sqlite3
import json
import csv
import argparse
from datetime import datetime, timedelta
from tabulate import tabulate  # 如果没有安装，需要先 pip install tabulate

# 添加src目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config():
    """加载配置（简化版，仅供查询脚本使用）"""
    # 获取项目根目录路径
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    return {
        "analysis_results_db": os.path.join(root_dir, "data/analysis_results.db"),
        "root_dir": root_dir,
    }

def get_leak_probability_stats(db_path):
    """获取漏水可能性统计数据"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT leak_probability, COUNT(*) as count 
            FROM analysis_results 
            GROUP BY leak_probability
            ORDER BY 
            CASE 
                WHEN leak_probability = '高' THEN 1
                WHEN leak_probability = '中' THEN 2
                WHEN leak_probability = '低' THEN 3
                WHEN leak_probability = '无' THEN 4
                ELSE 5
            END
        """)
        
        stats = cursor.fetchall()
        conn.close()
        
        # 格式化输出
        total = sum(row[1] for row in stats)
        result = []
        for prob, count in stats:
            percentage = count / total * 100 if total > 0 else 0
            result.append((prob, count, f"{percentage:.1f}%"))
        
        return result, total
    except Exception as e:
        print(f"获取漏水可能性统计失败: {str(e)}")
        return [], 0

def get_risk_stats_by_date(db_path, days=30):
    """获取最近一段时间内的风险统计变化"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 计算开始日期
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        cursor.execute("""
            SELECT 
                analysis_date, 
                COUNT(*) as total,
                SUM(CASE WHEN risk_warning = 1 THEN 1 ELSE 0 END) as risk_count,
                SUM(CASE WHEN leak_probability = '高' THEN 1 ELSE 0 END) as high_prob,
                SUM(CASE WHEN leak_probability = '中' THEN 1 ELSE 0 END) as med_prob,
                SUM(CASE WHEN leak_probability = '低' THEN 1 ELSE 0 END) as low_prob,
                SUM(CASE WHEN leak_probability = '无' THEN 1 ELSE 0 END) as no_prob
            FROM analysis_results 
            WHERE analysis_date >= ?
            GROUP BY analysis_date
            ORDER BY analysis_date
        """, (start_date,))
        
        result = cursor.fetchall()
        conn.close()
        
        return result
    except Exception as e:
        print(f"获取风险统计数据失败: {str(e)}")
        return []

def get_high_risk_accounts(db_path, limit=10):
    """获取高风险账户列表"""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                account_id, 
                analysis_date, 
                leak_probability, 
                max_usage,
                max_daily_avg,
                abnormal_usage,
                leak_risk_text
            FROM analysis_results 
            WHERE risk_warning = 1 OR leak_probability IN ('高', '中')
            ORDER BY 
                CASE 
                    WHEN leak_probability = '高' THEN 1
                    WHEN leak_probability = '中' THEN 2
                    ELSE 3
                END,
                analysis_date DESC
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in results]
    except Exception as e:
        print(f"获取高风险账户列表失败: {str(e)}")
        return []

def export_to_csv(results, output_file):
    """将结果导出为CSV文件"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # 写入CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if not results:
                print("没有数据可导出")
                return False
                
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
        print(f"已成功导出 {len(results)} 条记录到 {output_file}")
        return True
    except Exception as e:
        print(f"导出CSV失败: {str(e)}")
        return False

def export_to_json(results, output_file):
    """将结果导出为JSON文件"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # 写入JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"已成功导出 {len(results)} 条记录到 {output_file}")
        return True
    except Exception as e:
        print(f"导出JSON失败: {str(e)}")
        return False

def print_results(results, detailed=False):
    """打印查询结果"""
    if not results:
        print("没有找到符合条件的记录")
        return
    
    if detailed:
        print(f"\n找到 {len(results)} 条记录，显示详细信息:")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            account_id = result['account_id']
            analysis_date = result['analysis_date']
            leak_prob = result['leak_probability']
            risk_warning = "是" if result['risk_warning'] == 1 else "否"
            
            print(f"\n[{i}] 账户: {account_id}  分析日期: {analysis_date}")
            print(f"漏水可能性: {leak_prob}  风险警告: {risk_warning}")
            print("-" * 60)
            print(f"用水数据:")
            print(f"  最大用水量: {result['max_usage']}方 (发生于{result['max_usage_date']})")
            print(f"  平均用水量: {result['avg_usage']:.2f}方")
            print(f"  最大日均用水量: {result['max_daily_avg']:.2f}方")
            print(f"  历史日均用水量: {result['historical_avg']:.2f}方")
            print(f"  当前与历史比率: {result['increase_ratio']:.2f}")
            print("-" * 60)
            print(f"异常用水: {result['abnormal_usage']}")
            print(f"漏水风险: {result['leak_risk_text']}")
            print("=" * 80)
    else:
        # 使用tabulate创建表格输出
        table_data = []
        for result in results:
            table_data.append([
                result['account_id'],
                result['analysis_date'],
                result['leak_probability'],
                "是" if result['risk_warning'] == 1 else "否",
                f"{result['max_usage']:.2f}方",
                result['abnormal_usage'][:30] + ("..." if len(result['abnormal_usage']) > 30 else ""),
            ])
        
        headers = ["账户ID", "分析日期", "漏水可能性", "风险警告", "最大用水量", "异常用水"]
        print(f"\n找到 {len(results)} 条记录:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print("\n提示: 使用 --detailed 选项可以查看完整分析结果")

def query_analysis_results(db_path, **filters):
    """查询分析结果数据库"""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # 使结果以字典形式返回
        cursor = conn.cursor()
        
        # 构建查询
        query = "SELECT * FROM analysis_results WHERE 1=1"
        params = []
        
        # 应用过滤器
        if "account_id" in filters and filters["account_id"]:
            query += " AND account_id = ?"
            params.append(filters["account_id"])
        
        if "analysis_date" in filters and filters["analysis_date"]:
            query += " AND analysis_date = ?"
            params.append(filters["analysis_date"])
        
        if "leak_probability" in filters and filters["leak_probability"]:
            query += " AND leak_probability = ?"
            params.append(filters["leak_probability"])
        
        if "risk_warning" in filters and filters["risk_warning"]:
            query += " AND risk_warning = ?"
            params.append(1)
            
        if "min_usage" in filters and filters["min_usage"] is not None:
            query += " AND max_usage >= ?"
            params.append(float(filters["min_usage"]))
        
        if "days" in filters and filters["days"] is not None:
            # 计算开始日期
            start_date = (datetime.now() - timedelta(days=int(filters["days"]))).strftime("%Y-%m-%d")
            query += " AND analysis_date >= ?"
            params.append(start_date)
        
        # 排序
        query += " ORDER BY analysis_date DESC, account_id"
        
        # 执行查询
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # 转换为字典列表
        results_list = [dict(row) for row in results]
        
        conn.close()
        return results_list
    except Exception as e:
        print(f"查询分析结果数据库失败: {str(e)}")
        return []

def view_full_report(db_path, account_id, analysis_date=None):
    """查看指定账户的完整分析报告"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        if analysis_date:
            cursor.execute(
                "SELECT full_report FROM analysis_results WHERE account_id = ? AND analysis_date = ? ORDER BY analysis_date DESC LIMIT 1", 
                (account_id, analysis_date)
            )
        else:
            cursor.execute(
                "SELECT full_report FROM analysis_results WHERE account_id = ? ORDER BY analysis_date DESC LIMIT 1", 
                (account_id,)
            )
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            return result[0]
        else:
            return f"未找到账户 {account_id} 的分析报告"
    except Exception as e:
        return f"查看完整报告失败: {str(e)}"

def show_report(db_path, account_id, analysis_date=None):
    """显示完整的分析报告"""
    report = view_full_report(db_path, account_id, analysis_date)
    print("\n" + "=" * 80)
    print(f"账户 {account_id} 完整分析报告:")
    print("-" * 80)
    print(report)
    print("=" * 80)

def main():
    """主函数"""
    config = load_config()
    db_path = config["analysis_results_db"]
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='水用量分析结果查询工具')
    
    # 设置子命令
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 查询命令
    query_parser = subparsers.add_parser('query', help='查询分析结果')
    query_parser.add_argument('--account', help='按账户ID查询')
    query_parser.add_argument('--date', help='按分析日期查询 (YYYY-MM-DD)')
    query_parser.add_argument('--probability', help='按漏水可能性查询 (无/低/中/高)')
    query_parser.add_argument('--risk', action='store_true', help='只查询有风险警告的记录')
    query_parser.add_argument('--min-usage', type=float, help='按最小用水量查询 (单位:方)')
    query_parser.add_argument('--days', type=int, help='最近几天的记录 (例如: 7表示最近一周)')
    query_parser.add_argument('--export', help='导出结果 (指定文件路径，支持.csv或.json格式)')
    query_parser.add_argument('--detailed', action='store_true', help='显示详细信息')
    
    # 统计命令
    stats_parser = subparsers.add_parser('stats', help='显示统计信息')
    stats_parser.add_argument('--days', type=int, default=30, help='统计最近几天的数据 (默认: 30)')
    
    # 报告命令
    report_parser = subparsers.add_parser('report', help='查看完整分析报告')
    report_parser.add_argument('account', help='账户ID')
    report_parser.add_argument('--date', help='分析日期 (YYYY-MM-DD，默认为最新)')
    
    # 高风险账户命令
    risk_parser = subparsers.add_parser('risk', help='查看高风险账户列表')
    risk_parser.add_argument('--limit', type=int, default=10, help='显示的账户数量 (默认: 10)')
    
    # 解析参数
    args = parser.parse_args()
    
    # 如果数据库不存在
    if not os.path.exists(db_path):
        print(f"错误: 分析结果数据库不存在 ({db_path})")
        print("请先运行分析程序生成分析结果")
        return
    
    # 没有指定命令时显示帮助
    if not args.command:
        parser.print_help()
        return
    
    # 统计命令
    if args.command == 'stats':
        print("\n水用量分析结果统计")
        print("=" * 50)
        
        # 漏水可能性统计
        stats, total = get_leak_probability_stats(db_path)
        print(f"\n漏水可能性分布 (总共 {total} 条记录):")
        print("-" * 40)
        for prob, count, percentage in stats:
            print(f"  {prob or '未知'}: {count} 条 ({percentage})")
        
        # 风险警告统计
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM analysis_results WHERE risk_warning = 1")
        risk_count = cursor.fetchone()[0]
        risk_percentage = risk_count / total * 100 if total > 0 else 0
        print(f"\n风险警告: {risk_count} 条 ({risk_percentage:.1f}%)")
        
        # 最近分析日期
        cursor.execute("SELECT analysis_date FROM analysis_results ORDER BY analysis_date DESC LIMIT 1")
        result = cursor.fetchone()
        if result:
            print(f"\n最近分析日期: {result[0]}")
            
        # 账户数量
        cursor.execute("SELECT COUNT(DISTINCT account_id) FROM analysis_results")
        account_count = cursor.fetchone()[0]
        print(f"总分析账户数: {account_count}")
        
        # 时间范围内的风险变化统计
        print(f"\n最近 {args.days} 天内的风险统计:")
        risk_stats = get_risk_stats_by_date(db_path, args.days)
        if risk_stats:
            print("-" * 80)
            print("  日期      | 总计 | 风险数 | 高风险 | 中风险 | 低风险 | 无风险 ")
            print("-" * 80)
            for date, total, risk, high, med, low, no in risk_stats:
                print(f"  {date} | {total:4d} | {risk:6d} | {high:6d} | {med:6d} | {low:6d} | {no:6d}")
        else:
            print("  没有找到相关统计数据")
        
        conn.close()
        return
    
    # 高风险账户命令
    if args.command == 'risk':
        high_risk = get_high_risk_accounts(db_path, args.limit)
        if high_risk:
            print("\n高风险账户列表:")
            print("=" * 80)
            table_data = []
            for i, acct in enumerate(high_risk, 1):
                table_data.append([
                    i,
                    acct['account_id'],
                    acct['analysis_date'],
                    acct['leak_probability'],
                    f"{acct['max_usage']:.2f}方",
                    f"{acct['max_daily_avg']:.2f}方/日",
                    acct['leak_risk_text'][:40] + ("..." if len(acct['leak_risk_text']) > 40 else "")
                ])
            
            headers = ["序号", "账户ID", "分析日期", "漏水可能性", "最大用水量", "最大日均", "漏水风险"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        else:
            print("没有找到高风险账户")
        return
    
    # 查看报告命令
    if args.command == 'report':
        show_report(db_path, args.account, args.date)
        return
    
    # 查询命令
    if args.command == 'query':
        filters = {}
        
        if args.account:
            filters['account_id'] = args.account
            
        if args.date:
            filters['analysis_date'] = args.date
            
        if args.probability:
            filters['leak_probability'] = args.probability
            
        if args.risk:
            filters['risk_warning'] = True
            
        if args.min_usage is not None:
            filters['min_usage'] = args.min_usage
            
        if args.days is not None:
            filters['days'] = args.days
        
        # 执行查询
        results = query_analysis_results(db_path, **filters)
        
        # 打印结果
        print_results(results, args.detailed)
        
        # 如果指定了导出
        if args.export and results:
            file_path = args.export
            if file_path.lower().endswith('.csv'):
                export_to_csv(results, file_path)
            elif file_path.lower().endswith('.json'):
                export_to_json(results, file_path)
            else:
                print("不支持的导出格式，请使用.csv或.json后缀")
        
        return

if __name__ == "__main__":
    main() 