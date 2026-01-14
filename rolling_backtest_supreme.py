"""
GUCP-X Supreme Rolling Backtest Engine
首席全维量化科学家专用 - 滚动回测系统
此脚本为独立的高级外挂，用于执行严格的逐期滚动验证。
"""

import sys
import os
import json
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch

# 导入主程序核心引擎
try:
    # 假设主程序在同一目录下，或者在 PYTHONPATH 中
    import gucp_x_supreme_unified as supreme
except ImportError:
    # 尝试添加当前目录并重试
    sys.path.append(os.getcwd())
    import gucp_x_supreme_unified as supreme

class SupremeRollingBacktester:
    def __init__(self, validation_size=300, retrain_interval=1):
        self.logger = self._setup_logger()
        self.validation_size = validation_size
        self.retrain_interval = retrain_interval # 每隔 N 期重训一次
        
        # 初始化核心引擎
        supreme.SupremeConfig.init_environment()
        self.data_engine = supreme.DataEngine()
        self.results_dir = supreme.SupremeConfig.REPORT_DIR / "rolling_backtests"
        self.results_dir.mkdir(exist_ok=True)
        
        # 初始化模型管理器
        self.manager = supreme.SupremeManager()
        
        # 加载全量数据
        self.full_history = self.data_engine.history
        self._validate_data()

    def _setup_logger(self):
        logger = logging.getLogger("RollingBacktest")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
        
        # 控制台输出
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def _validate_data(self):
        total = len(self.full_history)
        if total < self.validation_size + 100:
            raise ValueError(f"数据量不足！总共 {total} 期，要求验证集 {self.validation_size} 期。")
        self.logger.info(f"📊 数据加载完成: 总计 {total} 期，验证集 {self.validation_size} 期")

    def run(self):
        """执行滚动回测主循环"""
        total = len(self.full_history)
        start_idx = total - self.validation_size
        
        # 初始训练集
        current_history = self.full_history[:start_idx]
        validation_set = self.full_history[start_idx:]
        
        results = []
        pbar = tqdm(total=len(validation_set), desc="🔄 Rolling Backtest")
        
        # 记录累积收益
        cum_pnl = 0
        
        for i, target_draw in enumerate(validation_set):
            period = target_draw['period']
            
            # 1. 动态训练 (Periodic Retraining)
            if i % self.retrain_interval == 0:
                self.logger.info(f"🧠 [Period {period}] Re-training models... (Train Size: {len(current_history)})")
                
                # 强制重训模式 mode='rolling'，避免读取非此轮的缓存
                # 注意：这里我们调用 train_or_load 但传入 force=True 来确保使用最新的 current_history
                # 为了效率，我们只重训核心模型：Global (MLP/RF) 和 Positional
                data_time = time.time() # 伪造时间戳强制更新
                
                # A. 市场感知获取推荐窗口
                try:
                    regime_info = supreme.MarketEngine.analyze_regime(current_history)
                    rec_window = regime_info['recommended_window']
                except:
                    rec_window = 12
                
                self.manager.global_ml.train_or_load(current_history, data_time, window=rec_window, mode='rolling', force=True)
                self.manager.pos_ml.train_or_load(current_history, data_time, mode='rolling', force=True)
                
                # TCN 训练较慢，可以选择不每期重顺，或者每 10 期重训一次
                if i % 10 == 0:
                     self.manager.tcn_engine.train_or_load(current_history, data_time, mode='rolling', force=True)
            
            # 2. 生成预测
            # 使用最新的 current_history 进行预测
            probs_dict = self.manager.global_ml.predict(current_history, window=12) # 窗口可以动态化
            pos_preds = self.manager.pos_ml.predict(current_history)
            tcn_probs = self.manager.tcn_engine.predict(current_history)
            arima_probs = self.manager.arima_engine.predict(current_history)
            
            # 3. 核心融合生成 Smart Pool
            pool_info = supreme.KernelEngine.generate_smart_pool(
                probs_dict, pos_preds, current_history,
                tcn_probs=tcn_probs,
                arima_probs=arima_probs,
                loaded_core_points=self.data_engine.core_points
            )
            
            smart_pool = pool_info['smart_pool']
            core_20 = pool_info['core_20']
            
            # 4. 验证结果
            real_nums = set(target_draw['sorted'])
            pool_hits = len(real_nums.intersection(smart_pool))
            core_hits = len(real_nums.intersection(core_20))
            
            # 简单 PnL 计算 (假设每次投入 Smart Pool 大小，中 1 个得 1 分 - 仅作示意)
            # 实际 PnL 应该更复杂，这里用 (命中数 * 4.6 - 投入本金) 简易模拟选五及格线
            # 假设智能底是选十玩法的大底
            pnl_step = pool_hits - (len(smart_pool) * 0.25) 
            cum_pnl += pnl_step
            
            # 5. 记录日志
            log_entry = {
                "period": period,
                "date": target_draw['date'],
                "train_size": len(current_history),
                "pool_size": len(smart_pool),
                "pool_hits": pool_hits,
                "core_hits": core_hits,
                "hit_rate_pool": round(pool_hits / len(smart_pool), 4),
                "core_vals": "-".join([f"{n:02d}" for n in core_20]),
                "real_vals": "-".join([f"{n:02d}" for n in sorted(list(real_nums))]),
                "pnl_step": round(pnl_step, 2),
                "cum_pnl": round(cum_pnl, 2),
                "regime": pool_info.get('regime', {}).get('status', 'N/A')
            }
            results.append(log_entry)
            
            # 控制台简报
            if i % 10 == 0 or pool_hits >= 10: # 高光时刻或定期输出
                self.logger.info(f"📍 Period {period} | Core Hits: {core_hits}/20 | Pool Hits: {pool_hits}/{len(smart_pool)} | Cumulative PnL: {cum_pnl:.1f}")
            
            # 6. 不达标反馈 (Adaptive Logic Demo)
            if core_hits < 3:
                self.logger.warning(f"⚠️ 核心命中偏低 ({core_hits})，下一轮将保持重训以调整状态。")
                # 这里可以加入逻辑，例如下一期强制重训 (如果 retrain_interval > 1)
            
            # 7. 滚动：将本期真值加入历史，用于下一期训练/特征提取
            current_history.append(target_draw)
            
            # 8. 定期保存中间结果
            if i % 10 == 0:
                self._save_results(results, is_final=False)
                
            pbar.update(1)
            
        pbar.close()
        self._save_results(results, is_final=True)
        self.logger.info("✅ 滚动回测全流程结束。")

    def _save_results(self, results, is_final=False):
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix = "FINAL" if is_final else "checkpoint"
        filename = f"rolling_backtest_{self.validation_size}periods_{suffix}.csv"
        path = self.results_dir / filename
        df.to_csv(path, index=False, encoding='utf-8-sig')
        if is_final:
            self.logger.info(f"📄 最终报告已生成: {path}")
            
            # 生成简单的统计摘要
            summary = {
                "total_periods": len(results),
                "avg_pool_hits": df['pool_hits'].mean(),
                "avg_core_hits": df['core_hits'].mean(),
                "total_pnl": df['pnl_step'].sum(),
                "win_rate_pool_gt_10": (df['pool_hits'] >= 10).mean()
            }
            summary_path = self.results_dir / f"summary_{timestamp}.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4)

if __name__ == "__main__":
    # 解析命令行参数 (可选)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", type=int, default=300, help="验证集期数 (最新 N 期)")
    parser.add_argument("--interval", type=int, default=1, help="重训间隔 (每 N 期重训一次)")
    args = parser.parse_args()
    
    print(f"🚀 启动滚动回测 | 验证期数: {args.periods} | 重训间隔: {args.interval}")
    
    try:
        tester = SupremeRollingBacktester(validation_size=args.periods, retrain_interval=args.interval)
        tester.run()
    except KeyboardInterrupt:
        print("\n🛑 用户中断测试")
    except Exception as e:
        print(f"\n❌ 发生严重错误: {e}")
        import traceback
        traceback.print_exc()
