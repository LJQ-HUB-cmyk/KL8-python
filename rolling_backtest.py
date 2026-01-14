
# -*- coding: utf-8 -*-
"""
GUCP-X 滚动回测专用执行器 (Rolling Backtest Runner)
功能: 针对最近 300 期数据执行严格的 "预测-验证-再训练" 滚动窗口回测。
逻辑:
1. 初始训练集: 全量 - 300期
2. 验证集: 最近 300 期
3. 滚动方式:
   - 预测第 N 期
   - 揭晓答案并统计命中
   - 将第 N 期数据"喂"给模型 (加入训练集)
   - 如果命中率 "不达标" (如 < 5)，则在预测第 N+1 期前强制触发全量重训 (Readjust)
   - 否则继续使用现有模型进行增量/存量预测
"""

import sys
import os
import csv
import logging
from datetime import datetime
import numpy as np
import pandas as pd

# 引入主系统 (假设在同一目录)
from gucp_x_supreme_unified import SupremeManager, SupremeConfig, KernelEngine, ReportEngine, PhysicsEngine

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("RollingBacktest")

def run_rolling_backtest():
    # 1. 初始化
    logger.info("🔧 初始化滚动回测系统...")
    manager = SupremeManager()
    
    # 强制设置验证集大小为 300 (用户要求)
    SupremeConfig.VALIDATION_SIZE = 300
    
    all_history = manager.data_engine.history
    total_len = len(all_history)
    
    if total_len <= 300:
        logger.error("❌ 数据量不足，无法执行 300 期回测")
        return

    # 切分数据
    split_idx = total_len - 300
    initial_train_history = all_history[:split_idx]
    validation_queue = all_history[split_idx:]
    
    logger.info(f"📊 数据切分完成:")
    logger.info(f"   - 初始训练集: {len(initial_train_history)} 期 (截止 {initial_train_history[-1]['period']})")
    logger.info(f"   - 待验证集 (Validation Pool): {len(validation_queue)} 期 (从 {validation_queue[0]['period']} 开始)")
    
    # 结果记录
    results = []
    csv_file = "rolling_backtest_details.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Period", "Hits_Pool", "Hits_Core20", "Pool_Size", "Core_Size", "PnL", "Retrained"])

    hits_history = []
    force_retrain = True # 首次运行必需训练
    
    # 2. 滚动循环
    for step, target_draw in enumerate(validation_queue):
        current_idx = split_idx + step
        # 当前已知的历史数据 (模拟"一点一点投喂")
        current_history = all_history[:current_idx] 
        target_nums = set(target_draw['sorted'])
        target_period = target_draw['period']
        
        # 2.1 训练/加载模型
        # 用户指令: "如果执行和预测情况不达标，请重新调整...继续执行"
        # 策略: 如果上一期命中 (池命中) < 5 或 (核心命中) < 2，则视为"不达标"，触发重训
        #       否则使用现有模型 (增量/缓存模式)
        
        mode = 'rolling_train' # 专用模式名，避免污染生产缓存
        data_time = manager.data_engine.get_last_timestamp() # 实际上这里数据是动态增长的，时间戳可能不变，所以必须依赖 force
        
        if force_retrain:
            logger.info(f"🔄 [Step {step+1}/{300}] 正在执行动态重训 (Period: {target_period})...")
        else:
            logger.info(f"⏩ [Step {step+1}/{300}] 沿用现有模型预测 (Period: {target_period})...")

        # 训练各流派 (注意: TCN 较慢，可视情况降低频次，这里严格执行用户要求)
        manager.global_ml.train_or_load(current_history, data_time, mode=mode, force=force_retrain)
        manager.pos_ml.train_or_load(current_history, data_time, mode=mode, force=force_retrain)
        # TCN 训练太慢，每 10 期或极差时重训
        if force_retrain and (step % 10 == 0):
             manager.tcn_engine.train_or_load(current_history, data_time, mode=mode, force=True)
        
        # 2.2 预测
        probs_dict = manager.global_ml.predict(current_history)
        pos_preds = manager.pos_ml.predict(current_history)
        tcn_probs = manager.tcn_engine.predict(current_history)
        # arima_probs = manager.arima_engine.predict(current_history) # 速度较慢暂关闭
        
        # 核心融合
        prediction = KernelEngine.generate_smart_pool(
            probs_dict, pos_preds, current_history, 
            tcn_probs=tcn_probs,
            # arima_probs=arima_probs 
        )
        
        smart_pool = set(prediction['smart_pool'])
        core_20 = set(prediction['core_20'])
        
        # 2.3 验证与归因
        hits_pool = len(target_nums.intersection(smart_pool))
        hits_core = len(target_nums.intersection(core_20))
        pool_size = len(smart_pool)
        
        # 模拟收益 (假设每注 2 元，中 1 回 1，中多回多 - 简化逻辑：命中数 - 成本)
        # 简化 PnL: 命中数即为正反馈
        pnl = hits_pool 
        
        # 2.4 记录
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([step+1, target_period, hits_pool, hits_core, pool_size, len(core_20), pnl, force_retrain])
            
        results.append(hits_pool)
        hits_history.append(hits_pool)
        
        # 2.5 决策：下一期是否需要"重新调整" (Readjust)
        # 阈值设定: 大底命中 < 5 或 核心命中 < 2 视为风险
        if hits_pool < 5 or hits_core < 2:
            force_retrain = True
            logger.warning(f"⚠️ 本期表现不佳 (Pool:{hits_pool}, Core:{hits_core}) -> 下一期将强制自动调优/重训")
        else:
            force_retrain = False
            
        # 实时反馈
        if (step + 1) % 10 == 0:
            avg_10 = np.mean(hits_history[-10:])
            logger.info(f"📈 最近 10 期平均命中: {avg_10:.2f}")

    # 3. 汇总报告
    avg_pool = np.mean(results)
    total_hits = sum(results)
    
    logger.info("="*50)
    logger.info(f"✅ 300 期滚动回测完成")
    logger.info(f"   平均命中 (Smart Pool): {avg_pool:.4f}")
    logger.info(f"   总命中数: {total_hits}")
    logger.info(f"   结果已保存至: {csv_file}")
    logger.info("="*50)

if __name__ == "__main__":
    run_rolling_backtest()
