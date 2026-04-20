#!/bin/bash
# SSAN改进模型实验命令集合

# SSAN模型debug测试
python src/train.py experiment=ssan_restaurants debug=fdr
python src/train.py experiment=ssan_laptops debug=fdr
python src/train.py experiment=ssan_tweets debug=fdr

# 模型在各数据集上的完整训练 
python src/train.py experiment=ssan_restaurants trainer=gpu
python src/train.py experiment=ssan_laptops trainer=gpu
python src/train.py experiment=ssan_tweets trainer=gpu


# ============================================================================
# 🔬 改进模型超参数调优实验
# ============================================================================

# Hyperparameter Sweep for Restaurants Dataset
# batch_size=[8,16,32,64], lr=[1e-3,2e-4,3e-5], seed=[1000,1024]
echo "🍽️ Starting Restaurants dataset hyperparameter sweep..."

# Restaurants - Batch size 8
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=8 model.optimizer.lr=1e-3 seed=1000
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=8 model.optimizer.lr=1e-3 seed=1024
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=8 model.optimizer.lr=2e-4 seed=1000
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=8 model.optimizer.lr=2e-4 seed=1024
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=8 model.optimizer.lr=3e-5 seed=1000
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=8 model.optimizer.lr=3e-5 seed=1024

# Restaurants - Batch size 16
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=16 model.optimizer.lr=1e-3 seed=1000
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=16 model.optimizer.lr=1e-3 seed=1024
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=16 model.optimizer.lr=2e-4 seed=1000
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=16 model.optimizer.lr=2e-4 seed=1024
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=16 model.optimizer.lr=3e-5 seed=1000
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=16 model.optimizer.lr=3e-5 seed=1024

# Restaurants - Batch size 32
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=32 model.optimizer.lr=1e-3 seed=1000
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=32 model.optimizer.lr=1e-3 seed=1024
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=32 model.optimizer.lr=2e-4 seed=1000
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=32 model.optimizer.lr=2e-4 seed=1024
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=32 model.optimizer.lr=3e-5 seed=1000
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=32 model.optimizer.lr=3e-5 seed=1024

# Restaurants - Batch size 64
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=64 model.optimizer.lr=1e-3 seed=1000
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=64 model.optimizer.lr=1e-3 seed=1024
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=64 model.optimizer.lr=2e-4 seed=1000
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=64 model.optimizer.lr=2e-4 seed=1024
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=64 model.optimizer.lr=3e-5 seed=1000
python src/train.py experiment=ssan_restaurants trainer=gpu data.batch_size=64 model.optimizer.lr=3e-5 seed=1024

echo "✅ Restaurants dataset sweep completed!"

# Hyperparameter Sweep for Laptops Dataset
# batch_size=[8,16,32,64], lr=[1e-3,2e-4,3e-5], seed=[1000,1024]
echo "💻 Starting Laptops dataset hyperparameter sweep..."

# Laptops - Batch size 8
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=8 model.optimizer.lr=1e-3 seed=1000
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=8 model.optimizer.lr=1e-3 seed=1024
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=8 model.optimizer.lr=2e-4 seed=1000
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=8 model.optimizer.lr=2e-4 seed=1024
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=8 model.optimizer.lr=3e-5 seed=1000
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=8 model.optimizer.lr=3e-5 seed=1024

# Laptops - Batch size 16
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=16 model.optimizer.lr=1e-3 seed=1000
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=16 model.optimizer.lr=1e-3 seed=1024
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=16 model.optimizer.lr=2e-4 seed=1000
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=16 model.optimizer.lr=2e-4 seed=1024
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=16 model.optimizer.lr=3e-5 seed=1000
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=16 model.optimizer.lr=3e-5 seed=1024

# Laptops - Batch size 32
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=32 model.optimizer.lr=1e-3 seed=1000
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=32 model.optimizer.lr=1e-3 seed=1024
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=32 model.optimizer.lr=2e-4 seed=1000
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=32 model.optimizer.lr=2e-4 seed=1024
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=32 model.optimizer.lr=3e-5 seed=1000
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=32 model.optimizer.lr=3e-5 seed=1024

# Laptops - Batch size 64
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=64 model.optimizer.lr=1e-3 seed=1000
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=64 model.optimizer.lr=1e-3 seed=1024
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=64 model.optimizer.lr=2e-4 seed=1000
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=64 model.optimizer.lr=2e-4 seed=1024
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=64 model.optimizer.lr=3e-5 seed=1000
python src/train.py experiment=ssan_laptops trainer=gpu data.batch_size=64 model.optimizer.lr=3e-5 seed=1024

echo "✅ Laptops dataset sweep completed!"

# Hyperparameter Sweep for Tweets Dataset
# batch_size=[8,16,32,64], lr=[1e-3,2e-4,3e-5], seed=[1000,1024]
echo "🐦 Starting Tweets dataset hyperparameter sweep..."

# Tweets - Batch size 8
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=8 model.optimizer.lr=1e-3 seed=1000
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=8 model.optimizer.lr=1e-3 seed=1024
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=8 model.optimizer.lr=2e-4 seed=1000
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=8 model.optimizer.lr=2e-4 seed=1024
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=8 model.optimizer.lr=3e-5 seed=1000
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=8 model.optimizer.lr=3e-5 seed=1024

# Tweets - Batch size 16
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=16 model.optimizer.lr=1e-3 seed=1000
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=16 model.optimizer.lr=1e-3 seed=1024
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=16 model.optimizer.lr=2e-4 seed=1000
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=16 model.optimizer.lr=2e-4 seed=1024
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=16 model.optimizer.lr=3e-5 seed=1000
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=16 model.optimizer.lr=3e-5 seed=1024

# Tweets - Batch size 32
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=32 model.optimizer.lr=1e-3 seed=1000
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=32 model.optimizer.lr=1e-3 seed=1024
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=32 model.optimizer.lr=2e-4 seed=1000
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=32 model.optimizer.lr=2e-4 seed=1024
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=32 model.optimizer.lr=3e-5 seed=1000
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=32 model.optimizer.lr=3e-5 seed=1024

# Tweets - Batch size 64
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=64 model.optimizer.lr=1e-3 seed=1000
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=64 model.optimizer.lr=1e-3 seed=1024
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=64 model.optimizer.lr=2e-4 seed=1000
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=64 model.optimizer.lr=2e-4 seed=1024
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=64 model.optimizer.lr=3e-5 seed=1000
python src/train.py experiment=ssan_tweets trainer=gpu data.batch_size=64 model.optimizer.lr=3e-5 seed=1024

echo "✅ Tweets dataset sweep completed!"

# ============================================================================
# 📊 超参数扫描总结
# ============================================================================
echo "🎯 超参数扫描总结："
echo "   - 数据集: Restaurants, Laptops, Tweets"
echo "   - Batch Size: [8, 16, 32, 64]"
echo "   - Learning Rate: [1e-3, 2e-4, 3e-5]"
echo "   - Random Seeds: [1000, 1024]"
echo "   - 总实验数量: 3 datasets × 4 batch_sizes × 3 lr × 2 seeds = 72 experiments"
echo "🚀 所有超参数扫描实验已完成！"

# ============================================================================
# 🎮 便捷运行函数 - 可以单独调用
# ============================================================================

# 运行单个数据集的超参数扫描
run_restaurants_sweep() {
    echo "🍽️ 运行 Restaurants 数据集超参数扫描..."
    # 这里可以复制上面 Restaurants 的所有命令
}

run_laptops_sweep() {
    echo "💻 运行 Laptops 数据集超参数扫描..."
    # 这里可以复制上面 Laptops 的所有命令
}

run_tweets_sweep() {
    echo "🐦 运行 Tweets 数据集超参数扫描..."
    # 这里可以复制上面 Tweets 的所有命令
}

# 运行所有数据集的超参数扫描
run_all_sweeps() {
    echo "🚀 开始运行所有数据集的超参数扫描..."
    run_restaurants_sweep
    run_laptops_sweep
    run_tweets_sweep
    echo "✅ 所有超参数扫描已完成！"
}

# ============================================================================
# 💡 使用说明
# ============================================================================
# 要运行超参数扫描，请使用以下命令：
#
# 1. 运行所有数据集的扫描（72个实验）：
#    bash scripts/run.sh
#
# 2. 或者在脚本中调用特定函数：
#    source scripts/run.sh
#    run_restaurants_sweep  # 只运行 Restaurants 扫描
#    run_laptops_sweep      # 只运行 Laptops 扫描  
#    run_tweets_sweep       # 只运行 Tweets 扫描
#
# 3. 或者直接执行特定部分的命令行
#
# 📊 超参数网格搜索覆盖:
# - Restaurants: 4×3×2 = 24 experiments
# - Laptops: 4×3×2 = 24 experiments  
# - Tweets: 4×3×2 = 24 experiments
# - 总计: 72 experiments
#
# ⚡ 推荐使用高效脚本:
#    bash scripts/hyperparameter_sweep.sh    # 更高效的批量运行
#    python analyze_hyperparameter_results.py # 分析结果和生成最佳配置
