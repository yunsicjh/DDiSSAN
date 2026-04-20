#!/bin/bash

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 设置PYTHONPATH并生成所有数据集的词汇表
echo "🔧 Building vocabularies for all datasets..."

echo "📝 Processing Restaurants_corenlp..."
PYTHONPATH=$(pwd) python src/data/preprocess/prepare_vocab.py --data_dir data/Restaurants_corenlp --vocab_dir data/Restaurants_corenlp

echo "📝 Processing Laptops_corenlp..."  
PYTHONPATH=$(pwd) python src/data/preprocess/prepare_vocab.py --data_dir data/Laptops_corenlp --vocab_dir data/Laptops_corenlp

echo "📝 Processing Tweets_corenlp..."
PYTHONPATH=$(pwd) python src/data/preprocess/prepare_vocab.py --data_dir data/Tweets_corenlp --vocab_dir data/Tweets_corenlp

# echo "📝 Processing MAMS_corenlp..."
# PYTHONPATH=$(pwd) python src/data/preprocess/prepare_vocab.py --data_dir data/MAMS_corenlp --vocab_dir data/MAMS_corenlp

# echo "📝 Processing semeval15..."
# PYTHONPATH=$(pwd) python src/data/preprocess/prepare_vocab.py --data_dir data/semeval15 --vocab_dir data/semeval15

# echo "📝 Processing semeval16..."
# PYTHONPATH=$(pwd) python src/data/preprocess/prepare_vocab.py --data_dir data/semeval16 --vocab_dir data/semeval16

echo "✅ All vocabularies built successfully!"
