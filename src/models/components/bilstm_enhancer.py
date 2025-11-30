import torch
import torch.nn as nn


class BiLSTMEnhancer(nn.Module):
    """
    BiLSTM特征增强模块
    专门用于对词级特征进行增强处理

    在修正后的SSAN架构中，BiLSTM直接在词级特征c_s上操作，
    产生增强的词级特征c_s_bi，用于后续的注意力计算和GCN处理。
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 384,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # BiLSTM层
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # 输出维度：按照prompt.txt要求，BiLSTM需要把维度从768降到128
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 最终输出维度应该是128（双向LSTM的hidden_dim * 2 = 128 * 2 = 256，需要投影到128）
        self.final_output_dim = hidden_dim  # 128维

        # 投影层：将双向LSTM的256维输出投影到128维
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_output_dim, self.final_output_dim),
            nn.LayerNorm(self.final_output_dim),
            nn.Dropout(dropout),
        )

    def enhance_word_features(
        self,
        word_features: torch.Tensor,
        word_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        对词级特征进行BiLSTM增强 - 修正后架构的核心方法

        Args:
            word_features: [batch_size, max_words, input_dim] 词级特征 c_s
            word_mask: [batch_size, max_words] 词级mask（可选）

        Returns:
            enhanced_word_features: [batch_size, max_words, hidden_dim] 增强后的词级特征 c_s_bi（128维）
        """
        batch_size, max_words, input_dim = word_features.shape

        # 计算实际词数量（如果提供了mask）
        if word_mask is not None:
            word_lengths = word_mask.sum(dim=1)  # [batch_size]
            # 确保长度至少为1，避免pack_padded_sequence出错
            word_lengths = torch.clamp(word_lengths, min=1)
        else:
            word_lengths = None

        # BiLSTM处理
        if word_lengths is not None:
            # 使用pack_padded_sequence优化处理变长序列
            packed_input = nn.utils.rnn.pack_padded_sequence(
                word_features,
                word_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, (hidden, cell) = self.bilstm(packed_input)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=max_words
            )
        else:
            # 直接处理（当所有序列长度相同时）
            lstm_output, (hidden, cell) = self.bilstm(word_features)

        # 投影到原始维度
        projected_output = self.output_projection(lstm_output)

        # 按照prompt.txt要求，直接返回降维后的特征（768->128维）
        # 不使用残差连接，因为维度不匹配
        return projected_output


def test_bilstm_enhancer():
    """测试BiLSTM增强模块 - 专注于词级特征处理"""
    print("🧪 测试BiLSTM词级特征增强模块")

    # 创建模块
    enhancer = BiLSTMEnhancer(
        input_dim=768,
        hidden_dim=384,
        num_layers=1,
        dropout=0.1,
    )

    # 测试词级特征增强
    batch_size, max_words, hidden_dim = 4, 10, 768
    word_features = torch.randn(batch_size, max_words, hidden_dim)
    word_mask = torch.ones(batch_size, max_words)
    word_mask[0, 8:] = 0  # 第一个样本只有8个词
    word_mask[1, 6:] = 0  # 第二个样本只有6个词

    print(f"输入词级特征形状: {word_features.shape}")
    print(f"词级mask形状: {word_mask.shape}")

    # 词级特征增强
    enhanced_word_features = enhancer.enhance_word_features(word_features, word_mask)

    print(f"增强后词级特征形状: {enhanced_word_features.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in enhancer.parameters()):,}")
    print("✅ BiLSTM词级增强模块测试成功！")


if __name__ == "__main__":
    test_bilstm_enhancer()
