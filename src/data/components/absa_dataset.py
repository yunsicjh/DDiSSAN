import torch
from torch.utils.data import Dataset
import json
from transformers import BertTokenizer
import numpy as np


class ABSADataset(Dataset):
    def __init__(
        self, data_path, vocab_dict, bert_model_name="bert-base-uncased", max_len=128
    ):
        self.data = self.load_data(data_path)
        self.vocab_dict = vocab_dict
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.max_len = max_len

        # 验证vocab_dict完整性
        self._validate_vocab_dict()

    def _validate_vocab_dict(self):
        """验证词表字典是否包含所有必要的词表"""
        required_vocabs = ["token", "pos", "dep", "position", "polarity"]
        for vocab_name in required_vocabs:
            if vocab_name not in self.vocab_dict:
                raise KeyError(f"Missing vocabulary: {vocab_name}")

            # 检查词表是否有必要的属性
            vocab = self.vocab_dict[vocab_name]
            if not hasattr(vocab, "stoi") or not hasattr(vocab, "itos"):
                raise AttributeError(
                    f"Vocabulary {vocab_name} missing stoi or itos attributes"
                )

    def load_data(self, data_path):
        with open(data_path, "r") as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return self.process_sample(sample)

    def process_sample(self, sample):
        """处理单个样本，保持一个句子对应一个样本"""
        tokens = sample["token"]
        aspects = sample["aspects"]

        if not aspects:  # 跳过没有方面词的句子
            return None

        # 构造BERT-SPC输入，但这次我们需要处理多个方面词
        sentence_tokens = tokens

        # 为每个方面词构造独立的BERT输入
        processed_aspects = []
        for aspect in aspects:
            aspect_tokens = tokens[aspect["from"] : aspect["to"]]

            # 构造输入序列：先连接文本，再分别tokenize sentence和aspect
            sentence_text = " ".join(sentence_tokens)
            aspect_text = " ".join(aspect_tokens)

            # 使用BERT-SPC格式：[CLS] sentence [SEP] aspect [SEP]
            # 注意：tokenizer会自动添加[CLS]，我们只需要构造 sentence [SEP] aspect [SEP]
            input_text = f"{sentence_text} [SEP] {aspect_text}"

            # 完整编码
            encoded = self.tokenizer(
                input_text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,  # 自动添加[CLS]和[SEP]
            )

            # 计算掩码
            bert_input_ids = encoded["input_ids"].squeeze(0)
            bert_attention_mask = encoded["attention_mask"].squeeze(0)

            # 关键修复：建立原始token到BERT subtoken的映射
            (
                token_to_subtoken_map,
                sentence_mask,
                aspect_mask,
                aspect_subtoken_indices,
            ) = self._build_token_alignment(
                sentence_tokens, aspect_tokens, aspect["from"], aspect["to"]
            )

            # 方面词在原句中的位置掩码（基于原始token）
            aspect_in_sentence_mask = torch.zeros(len(tokens), dtype=torch.bool)
            aspect_in_sentence_mask[aspect["from"] : aspect["to"]] = True

            # 转换polarity为数字索引
            polarity_idx = self.vocab_dict["polarity"][aspect["polarity"]]

            processed_aspects.append(
                {
                    "bert_input_ids": bert_input_ids,
                    "bert_attention_mask": bert_attention_mask,
                    "sentence_mask": sentence_mask,
                    "aspect_mask": aspect_mask,
                    "aspect_in_sentence_mask": aspect_in_sentence_mask,
                    "aspect_subtoken_indices": aspect_subtoken_indices,
                    "token_to_subtoken_map": token_to_subtoken_map,
                    "polarity": polarity_idx,
                }
            )

        # 处理依存关系信息（整个句子共享）
        adj_matrix, dep_rel_matrix, pos_ids = self.process_dependency_info(
            tokens,
            sample.get("head", []),
            sample.get("deprel", []),
            sample.get("pos", []),
        )

        return {
            "aspects": processed_aspects,
            "adj_matrix": adj_matrix,
            "dep_rel_matrix": dep_rel_matrix,
            "pos_ids": pos_ids,
            "num_aspects": len(processed_aspects),
        }

    def _build_token_alignment(
        self, sentence_tokens, aspect_tokens, aspect_from, aspect_to
    ):
        """
        建立原始token到BERT subtoken的对齐映射（简化版本）

        Args:
            sentence_tokens: 原始句子token列表
            aspect_tokens: 方面词token列表
            aspect_from: 方面词在句子中的起始位置
            aspect_to: 方面词在句子中的结束位置

        Returns:
            token_to_subtoken_map: 原始token到BERT subtoken indices的映射
            sentence_mask: 句子在BERT序列中的mask
            aspect_mask: 方面词在BERT序列中的mask
            aspect_subtoken_indices: 方面词对应的所有subtoken indices
        """
        sentence_text = " ".join(sentence_tokens)
        aspect_text = " ".join(aspect_tokens)

        # 使用更简单的方法进行token对齐
        sentence_encoding = self.tokenizer(
            sentence_text, add_special_tokens=False, return_tensors="pt"
        )

        aspect_encoding = self.tokenizer(
            aspect_text, add_special_tokens=False, return_tensors="pt"
        )

        # 处理单个token的情况
        if sentence_encoding["input_ids"].dim() == 1:
            sentence_subtokens = sentence_encoding["input_ids"].tolist()
        else:
            sentence_subtokens = sentence_encoding["input_ids"].squeeze().tolist()

        if aspect_encoding["input_ids"].dim() == 1:
            aspect_subtokens = aspect_encoding["input_ids"].tolist()
        else:
            aspect_subtokens = aspect_encoding["input_ids"].squeeze().tolist()

        # 确保是list
        if isinstance(sentence_subtokens, int):
            sentence_subtokens = [sentence_subtokens]
        if isinstance(aspect_subtokens, int):
            aspect_subtokens = [aspect_subtokens]

        # 计算各部分在完整序列中的位置
        sentence_start = 1  # 跳过[CLS]
        sentence_end = sentence_start + len(sentence_subtokens)
        aspect_start = sentence_end + 1  # +1 for [SEP]
        aspect_end = aspect_start + len(aspect_subtokens)

        # 创建masks
        max_len = min(self.max_len, aspect_end + 2)  # +2 for final tokens
        sentence_mask = torch.zeros(max_len, dtype=torch.bool)
        aspect_mask = torch.zeros(max_len, dtype=torch.bool)

        if sentence_start < max_len and sentence_end <= max_len:
            sentence_mask[sentence_start:sentence_end] = True
        if aspect_start < max_len and aspect_end <= max_len:
            aspect_mask[aspect_start:aspect_end] = True

        # 正确的映射：建立原始token到句子部分subtoken的映射
        token_to_subtoken_map = {}

        # 更精确的token对齐：逐个token编码然后对齐
        sentence_subtoken_offset = 0
        for token_idx, token in enumerate(sentence_tokens):
            # 单独编码每个token以获取准确的subtoken数量
            token_encoding = self.tokenizer.encode(token, add_special_tokens=False)
            token_subtoken_count = len(token_encoding)

            # 记录该token对应的subtoken indices（在句子部分中的相对位置）
            subtoken_indices = []
            for i in range(token_subtoken_count):
                if sentence_subtoken_offset + i < len(sentence_subtokens):
                    # 转换为在完整BERT序列中的绝对位置
                    subtoken_indices.append(
                        sentence_start + sentence_subtoken_offset + i
                    )

            token_to_subtoken_map[token_idx] = subtoken_indices
            sentence_subtoken_offset += token_subtoken_count

        # 获取方面词的subtoken indices（基于原始句子中的位置）
        aspect_subtoken_indices = []
        for token_idx in range(aspect_from, aspect_to):
            if token_idx in token_to_subtoken_map:
                aspect_subtoken_indices.extend(token_to_subtoken_map[token_idx])

        return (
            token_to_subtoken_map,
            sentence_mask,
            aspect_mask,
            aspect_subtoken_indices,
        )

    def process_dependency_info(self, tokens, heads, dep_rels, pos_tags):
        """
        处理依存关系信息，构建邻接矩阵、依存关系矩阵和POS标签ID
        """
        seq_len = len(tokens)

        # 构建邻接矩阵和依存关系矩阵
        adj_matrix, dep_rel_matrix = self.build_adjacency_matrix(
            heads, dep_rels, seq_len
        )

        # 编码POS标签
        pos_ids = self.encode_sequence(pos_tags, self.vocab_dict["pos"])

        return adj_matrix, dep_rel_matrix, pos_ids

    def build_bert_spc_input(self, sentence_tokens, aspect):
        """
        构建BERT-SPC输入格式: [CLS] sentence [SEP] aspect [SEP]
        """
        # 提取aspect term
        aspect_from = aspect["from"]
        aspect_to = aspect["to"]
        aspect_tokens = sentence_tokens[aspect_from:aspect_to]

        # 构建输入序列：[CLS] + sentence + [SEP] + aspect + [SEP]
        input_text = " ".join(sentence_tokens) + " [SEP] " + " ".join(aspect_tokens)

        # BERT编码
        encoding = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=False,
        )

        # 创建mask来标识sentence和aspect的位置
        # 重新分析token位置来创建精确的mask
        sentence_text = " ".join(sentence_tokens)
        aspect_text = " ".join(aspect_tokens)

        # 分别编码来获得准确的token数量
        sentence_encoding = self.tokenizer.encode(
            sentence_text, add_special_tokens=False
        )
        aspect_encoding = self.tokenizer.encode(aspect_text, add_special_tokens=False)

        # 构建mask
        total_len = encoding["input_ids"].shape[1]
        sentence_mask = torch.zeros(total_len, dtype=torch.bool)
        aspect_mask = torch.zeros(total_len, dtype=torch.bool)

        # sentence部分: [CLS] + sentence_tokens + [SEP]
        sentence_start = 1  # 跳过[CLS]
        sentence_end = sentence_start + len(sentence_encoding)
        sentence_mask[sentence_start:sentence_end] = True

        # aspect部分: sentence + [SEP] + aspect_tokens + [SEP]
        aspect_start = sentence_end + 1  # 跳过中间的[SEP]
        aspect_end = aspect_start + len(aspect_encoding)
        if aspect_end < total_len:  # 确保不超出范围
            aspect_mask[aspect_start:aspect_end] = True

        return encoding, sentence_mask, aspect_mask

    def build_adjacency_matrix(self, heads, dep_rels, seq_len):
        """根据依存关系构建邻接矩阵和依存关系矩阵"""
        # 邻接矩阵：1表示有边，0表示无边
        adj_matrix = np.zeros((seq_len, seq_len), dtype=np.float32)

        # 依存关系矩阵：存储依存关系的vocab id
        dep_rel_matrix = np.zeros((seq_len, seq_len), dtype=np.long)

        for i, (head, dep_rel) in enumerate(zip(heads, dep_rels)):
            if head > 0:  # head=0表示ROOT，跳过
                head_idx = head - 1  # 转换为0-based索引

                # 构建双向边（无向图）
                adj_matrix[i][head_idx] = 1.0
                adj_matrix[head_idx][i] = 1.0

                # 存储依存关系
                dep_rel_id = self.vocab_dict["dep"][dep_rel]
                dep_rel_matrix[i][head_idx] = dep_rel_id
                dep_rel_matrix[head_idx][i] = dep_rel_id

        # 添加自环
        for i in range(seq_len):
            adj_matrix[i][i] = 1.0
            # 自环使用特殊的依存关系标记，如果没有使用默认值0
            try:
                self_dep_id = self.vocab_dict["dep"]["<self>"]
            except:
                self_dep_id = 0  # 使用默认值
            dep_rel_matrix[i][i] = self_dep_id

        return torch.FloatTensor(adj_matrix), torch.LongTensor(dep_rel_matrix)

    def encode_sequence(self, sequence, vocab):
        """将序列编码为索引"""
        return torch.LongTensor(
            [vocab.stoi.get(item, vocab.unk_index) for item in sequence]
        )


def collate_fn(batch):
    """修改后的collate函数，支持多方面词"""
    # 过滤掉None样本
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # 展开所有方面词
    all_aspects = []
    batch_info = []  # 记录每个样本的方面词数量和依存信息

    for item in batch:
        start_idx = len(all_aspects)
        all_aspects.extend(item["aspects"])
        end_idx = len(all_aspects)

        batch_info.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "adj_matrix": item["adj_matrix"],
                "dep_rel_matrix": item["dep_rel_matrix"],
                "pos_ids": item["pos_ids"],
                "num_aspects": item["num_aspects"],
            }
        )

    if not all_aspects:
        return None

    # 处理BERT相关数据
    # 处理BERT相关数据 - 需要padding到统一长度
    bert_input_ids = torch.stack([item["bert_input_ids"] for item in all_aspects])
    bert_attention_mask = torch.stack(
        [item["bert_attention_mask"] for item in all_aspects]
    )

    # 对sentence_mask和aspect_mask进行padding
    max_bert_len = bert_input_ids.shape[1]  # 从input_ids获取最大长度

    sentence_masks = []
    aspect_masks = []

    for item in all_aspects:
        # Padding sentence_mask
        sentence_mask = item["sentence_mask"]
        if sentence_mask.shape[0] < max_bert_len:
            padded_sentence = torch.zeros(max_bert_len, dtype=sentence_mask.dtype)
            padded_sentence[: sentence_mask.shape[0]] = sentence_mask
        else:
            padded_sentence = sentence_mask[:max_bert_len]
        sentence_masks.append(padded_sentence)

        # Padding aspect_mask
        aspect_mask = item["aspect_mask"]
        if aspect_mask.shape[0] < max_bert_len:
            padded_aspect = torch.zeros(max_bert_len, dtype=aspect_mask.dtype)
            padded_aspect[: aspect_mask.shape[0]] = aspect_mask
        else:
            padded_aspect = aspect_mask[:max_bert_len]
        aspect_masks.append(padded_aspect)

    sentence_masks = torch.stack(sentence_masks)
    aspect_masks = torch.stack(aspect_masks)

    # 新增：处理subtoken映射信息
    aspect_subtoken_indices = [item["aspect_subtoken_indices"] for item in all_aspects]
    token_to_subtoken_maps = [item["token_to_subtoken_map"] for item in all_aspects]

    polarities = torch.tensor([item["polarity"] for item in all_aspects])

    # 处理依存关系数据（需要padding到统一长度）
    # 确保max_seq_len考虑所有相关数据的最大长度
    max_seq_len = max(
        max(info["adj_matrix"].shape[0] for info in batch_info),  # adj_matrix大小
        max(info["pos_ids"].shape[0] for info in batch_info),  # pos_ids长度
        max(
            item["aspect_in_sentence_mask"].shape[0] for item in all_aspects
        ),  # aspect_mask长度
    )

    padded_adj_matrices = []
    padded_dep_rel_matrices = []
    padded_pos_ids = []
    padded_aspect_masks = []

    for item in all_aspects:
        # 获取对应的依存信息
        for info in batch_info:
            if info["start_idx"] <= len(padded_adj_matrices) < info["end_idx"]:
                adj_matrix = info["adj_matrix"]
                dep_rel_matrix = info["dep_rel_matrix"]
                pos_ids = info["pos_ids"]
                break

        # Padding
        padded_adj = torch.zeros(max_seq_len, max_seq_len)
        padded_adj[: adj_matrix.shape[0], : adj_matrix.shape[1]] = adj_matrix
        padded_adj_matrices.append(padded_adj)

        padded_dep = torch.zeros(max_seq_len, max_seq_len, dtype=torch.long)
        padded_dep[: dep_rel_matrix.shape[0], : dep_rel_matrix.shape[1]] = (
            dep_rel_matrix
        )
        padded_dep_rel_matrices.append(padded_dep)

        padded_pos = torch.zeros(max_seq_len, dtype=torch.long)
        padded_pos[: len(pos_ids)] = pos_ids
        padded_pos_ids.append(padded_pos)

        # Padding aspect_in_sentence_mask
        aspect_mask = item["aspect_in_sentence_mask"]
        padded_aspect_mask = torch.zeros(max_seq_len, dtype=torch.bool)
        padded_aspect_mask[: len(aspect_mask)] = aspect_mask
        padded_aspect_masks.append(padded_aspect_mask)

    return {
        "bert_input_ids": bert_input_ids,
        "bert_attention_mask": bert_attention_mask,
        "sentence_mask": sentence_masks,
        "aspect_mask": aspect_masks,
        "adj_matrix": torch.stack(padded_adj_matrices),
        "dep_rel_matrix": torch.stack(padded_dep_rel_matrices),
        "pos_ids": torch.stack(padded_pos_ids),
        "aspect_in_sentence_mask": torch.stack(padded_aspect_masks),
        "aspect_subtoken_indices": aspect_subtoken_indices,
        "token_to_subtoken_maps": token_to_subtoken_maps,
        "polarity": polarities,
        "batch_info": batch_info,
    }
