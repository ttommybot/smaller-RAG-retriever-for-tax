import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDistillationPipeline:
    """
    综合RocketQA + GPL + DPR + Hofstätter的数据蒸馏Pipeline

    用途：清洗大检索模型生成的{query, chunks}对数据，生成高质量的训练数据

    流程：
    1. 检索候选（top-200）
    2. 用cross-encoder打分
    3. 选择正样本（基础 + 数据增强）
    4. 挖掘困难负样本（去噪）
    5. 添加简单负样本（平衡难度）
    6. 质量过滤（长度、分数、去重）
    """

    def __init__(self, config: Dict = None):
        """
        初始化Pipeline

        Args:
            config: 超参数配置字典
        """
        self.config = config or self._default_config()
        self.stats = defaultdict(int)

    @staticmethod
    def _default_config() -> Dict:
        """
        默认配置（基于论文最佳实践）

        参数来源：
        - ANCE: retrieval_top_k
        - Hofstätter: positive_threshold, teacher_min_score
        - RocketQA: hard_negative_start/end, augmentation_threshold
        - DPR: negatives_per_query, doc长度
        - GPL: dedup_threshold, query长度
        """
        return {
            # 检索阶段
            "retrieval_top_k": 200,              # ANCE建议

            # 正样本采样
            "positive_threshold": 0.6,           # Hofstätter: 基础正样本最低分
            "augmentation_threshold": 0.8,       # RocketQA: 数据增强阈值
            "max_positives_per_query": 3,        # GPL: 每个query最多正样本

            # 负样本采样（RocketQA的去噪方法）
            "hard_negative_start": 10,           # 跳过top-10（避免假负样本）
            "hard_negative_end": 60,             # 到第60名
            "hard_negative_range": (0.3, 0.7),   # 困难负样本分数区间
            "negatives_per_query": 7,            # DPR建议
            "easy_negatives_per_query": 2,       # 简单负样本数

            # 质量控制
            "teacher_min_score": 0.3,            # Hofstätter: 去噪阈值
            "min_doc_length": 50,                # DPR: 最短文档
            "max_doc_length": 512,               # DPR: 最长文档
            "dedup_threshold": 0.9,              # GPL: 去重相似度
            "min_query_length": 5,               # GPL: 最短query

            # 训练参数
            "batch_size": 32,
            "use_cross_batch_negatives": True,   # RocketQA
        }

    def process_query(self,
                     query: str,
                     candidates: List[Dict],
                     scores: List[float]) -> Optional[Dict]:
        """
        处理单个query的完整流程

        Args:
            query: 查询文本
            candidates: 候选文档列表，每个元素为 {"id": str, "text": str}
            scores: 对应的cross-encoder分数

        Returns:
            训练样本字典，或None（如果过滤掉）

        格式：
        {
            "query": str,
            "positives": [(doc_dict, score), ...],
            "negatives": [(doc_dict, score), ...]
        }
        """

        # 步骤1：基础检查
        if not self._check_query_quality(query):
            self.stats["filtered_query_quality"] += 1
            return None

        # 步骤2：排序候选
        ranked = self._rank_candidates(candidates, scores)

        # 步骤3：选择正样本
        positives = self._select_positives(ranked)
        if not positives:
            self.stats["filtered_no_positives"] += 1
            return None

        # 步骤4：挖掘困难负样本（RocketQA去噪）
        hard_negatives = self._mine_hard_negatives(ranked, positives)

        # 步骤5：添加简单负样本
        easy_negatives = self._mine_easy_negatives(ranked, positives, hard_negatives)

        all_negatives = hard_negatives + easy_negatives

        # 步骤6：质量过滤
        sample = self._quality_filter(query, positives, all_negatives)

        if sample:
            self.stats["valid_samples"] += 1
        else:
            self.stats["filtered_quality"] += 1

        return sample

    def _check_query_quality(self, query: str) -> bool:
        """检查query质量"""
        if len(query) < self.config["min_query_length"]:
            return False
        if len(query) > 1000:  # 过长的query
            return False
        return True

    def _rank_candidates(self,
                        candidates: List[Dict],
                        scores: List[float]) -> List[Tuple[Dict, float]]:
        """按分数排序候选"""
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked

    def _select_positives(self,
                         ranked: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        """
        选择正样本

        策略：
        1. 基础正样本：score > positive_threshold的top-1
        2. 数据增强：score > augmentation_threshold的额外正样本（RocketQA）
        """
        positives = []

        for doc, score in ranked:
            # 基础正样本
            if len(positives) == 0 and score > self.config["positive_threshold"]:
                positives.append((doc, score))
                self.stats["basic_positives"] += 1

            # 数据增强正样本（RocketQA方法）
            elif (len(positives) < self.config["max_positives_per_query"] and
                  score > self.config["augmentation_threshold"]):
                positives.append((doc, score))
                self.stats["augmented_positives"] += 1
            else:
                break

        return positives

    def _mine_hard_negatives(self,
                            ranked: List[Tuple[Dict, float]],
                            positives: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        """
        挖掘困难负样本（RocketQA的核心去噪方法）

        关键思想：
        1. 跳过top-K（可能是假负样本 - 实际相关但没标注）
        2. 从中间区间选择（分数在hard_negative_range内）
        3. 这样避免了假负样本污染训练数据

        假负样本问题：
        - bi-encoder检索的top结果中，可能有实际相关但没标注的文档
        - 这些文档会误导student模型学习
        - 解决：跳过top-10，从第10-60名选择，这些更可能是真正的负样本
        """
        positive_ids = {doc["id"] for doc, _ in positives}

        # 跳过top-K
        start = self.config["hard_negative_start"]
        end = self.config["hard_negative_end"]
        candidates = ranked[start:end]

        # 分数过滤
        min_score, max_score = self.config["hard_negative_range"]
        hard_negatives = [
            (doc, score) for doc, score in candidates
            if (min_score < score < max_score and
                doc["id"] not in positive_ids)
        ]

        # 采样
        n = self.config["negatives_per_query"]
        if len(hard_negatives) > n:
            sampled = random.sample(hard_negatives, n)
        else:
            sampled = hard_negatives

        self.stats["hard_negatives_mined"] += len(sampled)
        return sampled

    def _mine_easy_negatives(self,
                            ranked: List[Tuple[Dict, float]],
                            positives: List[Tuple[Dict, float]],
                            hard_negatives: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        """
        挖掘简单负样本（DPR方法）

        目的：平衡难度，避免过拟合

        为什么需要简单负样本？
        - 只用困难负样本，student容易过拟合
        - 混合困难和简单负样本，能更好地学习排序
        """
        positive_ids = {doc["id"] for doc, _ in positives}
        hard_neg_ids = {doc["id"] for doc, _ in hard_negatives}

        # 从低分区域采样
        min_score = self.config["hard_negative_range"][0]
        easy_candidates = [
            (doc, score) for doc, score in ranked
            if (score < min_score and
                doc["id"] not in positive_ids and
                doc["id"] not in hard_neg_ids)
        ]

        # 采样
        n = self.config["easy_negatives_per_query"]
        if len(easy_candidates) > n:
            sampled = random.sample(easy_candidates, n)
        else:
            sampled = easy_candidates

        self.stats["easy_negatives_mined"] += len(sampled)
        return sampled

    def _quality_filter(self,
                       query: str,
                       positives: List[Tuple[Dict, float]],
                       negatives: List[Tuple[Dict, float]]) -> Optional[Dict]:
        """
        质量过滤（综合DPR + GPL + Hofstätter）

        过滤标准：
        1. 文档长度
        2. Teacher最低分（去噪）
        3. 必须有正样本
        4. 去重
        """

        # 过滤1：文档长度
        def valid_length(doc):
            length = len(doc["text"])
            return (self.config["min_doc_length"] <= length <=
                   self.config["max_doc_length"])

        positives = [(doc, score) for doc, score in positives
                     if valid_length(doc)]
        negatives = [(doc, score) for doc, score in negatives
                     if valid_length(doc)]

        # 过滤2：Teacher最低分（Hofstätter的去噪方法）
        min_score = self.config["teacher_min_score"]
        positives = [(doc, score) for doc, score in positives
                     if score > min_score]

        # 过滤3：必须有正样本
        if len(positives) == 0:
            return None

        # 过滤4：去重（基于文本相似度）
        positives = self._deduplicate(positives)
        negatives = self._deduplicate(negatives)

        return {
            "query": query,
            "positives": positives,
            "negatives": negatives
        }

    def _deduplicate(self,
                    docs_with_scores: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        """
        去重：移除高度相似的文档（GPL方法）

        为什么需要去重？
        - 高度相似的文档会浪费训练数据
        - 确保负样本多样性

        使用简单的文本相似度（可以改进为embedding相似度）
        """
        if len(docs_with_scores) <= 1:
            return docs_with_scores

        # 简单实现：基于文本长度和内容的快速去重
        keep = []
        kept_texts = []

        for doc, score in docs_with_scores:
            text = doc["text"]

            # 检查是否与已保留的文档过于相似
            is_duplicate = False
            for kept_text in kept_texts:
                # 简单的相似度计算（可以改进）
                similarity = self._text_similarity(text, kept_text)
                if similarity > self.config["dedup_threshold"]:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep.append((doc, score))
                kept_texts.append(text)

        self.stats["dedup_removed"] += len(docs_with_scores) - len(keep)
        return keep

    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """
        简单的文本相似度计算

        实际应该用embedding相似度，这里用快速的方法
        使用Jaccard相似度（基于字符n-gram）
        """
        # 如果完全相同
        if text1 == text2:
            return 1.0

        # 计算Jaccard相似度（基于字符n-gram）
        n = 3
        set1 = set([text1[i:i+n] for i in range(len(text1)-n+1)])
        set2 = set([text2[i:i+n] for i in range(len(text2)-n+1)])

        if len(set1) == 0 or len(set2) == 0:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def run(self,
           query_candidate_pairs: List[Dict]) -> List[Dict]:
        """
        运行完整pipeline

        Args:
            query_candidate_pairs: 列表，每个元素为：
                {
                    "query": str,
                    "candidates": [{"id": str, "text": str}, ...],
                    "scores": [float, ...]
                }

        Returns:
            训练样本列表
        """
        training_data = []

        logger.info(f"Processing {len(query_candidate_pairs)} queries...")

        for item in tqdm(query_candidate_pairs):
            query = item["query"]
            candidates = item["candidates"]
            scores = item["scores"]

            sample = self.process_query(query, candidates, scores)

            if sample is not None:
                training_data.append(sample)

        self._log_statistics(len(query_candidate_pairs))

        return training_data

    def _log_statistics(self, total_queries: int):
        """打印统计信息"""
        logger.info("\n" + "="*50)
        logger.info("Pipeline Statistics")
        logger.info("="*50)
        logger.info(f"Total queries: {total_queries}")
        logger.info(f"Valid samples: {self.stats['valid_samples']}")
        logger.info(f"Filtered (query quality): {self.stats['filtered_query_quality']}")
        logger.info(f"Filtered (no positives): {self.stats['filtered_no_positives']}")
        logger.info(f"Filtered (quality): {self.stats['filtered_quality']}")
        logger.info(f"Basic positives: {self.stats['basic_positives']}")
        logger.info(f"Augmented positives: {self.stats['augmented_positives']}")
        logger.info(f"Hard negatives mined: {self.stats['hard_negatives_mined']}")
        logger.info(f"Easy negatives mined: {self.stats['easy_negatives_mined']}")
        logger.info(f"Dedup removed: {self.stats['dedup_removed']}")
        logger.info("="*50 + "\n")

    def save_training_data(self,
                          training_data: List[Dict],
                          output_path: str):
        """
        保存训练数据为JSONL格式

        格式：每行一个训练样本
        {
            "query": str,
            "positive": str,
            "negative": str,
            "pos_score": float,
            "neg_score": float,
            "margin": float
        }

        Args:
            training_data: 训练样本列表
            output_path: 输出文件路径
        """
        logger.info(f"Saving training data to {output_path}...")

        total_lines = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in training_data:
                query = sample["query"]

                # 为每个正样本配对所有负样本
                for pos_doc, pos_score in sample["positives"]:
                    for neg_doc, neg_score in sample["negatives"]:
                        line = {
                            "query": query,
                            "positive": pos_doc["text"],
                            "negative": neg_doc["text"],
                            "pos_score": float(pos_score),
                            "neg_score": float(neg_score),
                            "margin": float(pos_score - neg_score)  # 用于Margin-MSE loss
                        }
                        f.write(json.dumps(line, ensure_ascii=False) + '\n')
                        total_lines += 1

        logger.info(f"Saved {total_lines} training lines from {len(training_data)} samples")
