
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RerankerConfig:
    bge_reranker_model_path: str  # 本地模型路径
    bge_reranker_device: str       # 模型仓库标识
    bge_reranker_fp16: bool    # 是否开启半精度（1=True/0=False）

# 实例化配置对象，和原代码lm_config风格保持一致
reranker_config = RerankerConfig(
    bge_reranker_model_path=os.getenv("BGE_RERANKER_MODEL_PATH"),
    bge_reranker_device=os.getenv("BGE_RERANKER_DEVICE"),
    # 特殊处理：将.env中的1/0转为布尔值，兼容常见的数字/字符串格式
    bge_reranker_fp16=os.getenv("BGE_RERANKER_FP16") in ("1", "True", "true", 1)
)