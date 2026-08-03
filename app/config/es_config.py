# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv


load_dotenv()



@dataclass
class ESConfig:
    endpoint: str    # MinIO服务地址（含http/https和端口）
    index_name: str




es_config = ESConfig(
    endpoint=os.getenv("ES_URL"),
    index_name=os.getenv("ES_INDEX_NAME"),
)

