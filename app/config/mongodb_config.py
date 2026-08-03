# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv


load_dotenv()

@dataclass
class MongoDBConfig:
    endpoint: str    # MinIO服务地址（含http/https和端口）
    database_name: str
    username: str
    password: str




mongodb_config = MongoDBConfig(
    endpoint=os.getenv("MONGO_BASE_URL"),
    database_name=os.getenv("MONGO_DB_NAME"),
    username=os.getenv("MONGO_USERNAME"),
    password=os.getenv("MONGO_PASSWORD"),
)
