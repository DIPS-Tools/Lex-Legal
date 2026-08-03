# 导入Python内置模块
import os
import json
# 导入MinIO官方Python SDK核心类
from minio import Minio
# 项目内部配置与日志
from app.config.minio_config import minio_config
from app.core.logger import logger


"""
GUI 使用：
登录
创建桶
设置权限 --》 Access Policy: custom 

{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": [
                    "*"
                ]
            },
            "Action": [
                "s3:GetObject"
            ],
            "Resource": [
                "arn:aws:s3:::test-bucket/*"
            ]
        }
    ]
}

上传文件
拼接地址：e.g., http://10.22.12.250:9000/test-bucket/dips-q-icon-blue.svg

"""
# 配置参数导入到 .env
# 本次要获取minio的客户端，并且创建好桶和设置桶的访问权限（备用）
#1. 相当于你登录了minio
minio_client = Minio(
    # 端点
    endpoint=minio_config.endpoint,
    # 账号
    access_key=minio_config.access_key,
    # 密码
    secret_key=minio_config.secret_key,
    secure=False # 使用http协议  True https协议
)

# 2. 创建一个桶（区分文件的文件夹，每个项目，创建一个桶）
# 没有才创建桶
bucket_name = minio_config.bucket_name

if not minio_client.bucket_exists(bucket_name):
    # 不存在桶，创建，并设置访问权限
    minio_client.make_bucket(bucket_name)
    # 设置桶的访问权限
    bucket_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"AWS": ["*"]},  # *表示所有匿名用户（S3兼容标识）
            "Action": ["s3:GetObject"],  # 仅授权文件获取/访问操作
            "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
        }]
    }
    minio_client.set_bucket_policy(bucket_name, json.dumps(bucket_policy))

def get_minio_client():
    return minio_client

