import os
import boto3
from tools.utils.log import log
from botocore.exceptions import NoCredentialsError
from langchain.tools import tool


@tool("get_s3_image_url", description="Constructs S3 URL for a given image filename")
def get_s3_image_url(filename):
    """Constructs S3 URL for a given image filename"""
    return f"{BUCKET_BASE_URL}{filename}"


@tool("download_from_s3", description="Downloads a file from S3 to a local path")
def download_from_s3(s3_key, local_path):
    """Downloads a file from S3 to a local path"""
    BUCKET_BASE_URL = os.getenv("BUCKET_BASE_URL")
    S3_APPAREL_BUCKET_NAME = os.getenv("APPARELS_S3_BUCKET_NAME")
    print("Downloading from S3 with key:", s3_key)
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
            print(f"File does not exist locally. Downloading from S3: {s3_key}")
            s3.download_file(S3_APPAREL_BUCKET_NAME, s3_key, local_path)
        return f"Downloaded {s3_key} to {local_path}"
    except NoCredentialsError:
        return "AWS credentials not available."
    except Exception as e:
        return f"Error downloading file: {e}"


@tool("list_s3_files", description="Lists files in a given S3 prefix")
def list_s3_files(prefix):
    """Lists files in a given S3 prefix"""
    BUCKET_BASE_URL = os.getenv("BUCKET_BASE_URL")
    S3_APPAREL_BUCKET_NAME = os.getenv("APPARELS_S3_BUCKET_NAME")
    log(f"Listing S3 files with prefix: {prefix}")
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    try:
        response = s3.list_objects_v2(Bucket=S3_APPAREL_BUCKET_NAME, Prefix=prefix)

        if "Contents" in response:
            return [obj["Key"] for obj in response["Contents"]]
        else:
            return []
    except NoCredentialsError:
        return "AWS credentials not available."
    except Exception as e:
        return f"Error listing files: {e}"


@tool(
    "upload_to_s3_tool", description="Uploads a file to S3 and returns the public URL"
)
def upload_to_s3_tool(file_path, s3_key):
    upload_to_s3(file_path, s3_key)


def upload_to_s3(file_path, s3_key):
    """Uploads a file to S3 and returns the public URL"""
    BUCKET_BASE_URL = os.getenv("BUCKET_BASE_URL")
    S3_APPAREL_BUCKET_NAME = os.getenv("APPARELS_S3_BUCKET_NAME")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    if not os.path.exists(file_path):
        log(f"File not found: {file_path}", level="ERROR")
        return "File not found."

    log(
        f"Uploading {file_path} to S3 with key: {s3_key}, bucket: {S3_APPAREL_BUCKET_NAME}"
    )

    try:
        s3.upload_file(
            file_path,
            S3_APPAREL_BUCKET_NAME,
            s3_key,
            ExtraArgs={"ContentType": "image/png"},
        )
        return f"{BUCKET_BASE_URL}{s3_key}"
    except FileNotFoundError:
        log(f"File not found: {file_path}", level="ERROR")
        return "File not found."
    except NoCredentialsError:
        log("AWS credentials not available.", level="ERROR")
        return "AWS credentials not available."
