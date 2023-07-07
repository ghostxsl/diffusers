import os
import boto3
import torch
from PIL import Image
from io import BytesIO
import pickle
import traceback

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
#关闭安全请求警告
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


__all__ = ['VOSClient']


class VOSClient:
    def __init__(self, bucket='public'):
        info_dict = {
            
        }
        assert bucket in info_dict.keys()

        self.bucket = info_dict[bucket]['bucket']
        self.access_key = info_dict[bucket]['access_key']
        self.secret_key = info_dict[bucket]['secret_key']
        self.endpoint = info_dict[bucket]['endpoint']

        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            region_name='local',
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            verify=False,
            # use_ssl=False,
        )

        self.custom_headers = {'x-vip-force-rewrite', 'true'}

    @staticmethod
    def get_pil_bytes(img, format='JPEG', quality=90):
        # format: JPEG, PNG, GIF
        buf = BytesIO()
        img.save(buf, format=format, quality=quality)
        img_bytes = buf.getvalue()
        return img_bytes

    def upload_vos_bytes(self, file_bytes, s3_path):
        try:
            self.s3_client.put_object(
                    Body=file_bytes, Bucket=self.bucket, Key=s3_path,
                )
        except Exception as e:
            print(f"upload error: {s3_path} [{e}]")
            raise Exception(traceback.format_exc())
        return f"s3://{self.bucket}/{s3_path}"

    def upload_vos_pil(self, img, s3_path, format='JPEG', quality=90):
        # format: JPEG, PNG, GIF
        img_bytes = self.get_pil_bytes(img, format, quality=quality)
        return self.upload_vos_bytes(img_bytes, s3_path)

    def upload_vos_pkl(self, obj, s3_path):
        pkl_bytes = pickle.dumps(obj)
        return self.upload_vos_bytes(pkl_bytes, s3_path)

    def upload_vos_pt(self, obj, s3_path):
        buf = BytesIO()
        torch.save(obj, buf)
        torch_bytes = buf.getvalue()
        return self.upload_vos_bytes(torch_bytes, s3_path)

    def download_vos_bytes(self, s3_path):
        try:
            s3_response_object = self.s3_client.get_object(Bucket=self.bucket, Key=s3_path)
        except Exception as e:
            print(f"download error: {s3_path} [{e}]")
            raise Exception(traceback.format_exc())
        return s3_response_object['Body'].read()

    def download_vos_pil(self, s3_path):
        try:
            content = self.download_vos_bytes(s3_path)
        except Exception as e:
            print(f"download error: {s3_path} [{e}]")
            raise Exception(traceback.format_exc())
        return Image.open(BytesIO(content))

    def download_vos_pkl(self, s3_path):
        try:
            content = self.download_vos_bytes(s3_path)
        except Exception as e:
            print(f"download error: {s3_path} [{e}]")
            raise Exception(traceback.format_exc())
        return pickle.loads(content)

    def download_vos_pt(self, s3_path):
        try:
            content = self.download_vos_bytes(s3_path)
        except Exception as e:
            print(f"download error: {s3_path} [{e}]")
            raise Exception(traceback.format_exc())
        return torch.load(BytesIO(content), weights_only=True)

    def delete_vos_file(self, s3_path):
        s3_response_object = self.s3_client.delete_object(Bucket=self.bucket, Key=s3_path)
        return s3_response_object['DeleteMarker']

    def list_vos_files(self, s3_path):
        # 'Contents' in s3_response_object
        s3_response_object = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=s3_path)
        return s3_response_object

    def _upload_file(self, file_path, s3_path):
        self.s3_client.upload_file(file_path, self.bucket, s3_path)

    def _download_file(self, file_path, s3_path):
        self.s3_client.download_file(self.bucket, file_path, s3_path)
