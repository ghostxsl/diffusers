import os
import boto3
from PIL import Image
from io import BytesIO
import pickle

__all__ = ['VOSClient']


class VOSClient:
    def __init__(
            self,
            is_public=True,
            endpoint="http://gd17-ai-inner-storegw.api.vip.com",
    ):
        if is_public:
            # public
            self.bucket = 'llm-cv-public'
            self.access_key = "AKIAYY4HQFBNAOZCYXZQ"
            self.secret_key = "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"
        else:
            # personal
            self.bucket = 'llm-cv-personal'
            self.access_key = ""
            self.secret_key = ""

        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint,
            region_name='local',
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            verify=False,
            use_ssl=False)

        self.custom_headers = {'x-vip-force-rewrite', 'true'}

    @staticmethod
    def get_pil_bytes(img, format='PNG', quality=90):
        # format: JPEG, PNG, GIF
        buf = BytesIO()
        img.save(buf, format=format, quality=quality)
        img_bytes = buf.getvalue()
        return img_bytes

    def upload_vos_bytes(self, file_bytes, s3_path):
        self.s3_client.put_object(
                Body=file_bytes, Bucket=self.bucket, Key=s3_path,
            )
        return f"s3://{self.bucket}/{s3_path}"

    def upload_vos_pil(self, img, s3_path, format='PNG', quality=90):
        img_bytes = self.get_pil_bytes(img, format, quality=quality)
        return self.upload_vos_bytes(img_bytes, s3_path)

    def upload_vos_pkl(self, obj, s3_path):
        pkl_bytes = pickle.dumps(obj)
        return self.upload_vos_bytes(pkl_bytes, s3_path)

    def download_vos_bytes(self, s3_path):
        s3_response_object = self.s3_client.get_object(Bucket=self.bucket, Key=s3_path)
        return s3_response_object['Body'].read()

    def download_vos_pil(self, s3_path):
        content = self.download_vos_bytes(s3_path)
        img = Image.open(BytesIO(content))
        return img

    def download_vos_pkl(self, s3_path):
        content = self.download_vos_bytes(s3_path)
        return pickle.loads(content)

    def delete_vos_file(self, s3_path):
        s3_response_object = self.s3_client.delete_object(Bucket=self.bucket, Key=s3_path)
        return s3_response_object['DeleteMarker']

    def list_vos_files(self, s3_path):
        s3_response_object = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=s3_path)
        return s3_response_object['Contents']
