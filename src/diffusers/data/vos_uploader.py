import os
import boto3
from PIL import Image
from io import BytesIO
import pickle

class VOSUploader:
    def __init__(self, is_public=True):
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
        self.endpoint = 'http://gd17-ai-inner-storegw.api.vip.com'
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            region_name='local',
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            verify=False,
            use_ssl=False)
    def get_bytes(self, img):
        buf = BytesIO()
        img.save(buf, format='png')
        img_bytes = buf.getvalue()
        return img_bytes
    def upload_vos(self, im, img_path, sub_dir=None):
        if sub_dir:
            key = os.path.join(sub_dir, os.path.basename(img_path))
        else:
            key = os.path.basename(img_path)
        im_bytes = self.get_bytes(im)
        self.s3_client.put_object(
                Bucket=self.bucket, Key=key, Body=im_bytes
            )
        full_url = '{}/{}/{}'.format(self.endpoint, self.bucket, key)
        return full_url
    def upload_vos_json(self, json_dict, img_path, sub_dir=None):
        if sub_dir:
            key = os.path.join(sub_dir, os.path.basename(img_path))
        else:
            key = os.path.basename(img_path)
        im_bytes = pickle.dumps((json_dict['points'], json_dict['score']))
        self.s3_client.put_object(
                Bucket=self.bucket, Key=key, Body=im_bytes
            )
        full_url = '{}/{}/{}'.format(self.endpoint, self.bucket, key)
        return full_url

    def display_vos(self, url):
        key = url.split(self.bucket)[-1].lstrip('/')
        s3_response_object = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        object_content = s3_response_object['Body'].read()
        img = Image.open(BytesIO(object_content))
        return img
    
    def get_keypoints_json(self, url):
        key = url.split(self.bucket)[-1].lstrip('/')
        s3_response_object = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        object_content = s3_response_object['Body'].read()
        keypoint_json_data = object_content
        keypoint_json_data = pickle.loads(keypoint_json_data)
    
        return keypoint_json_data
