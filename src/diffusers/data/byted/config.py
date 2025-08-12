from kmsv2 import KmsClient, Region

kc = KmsClient(region=Region.Row)
KMS_SOURCE = "creativeimagesolution"

LabCVAppKey = kc.get_secret(f"{KMS_SOURCE}/labcv_algo_vproxy/app_key")
LabCVAppSecret = kc.get_secret(f"{KMS_SOURCE}/labcv_algo_vproxy/app_secret")
