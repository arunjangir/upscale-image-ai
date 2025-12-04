# aws_handler.py
import os
import boto3
import requests
import time
import base64
from dotenv import load_dotenv
from database import DatabaseHandler
from whatsapp_handler import WhatsAppHandler

load_dotenv()

class AWSHandler:
    def __init__(self):
        self.ec2 = boto3.client('ec2',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.s3 = boto3.client('s3',
            endpoint_url=os.getenv('R2_ENDPOINT'),
            aws_access_key_id=os.getenv('R2_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('R2_SECRET_KEY'),
            region_name='auto'
        )

        self.bucket          = os.getenv('R2_BUCKET')
        self.ami_id          = os.getenv('AMI_ID')
        self.instance_type   = os.getenv('INSTANCE_TYPE', 'g4dn.large')
        self.iam_role        = os.getenv('IAM_ROLE')
        self.security_group  = os.getenv('SECURITY_GROUP', 'default')
        self.spot_price      = os.getenv('SPOT_PRICE', '0.50')

        # capacity parameters
        self.JOBS_PER_INSTANCE = 60  
        self.MIN_INSTANCES     = 0  

        self.db       = DatabaseHandler()
        self.whatsapp = WhatsAppHandler()

        # sanity check
        required = ['AMI_ID','AWS_ACCESS_KEY','AWS_SECRET_KEY','R2_BUCKET','R2_ACCESS_KEY','R2_SECRET_KEY','R2_ENDPOINT']
        missing  = [v for v in required if not os.getenv(v)]
        if missing:
            raise ValueError(f"Missing env vars: {missing}")

    def get_active_instances(self):
        try:
            resp = self.ec2.describe_instances(
                Filters=[
                  {'Name':'image-id',            'Values':[self.ami_id]},
                  {'Name':'instance-state-name', 'Values':['running','pending']},
                  {'Name':'instance-lifecycle',  'Values':['spot']},
                  {'Name':'tag:Role',            'Values':['realesrgan-processor']}
                ]
            )
            insts = []
            for r in resp['Reservations']:
                for i in r['Instances']:
                    insts.append({
                      'InstanceId':  i['InstanceId'],
                      'State':       i['State']['Name'],
                      'InstanceType':i['InstanceType']
                    })
            return insts
        except Exception as e:
            print(f"Error describing instances: {e}")
            return []

    def get_pending_jobs_count(self):
        try:
            conn = self.db.get_connection()
            cur  = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM jobs WHERE status='pending'")
            cnt = cur.fetchone()[0]
            conn.close()
            return cnt
        except Exception as e:
            print(f"DB error (pending_count): {e}")
            return 0

    def get_processing_jobs_count(self):
        try:
            conn = self.db.get_connection()
            cur  = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM jobs WHERE status='processing'")
            cnt = cur.fetchone()[0]
            conn.close()
            return cnt
        except Exception as e:
            print(f"DB error (processing_count): {e}")
            return 0

    def calculate_required_instances(self):
        pending    = self.get_pending_jobs_count()
        processing = self.get_processing_jobs_count()
        total_jobs = pending + processing

        required = max(0, (total_jobs + self.JOBS_PER_INSTANCE - 1) // self.JOBS_PER_INSTANCE)
        running  = len([i for i in self.get_active_instances() if i['State']=='running'])

        print(f"Jobs: pending={pending}, processing={processing}, total={total_jobs}")
        print(f"Instances: running={running}, required={required}")

        return required, running, pending

    def should_launch_instance(self):
        required, running, pending = self.calculate_required_instances()
        # need more capacity?
        if required > running:
            return True
        # or pending jobs but zero running
        if pending > 0 and running == 0:
            return True
        return False

    def _get_user_data(self):
        script = """#!/bin/bash
echo "Instance started at $(date)" >> /var/log/startup.log
"""
        return base64.b64encode(script.encode()).decode()

    def launch_spot_instance(self):
        try:
            spec = {
              'ImageId':      self.ami_id,
              'InstanceType': self.instance_type,
              'SecurityGroups':[self.security_group],
              'UserData':     self._get_user_data()
            }
            if self.iam_role:
                spec['IamInstanceProfile'] = {'Name': self.iam_role}

            resp = self.ec2.request_spot_instances(
              SpotPrice=self.spot_price,
              InstanceCount=1,
              LaunchSpecification=spec,
              Type='one-time'
            )
            req_id = resp['SpotInstanceRequests'][0]['SpotInstanceRequestId']
            print(f"Spot request {req_id}")

            # wait up to 5m
            for _ in range(30):
                time.sleep(10)
                info = self.ec2.describe_spot_instance_requests(SpotInstanceRequestIds=[req_id])
                state = info['SpotInstanceRequests'][0]['State']
                if state == 'active':
                    inst_id = info['SpotInstanceRequests'][0]['InstanceId']
                    print(f"Fulfilled: {inst_id}")
                    # tag it
                    self.ec2.create_tags(
                      Resources=[inst_id],
                      Tags=[
                        {'Key':'Role',      'Value':'realesrgan-processor'},
                        {'Key':'Name',      'Value':'RealESRGAN-Processor'},
                        {'Key':'LaunchedBy','Value':'AWSHandler'}
                      ]
                    )
                    return inst_id
                elif state in ['cancelled','failed','closed']:
                    print("Spot request failed or closed")
                    return None
                else:
                    print(f"Waiting: {state}")
            print("Timeout waiting for spot")
            return None

        except Exception as e:
            print(f"Error launching spot: {e}")
            return None

    def manage_instances(self):
        if self.should_launch_instance():
            print("Launching new spot instance…")
            inst = self.launch_spot_instance()
            return bool(inst)
        else:
            print("No new instances needed")
            return False

    def upload_to_r2(self, file_url, filename):
        try:
            resp = requests.get(file_url, headers={
                'Authorization': f'Bearer {os.getenv("WHATSAPP_ACCESS_TOKEN")}'
            }, timeout=30)
            if resp.status_code != 200:
                print(f"Download failed: {resp.status_code}")
                return None
            self.s3.put_object(Bucket=self.bucket, Key=filename, Body=resp.content, ContentType='image/jpeg')
            return filename
        except Exception as e:
            print(f"R2 upload error: {e}")
            return None

    def download_from_r2(self, filename):
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=filename)
            return resp['Body'].read()
        except Exception as e:
            print(f"R2 download error: {e}")
            return None
