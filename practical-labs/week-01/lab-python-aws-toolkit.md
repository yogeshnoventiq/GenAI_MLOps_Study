# Lab 1.1: Python AWS Automation Toolkit
*Build essential Python tools for AWS infrastructure management*

## 🎯 **Objectives**
- Master Python basics with AWS SDK (Boto3)
- Create reusable infrastructure automation scripts
- Build monitoring and management utilities

---

## 📋 **Prerequisites**
```bash
# Install Python 3.8+
python3 --version

# Create virtual environment
python3 -m venv mlops-env
source mlops-env/bin/activate  # Linux/Mac
# mlops-env\Scripts\activate  # Windows

# Install dependencies
pip install boto3 pandas matplotlib jupyter
```

---

## 🛠️ **Lab Implementation**

### **Step 1: AWS Configuration**
```bash
# Configure AWS CLI
aws configure
# Enter your Access Key ID, Secret Access Key, Region, Output format

# Verify configuration
aws sts get-caller-identity
```

### **Step 2: EC2 Management Script**
Create `ec2_manager.py`:

```python
import boto3
import pandas as pd
from datetime import datetime
import json

class EC2Manager:
    def __init__(self, region='us-east-1'):
        self.ec2 = boto3.client('ec2', region_name=region)
        self.region = region
    
    def list_instances(self):
        """List all EC2 instances with details"""
        response = self.ec2.describe_instances()
        instances = []
        
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instances.append({
                    'InstanceId': instance['InstanceId'],
                    'InstanceType': instance['InstanceType'],
                    'State': instance['State']['Name'],
                    'LaunchTime': instance['LaunchTime'],
                    'PublicIP': instance.get('PublicIpAddress', 'N/A'),
                    'PrivateIP': instance.get('PrivateIpAddress', 'N/A')
                })
        
        return pd.DataFrame(instances)
    
    def start_instances(self, instance_ids):
        """Start EC2 instances"""
        try:
            response = self.ec2.start_instances(InstanceIds=instance_ids)
            print(f"Starting instances: {instance_ids}")
            return response
        except Exception as e:
            print(f"Error starting instances: {e}")
    
    def stop_instances(self, instance_ids):
        """Stop EC2 instances"""
        try:
            response = self.ec2.stop_instances(InstanceIds=instance_ids)
            print(f"Stopping instances: {instance_ids}")
            return response
        except Exception as e:
            print(f"Error stopping instances: {e}")
    
    def get_instance_metrics(self, instance_id, hours=24):
        """Get CloudWatch metrics for instance"""
        cloudwatch = boto3.client('cloudwatch', region_name=self.region)
        
        end_time = datetime.utcnow()
        start_time = end_time - pd.Timedelta(hours=hours)
        
        metrics = cloudwatch.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName='CPUUtilization',
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,
            Statistics=['Average', 'Maximum']
        )
        
        return pd.DataFrame(metrics['Datapoints'])

# Usage example
if __name__ == "__main__":
    ec2_mgr = EC2Manager()
    
    # List all instances
    instances_df = ec2_mgr.list_instances()
    print("Current EC2 Instances:")
    print(instances_df)
    
    # Get metrics for first instance (if exists)
    if not instances_df.empty:
        instance_id = instances_df.iloc[0]['InstanceId']
        metrics_df = ec2_mgr.get_instance_metrics(instance_id)
        print(f"\nMetrics for {instance_id}:")
        print(metrics_df)
```

### **Step 3: S3 Data Processing Utility**
Create `s3_processor.py`:

```python
import boto3
import pandas as pd
import json
from datetime import datetime
import os

class S3Processor:
    def __init__(self, region='us-east-1'):
        self.s3 = boto3.client('s3', region_name=region)
        self.region = region
    
    def list_buckets(self):
        """List all S3 buckets"""
        response = self.s3.list_buckets()
        buckets = []
        
        for bucket in response['Buckets']:
            buckets.append({
                'Name': bucket['Name'],
                'CreationDate': bucket['CreationDate']
            })
        
        return pd.DataFrame(buckets)
    
    def analyze_bucket_contents(self, bucket_name):
        """Analyze contents of S3 bucket"""
        try:
            response = self.s3.list_objects_v2(Bucket=bucket_name)
            
            if 'Contents' not in response:
                return pd.DataFrame()
            
            objects = []
            total_size = 0
            
            for obj in response['Contents']:
                size_mb = obj['Size'] / (1024 * 1024)
                total_size += size_mb
                
                objects.append({
                    'Key': obj['Key'],
                    'Size_MB': round(size_mb, 2),
                    'LastModified': obj['LastModified'],
                    'StorageClass': obj.get('StorageClass', 'STANDARD')
                })
            
            df = pd.DataFrame(objects)
            
            # Add summary statistics
            print(f"Bucket: {bucket_name}")
            print(f"Total Objects: {len(objects)}")
            print(f"Total Size: {round(total_size, 2)} MB")
            print(f"Average Size: {round(total_size/len(objects), 2)} MB")
            
            return df
            
        except Exception as e:
            print(f"Error analyzing bucket {bucket_name}: {e}")
            return pd.DataFrame()
    
    def upload_file(self, local_file, bucket_name, s3_key):
        """Upload file to S3"""
        try:
            self.s3.upload_file(local_file, bucket_name, s3_key)
            print(f"Uploaded {local_file} to s3://{bucket_name}/{s3_key}")
        except Exception as e:
            print(f"Error uploading file: {e}")
    
    def download_file(self, bucket_name, s3_key, local_file):
        """Download file from S3"""
        try:
            self.s3.download_file(bucket_name, s3_key, local_file)
            print(f"Downloaded s3://{bucket_name}/{s3_key} to {local_file}")
        except Exception as e:
            print(f"Error downloading file: {e}")
    
    def process_csv_from_s3(self, bucket_name, csv_key):
        """Process CSV file directly from S3"""
        try:
            obj = self.s3.get_object(Bucket=bucket_name, Key=csv_key)
            df = pd.read_csv(obj['Body'])
            
            print(f"CSV Analysis for s3://{bucket_name}/{csv_key}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print("\nFirst 5 rows:")
            print(df.head())
            
            return df
            
        except Exception as e:
            print(f"Error processing CSV: {e}")
            return None

# Usage example
if __name__ == "__main__":
    s3_proc = S3Processor()
    
    # List all buckets
    buckets_df = s3_proc.list_buckets()
    print("S3 Buckets:")
    print(buckets_df)
    
    # Analyze first bucket (if exists)
    if not buckets_df.empty:
        bucket_name = buckets_df.iloc[0]['Name']
        contents_df = s3_proc.analyze_bucket_contents(bucket_name)
        print(f"\nContents of {bucket_name}:")
        print(contents_df.head())
```

### **Step 4: CloudWatch Log Analyzer**
Create `cloudwatch_analyzer.py`:

```python
import boto3
import pandas as pd
from datetime import datetime, timedelta
import json
import re

class CloudWatchAnalyzer:
    def __init__(self, region='us-east-1'):
        self.logs = boto3.client('logs', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.region = region
    
    def list_log_groups(self):
        """List all CloudWatch log groups"""
        response = self.logs.describe_log_groups()
        log_groups = []
        
        for group in response['logGroups']:
            log_groups.append({
                'LogGroupName': group['logGroupName'],
                'CreationTime': datetime.fromtimestamp(group['creationTime']/1000),
                'StoredBytes': group.get('storedBytes', 0),
                'RetentionInDays': group.get('retentionInDays', 'Never Expire')
            })
        
        return pd.DataFrame(log_groups)
    
    def analyze_log_events(self, log_group_name, hours=24, filter_pattern=''):
        """Analyze log events from a log group"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            kwargs = {
                'logGroupName': log_group_name,
                'startTime': int(start_time.timestamp() * 1000),
                'endTime': int(end_time.timestamp() * 1000)
            }
            
            if filter_pattern:
                kwargs['filterPattern'] = filter_pattern
            
            response = self.logs.filter_log_events(**kwargs)
            
            events = []
            for event in response['events']:
                events.append({
                    'Timestamp': datetime.fromtimestamp(event['timestamp']/1000),
                    'Message': event['message'],
                    'LogStream': event['logStreamName']
                })
            
            df = pd.DataFrame(events)
            
            if not df.empty:
                print(f"Log Group: {log_group_name}")
                print(f"Total Events: {len(events)}")
                print(f"Time Range: {start_time} to {end_time}")
                
                # Analyze error patterns
                if not df.empty:
                    error_events = df[df['Message'].str.contains('ERROR|Error|error', na=False)]
                    print(f"Error Events: {len(error_events)}")
            
            return df
            
        except Exception as e:
            print(f"Error analyzing logs: {e}")
            return pd.DataFrame()
    
    def get_metric_statistics(self, namespace, metric_name, dimensions, hours=24):
        """Get CloudWatch metric statistics"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            response = self.cloudwatch.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric_name,
                Dimensions=dimensions,
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Average', 'Maximum', 'Minimum']
            )
            
            df = pd.DataFrame(response['Datapoints'])
            df = df.sort_values('Timestamp')
            
            return df
            
        except Exception as e:
            print(f"Error getting metrics: {e}")
            return pd.DataFrame()
    
    def create_custom_metric(self, namespace, metric_name, value, unit='Count'):
        """Create custom CloudWatch metric"""
        try:
            self.cloudwatch.put_metric_data(
                Namespace=namespace,
                MetricData=[
                    {
                        'MetricName': metric_name,
                        'Value': value,
                        'Unit': unit,
                        'Timestamp': datetime.utcnow()
                    }
                ]
            )
            print(f"Created metric: {namespace}/{metric_name} = {value}")
        except Exception as e:
            print(f"Error creating metric: {e}")

# Usage example
if __name__ == "__main__":
    cw_analyzer = CloudWatchAnalyzer()
    
    # List log groups
    log_groups_df = cw_analyzer.list_log_groups()
    print("CloudWatch Log Groups:")
    print(log_groups_df)
    
    # Analyze first log group (if exists)
    if not log_groups_df.empty:
        log_group = log_groups_df.iloc[0]['LogGroupName']
        events_df = cw_analyzer.analyze_log_events(log_group, hours=1)
        print(f"\nRecent events from {log_group}:")
        print(events_df.head())
```

### **Step 5: Main Automation Script**
Create `aws_toolkit.py`:

```python
#!/usr/bin/env python3
"""
AWS DevOps Automation Toolkit
Main script to orchestrate AWS infrastructure management
"""

import argparse
import sys
from ec2_manager import EC2Manager
from s3_processor import S3Processor
from cloudwatch_analyzer import CloudWatchAnalyzer

def main():
    parser = argparse.ArgumentParser(description='AWS DevOps Automation Toolkit')
    parser.add_argument('--service', choices=['ec2', 's3', 'cloudwatch'], 
                       required=True, help='AWS service to manage')
    parser.add_argument('--action', required=True, 
                       help='Action to perform')
    parser.add_argument('--region', default='us-east-1', 
                       help='AWS region')
    
    args = parser.parse_args()
    
    try:
        if args.service == 'ec2':
            ec2_mgr = EC2Manager(args.region)
            
            if args.action == 'list':
                df = ec2_mgr.list_instances()
                print(df)
            elif args.action == 'metrics':
                df = ec2_mgr.list_instances()
                if not df.empty:
                    instance_id = df.iloc[0]['InstanceId']
                    metrics = ec2_mgr.get_instance_metrics(instance_id)
                    print(metrics)
        
        elif args.service == 's3':
            s3_proc = S3Processor(args.region)
            
            if args.action == 'list':
                df = s3_proc.list_buckets()
                print(df)
            elif args.action == 'analyze':
                buckets_df = s3_proc.list_buckets()
                if not buckets_df.empty:
                    bucket_name = buckets_df.iloc[0]['Name']
                    contents = s3_proc.analyze_bucket_contents(bucket_name)
                    print(contents)
        
        elif args.service == 'cloudwatch':
            cw_analyzer = CloudWatchAnalyzer(args.region)
            
            if args.action == 'list':
                df = cw_analyzer.list_log_groups()
                print(df)
            elif args.action == 'analyze':
                log_groups = cw_analyzer.list_log_groups()
                if not log_groups.empty:
                    log_group = log_groups.iloc[0]['LogGroupName']
                    events = cw_analyzer.analyze_log_events(log_group)
                    print(events.head())
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## 🧪 **Testing & Validation**

### **Test Script**
Create `test_toolkit.py`:

```python
import unittest
from ec2_manager import EC2Manager
from s3_processor import S3Processor
from cloudwatch_analyzer import CloudWatchAnalyzer

class TestAWSToolkit(unittest.TestCase):
    
    def setUp(self):
        self.ec2_mgr = EC2Manager()
        self.s3_proc = S3Processor()
        self.cw_analyzer = CloudWatchAnalyzer()
    
    def test_ec2_list_instances(self):
        """Test EC2 instance listing"""
        df = self.ec2_mgr.list_instances()
        self.assertIsNotNone(df)
        print(f"Found {len(df)} EC2 instances")
    
    def test_s3_list_buckets(self):
        """Test S3 bucket listing"""
        df = self.s3_proc.list_buckets()
        self.assertIsNotNone(df)
        print(f"Found {len(df)} S3 buckets")
    
    def test_cloudwatch_log_groups(self):
        """Test CloudWatch log groups"""
        df = self.cw_analyzer.list_log_groups()
        self.assertIsNotNone(df)
        print(f"Found {len(df)} log groups")

if __name__ == '__main__':
    unittest.main()
```

---

## 🚀 **Usage Examples**

```bash
# List EC2 instances
python aws_toolkit.py --service ec2 --action list

# Analyze S3 buckets
python aws_toolkit.py --service s3 --action analyze

# Check CloudWatch logs
python aws_toolkit.py --service cloudwatch --action analyze

# Run tests
python test_toolkit.py
```

---

## 📊 **Expected Outputs**

### **EC2 Instances:**
```
  InstanceId InstanceType      State            LaunchTime    PublicIP
0  i-1234567      t2.micro    running  2024-01-15 10:30:00  1.2.3.4
1  i-7890123      t3.small    stopped  2024-01-14 15:45:00      N/A
```

### **S3 Analysis:**
```
Bucket: my-data-bucket
Total Objects: 150
Total Size: 2048.5 MB
Average Size: 13.66 MB
```

### **CloudWatch Metrics:**
```
            Timestamp  Average  Maximum
0 2024-01-15 10:00:00     45.2     78.5
1 2024-01-15 11:00:00     52.1     85.3
```

---

## 🧹 **Cleanup**

```bash
# Deactivate virtual environment
deactivate

# Remove temporary files
rm -rf __pycache__/
rm -f *.pyc
```

---

## 🎯 **Next Steps**
- Extend scripts with more AWS services
- Add error handling and logging
- Create configuration files
- Build web dashboard interface
- Integrate with CI/CD pipelines

**Completion Time:** 4-6 hours
**Difficulty:** Beginner to Intermediate