# Lab 2.1: Production SageMaker Pipeline
*Build end-to-end ML pipeline with SageMaker*

## 🎯 **Objectives**
- Create automated ML training pipeline
- Deploy model with CI/CD integration
- Set up monitoring and alerting

---

## 📋 **Prerequisites**
```bash
pip install sagemaker boto3 pandas scikit-learn
aws configure
```

---

## 🛠️ **Implementation**

### **Step 1: Data Preparation**
Create `data_prep.py`:

```python
import pandas as pd
import boto3
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

def prepare_data():
    # Load sample dataset
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['target'] = boston.target
    
    # Split data
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Upload to S3
    s3 = boto3.client('s3')
    bucket = 'your-ml-bucket'
    
    train.to_csv('/tmp/train.csv', index=False)
    test.to_csv('/tmp/test.csv', index=False)
    
    s3.upload_file('/tmp/train.csv', bucket, 'data/train.csv')
    s3.upload_file('/tmp/test.csv', bucket, 'data/test.csv')
    
    return f's3://{bucket}/data/'

if __name__ == "__main__":
    data_path = prepare_data()
    print(f"Data uploaded to: {data_path}")
```

### **Step 2: Training Script**
Create `train.py`:

```python
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

def train_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    args = parser.parse_args()
    
    # Load data
    train_df = pd.read_csv(f"{args.train}/train.csv")
    test_df = pd.read_csv(f"{args.test}/test.csv")
    
    # Prepare features
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Test MSE: {mse}")
    
    # Save model
    joblib.dump(model, f"{args.model_dir}/model.joblib")

if __name__ == "__main__":
    train_model()
```

### **Step 3: SageMaker Pipeline**
Create `sagemaker_pipeline.py`:

```python
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.model import Model
from sagemaker.pipeline import Pipeline
from sagemaker.pipeline.steps import TrainingStep, CreateModelStep
from sagemaker.pipeline.parameters import ParameterString

class MLPipeline:
    def __init__(self, role, bucket):
        self.session = sagemaker.Session()
        self.role = role
        self.bucket = bucket
        self.region = self.session.boto_region_name
    
    def create_pipeline(self):
        # Parameters
        input_data = ParameterString(
            name="InputData",
            default_value=f"s3://{self.bucket}/data"
        )
        
        # Training step
        sklearn_estimator = SKLearn(
            entry_point="train.py",
            framework_version="0.23-1",
            instance_type="ml.m5.large",
            role=self.role,
            sagemaker_session=self.session
        )
        
        training_step = TrainingStep(
            name="TrainModel",
            estimator=sklearn_estimator,
            inputs={
                "train": f"{input_data}/train.csv",
                "test": f"{input_data}/test.csv"
            }
        )
        
        # Model creation step
        model = Model(
            image_uri=sklearn_estimator.image_uri,
            model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            sagemaker_session=self.session,
            role=self.role
        )
        
        create_model_step = CreateModelStep(
            name="CreateModel",
            model=model
        )
        
        # Pipeline
        pipeline = Pipeline(
            name="MLOpsPipeline",
            parameters=[input_data],
            steps=[training_step, create_model_step],
            sagemaker_session=self.session
        )
        
        return pipeline
    
    def execute_pipeline(self):
        pipeline = self.create_pipeline()
        pipeline.upsert(role_arn=self.role)
        execution = pipeline.start()
        return execution

if __name__ == "__main__":
    role = "arn:aws:iam::123456789012:role/SageMakerRole"
    bucket = "your-ml-bucket"
    
    ml_pipeline = MLPipeline(role, bucket)
    execution = ml_pipeline.execute_pipeline()
    print(f"Pipeline execution started: {execution.arn}")
```

---

## 🚀 **Deployment**

### **CloudFormation Template**
Create `infrastructure.yaml`:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'SageMaker MLOps Infrastructure'

Parameters:
  BucketName:
    Type: String
    Default: mlops-pipeline-bucket

Resources:
  MLBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Ref BucketName
      VersioningConfiguration:
        Status: Enabled

  SageMakerRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: sagemaker.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
        - arn:aws:iam::aws:policy/AmazonS3FullAccess

Outputs:
  BucketName:
    Value: !Ref MLBucket
  SageMakerRoleArn:
    Value: !GetAtt SageMakerRole.Arn
```

### **Deployment Script**
Create `deploy.sh`:

```bash
#!/bin/bash

# Deploy infrastructure
aws cloudformation deploy \
  --template-file infrastructure.yaml \
  --stack-name mlops-pipeline \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides BucketName=your-ml-bucket

# Get outputs
BUCKET=$(aws cloudformation describe-stacks \
  --stack-name mlops-pipeline \
  --query 'Stacks[0].Outputs[?OutputKey==`BucketName`].OutputValue' \
  --output text)

ROLE=$(aws cloudformation describe-stacks \
  --stack-name mlops-pipeline \
  --query 'Stacks[0].Outputs[?OutputKey==`SageMakerRoleArn`].OutputValue' \
  --output text)

echo "Bucket: $BUCKET"
echo "Role: $ROLE"

# Prepare and upload data
python data_prep.py

# Execute pipeline
python sagemaker_pipeline.py
```

---

## 📊 **Monitoring Setup**

Create `monitoring.py`:

```python
import boto3
import json

def setup_monitoring():
    cloudwatch = boto3.client('cloudwatch')
    
    # Create custom metric for model performance
    cloudwatch.put_metric_alarm(
        AlarmName='ModelPerformanceAlarm',
        ComparisonOperator='GreaterThanThreshold',
        EvaluationPeriods=1,
        MetricName='ModelMSE',
        Namespace='MLOps/Model',
        Period=300,
        Statistic='Average',
        Threshold=10.0,
        ActionsEnabled=True,
        AlarmActions=[
            'arn:aws:sns:us-east-1:123456789012:model-alerts'
        ],
        AlarmDescription='Alert when model MSE exceeds threshold'
    )
    
    print("Monitoring setup complete")

if __name__ == "__main__":
    setup_monitoring()
```

---

## 🧪 **Testing**

Create `test_pipeline.py`:

```python
import unittest
import boto3
from sagemaker_pipeline import MLPipeline

class TestMLPipeline(unittest.TestCase):
    
    def setUp(self):
        self.role = "arn:aws:iam::123456789012:role/SageMakerRole"
        self.bucket = "your-ml-bucket"
        self.pipeline = MLPipeline(self.role, self.bucket)
    
    def test_pipeline_creation(self):
        pipeline = self.pipeline.create_pipeline()
        self.assertIsNotNone(pipeline)
        self.assertEqual(pipeline.name, "MLOpsPipeline")
    
    def test_data_exists(self):
        s3 = boto3.client('s3')
        try:
            s3.head_object(Bucket=self.bucket, Key='data/train.csv')
            s3.head_object(Bucket=self.bucket, Key='data/test.csv')
        except:
            self.fail("Training data not found in S3")

if __name__ == '__main__':
    unittest.main()
```

---

## 🎯 **Expected Results**

1. **Pipeline Execution**: Successful training and model creation
2. **Model Artifacts**: Stored in S3 with versioning
3. **Monitoring**: CloudWatch alarms for model performance
4. **Automation**: Repeatable pipeline execution

**Completion Time:** 6-8 hours
**Difficulty:** Intermediate