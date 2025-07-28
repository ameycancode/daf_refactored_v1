#!/usr/bin/env python3
"""
Energy Forecasting MLOps Codebase Generator (No IAM Role Management)
Creates the complete file and folder structure for the refactored implementation
Run this script in VS Code terminal from your desired project directory
"""

import os
import json
from datetime import datetime

def create_directory(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    print(f"📁 Created directory: {path}")

def create_file(filepath, content):
    """Create file with content"""
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
   
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"📄 Created file: {filepath}")

def main():
    """Generate complete codebase structure without IAM role management"""
   
    print("🚀 Energy Forecasting MLOps Codebase Generator (No IAM Role Management)")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Working Directory: {os.getcwd()}")
    print()
   
    # Root project structure
    project_structure = {
        "containers": {
            "preprocessing": {
                "src": {},
                "files": ["Dockerfile", "requirements.txt", "build_and_push.sh"]
            },
            "training": {
                "src": {},
                "files": ["Dockerfile", "requirements.txt", "build_and_push.sh"]
            },
            "prediction": {
                "src": {},
                "files": ["Dockerfile", "requirements.txt", "build_and_push.sh"]
            }
        },
        "lambda-functions": {
            "model-registry": {},
            "endpoint-management": {}
        },
        "infrastructure": {},
        "config": {},
        "deployment": {},
        "admin-requests": {},
        "scripts": {},
        "docs": {}
    }
   
    # Create directory structure
    print("Creating directory structure...")
    for root_dir, subdirs in project_structure.items():
        create_directory(root_dir)
        if isinstance(subdirs, dict):
            for subdir, subsubdirs in subdirs.items():
                subdir_path = os.path.join(root_dir, subdir)
                create_directory(subdir_path)
                if isinstance(subsubdirs, dict):
                    for subsubdir in subsubdirs:
                        create_directory(os.path.join(subdir_path, subsubdir))
   
    print("\n" + "=" * 70)
    print("Creating configuration files...")
   
    # 1. Root configuration files
    create_file("README.md", get_readme_content())
    create_file(".gitignore", get_gitignore_content())
    create_file("requirements.txt", get_root_requirements())
   
    # 2. Container files
    print("\nCreating container files...")
   
    # Preprocessing container
    create_file("containers/preprocessing/Dockerfile", get_preprocessing_dockerfile())
    create_file("containers/preprocessing/requirements.txt", get_preprocessing_requirements())
    create_file("containers/preprocessing/build_and_push.sh", get_build_script("energy-preprocessing"))
    create_file("containers/preprocessing/src/__init__.py", "")
    create_file("containers/preprocessing/src/main.py", "# Preprocessing main.py - Use from previous artifacts")
    create_file("containers/preprocessing/src/process_data.py", "# Process data module - Use from previous artifacts")
    create_file("containers/preprocessing/src/feature_engineering.py", "# Feature engineering - Use from previous artifacts")
    create_file("containers/preprocessing/src/data_split.py", "# Data split module - Use from previous artifacts")
    create_file("containers/preprocessing/src/utils.py", "# Utilities module - Use from previous artifacts")
   
    # Training container
    create_file("containers/training/Dockerfile", get_training_dockerfile())
    create_file("containers/training/requirements.txt", get_training_requirements())
    create_file("containers/training/build_and_push.sh", get_build_script("energy-training"))
    create_file("containers/training/src/__init__.py", "")
    create_file("containers/training/src/main.py", "# MODIFIED Training main.py - Use modified version from artifacts")
    create_file("containers/training/src/data_preparation.py", "# Data preparation - Use from previous artifacts")
    create_file("containers/training/src/train_model.py", "# Train model module - Use from previous artifacts")
    create_file("containers/training/src/model_evaluation.py", "# Model evaluation - Use from previous artifacts")
    create_file("containers/training/src/utils.py", "# Utilities module - Use from previous artifacts")
   
    # Prediction container (for future use)
    create_file("containers/prediction/Dockerfile", get_prediction_dockerfile())
    create_file("containers/prediction/requirements.txt", get_prediction_requirements())
    create_file("containers/prediction/build_and_push.sh", get_build_script("energy-prediction"))
    create_file("containers/prediction/src/__init__.py", "")
    create_file("containers/prediction/src/main.py", "# Prediction main.py - For future daily predictions")
    create_file("containers/prediction/src/predictions.py", "# Predictions module - For future use")
    create_file("containers/prediction/src/T_forecast.py", "# Temperature forecast - For future use")
    create_file("containers/prediction/src/R_forecast.py", "# Radiation forecast - For future use")
    create_file("containers/prediction/src/visualization.py", "# Visualization - For future use")
    create_file("containers/prediction/src/utils.py", "# Utilities module - For future use")
   
    # 3. Lambda functions
    print("\nCreating Lambda function files...")
   
    create_file("lambda-functions/model-registry/lambda_function.py", "# Model Registry Lambda - Use from artifacts")
    create_file("lambda-functions/model-registry/requirements.txt", get_lambda_model_registry_requirements())
   
    create_file("lambda-functions/endpoint-management/lambda_function.py", "# Endpoint Management Lambda - Use from artifacts")
    create_file("lambda-functions/endpoint-management/requirements.txt", get_lambda_endpoint_requirements())
   
    # 4. Infrastructure files (WITHOUT IAM role management)
    print("\nCreating infrastructure files (no IAM role management)...")
   
    create_file("infrastructure/__init__.py", "")
    create_file("infrastructure/setup_infrastructure.py", "# Infrastructure setup WITHOUT role creation - Use modified version")
    create_file("infrastructure/step_functions_definitions.py", "# Step Functions definitions - Use modified version from artifacts")
   
    # 5. Configuration files
    print("\nCreating configuration files...")
   
    create_file("config/mlops_config.json", "# MLOps configuration - Use from artifacts")
    create_file("config/model_config.json", get_model_config())
    create_file("config/data_paths.json", get_data_paths_config())
    create_file("config/endpoint_config_template.json", get_endpoint_config_template())
   
    # 6. Deployment files (WITHOUT IAM role creation)
    print("\nCreating deployment files...")
   
    create_file("deployment/__init__.py", "")
    create_file("deployment/deploy_system.py", "# System deployment script WITHOUT role creation")
    create_file("deployment/lambda_deployer.py", "# Lambda deployer WITHOUT role creation - Use modified version")
    create_file("deployment/test_pipeline.py", get_test_pipeline_script())
   
    # 7. Admin request files
    print("\nCreating admin request files...")
   
    create_file("admin-requests/required_permissions.md", get_required_permissions_request())
    create_file("admin-requests/required_iam_roles.md", get_iam_roles_specification())
    create_file("admin-requests/jira_ticket_template.md", get_jira_ticket_template())
   
    # 8. Scripts
    print("\nCreating utility scripts...")
   
    create_file("scripts/setup_aws_credentials.sh", get_aws_setup_script())
    create_file("scripts/deploy_all.sh", get_deploy_all_script())
    create_file("scripts/test_pipeline.sh", get_test_script())
    create_file("scripts/cleanup.sh", get_cleanup_script())
    create_file("scripts/verify_roles.py", get_role_verification_script())
   
    # 9. Documentation
    print("\nCreating documentation files...")
   
    create_file("docs/deployment_guide.md", get_deployment_guide())
    create_file("docs/mlops_architecture.md", get_architecture_doc())
    create_file("docs/admin_setup_required.md", get_admin_setup_doc())
   
    # 10. Create file mapping guide
    create_file("FILE_MAPPING_GUIDE.md", get_file_mapping_guide())
   
    print("\n" + "=" * 70)
    print("✅ CODEBASE GENERATION COMPLETE!")
    print("=" * 70)
    print()
    print("📋 NEXT STEPS:")
    print("1. ⚠️  FIRST: Create JIRA ticket using admin-requests/jira_ticket_template.md")
    print("2. ⏳ WAIT: For admin team to create IAM roles and grant permissions")
    print("3. ✅ THEN: Review FILE_MAPPING_GUIDE.md for artifact mappings")
    print("4. ✅ Copy specific code from artifacts to marked files")
    print("5. ✅ Run: python scripts/verify_roles.py (to check admin setup)")
    print("6. ✅ Run: python infrastructure/setup_infrastructure.py")
    print("7. ✅ Run: python deployment/lambda_deployer.py")
    print("8. ✅ Run: bash scripts/deploy_all.sh")
    print()
    print("📂 Generated Structure:")
    print_directory_tree(".", 0, 3)

def print_directory_tree(path, level, max_level):
    """Print directory tree structure"""
    if level > max_level:
        return
   
    items = []
    if os.path.exists(path):
        items = sorted(os.listdir(path))
   
    for item in items:
        if item.startswith('.'):
            continue
           
        item_path = os.path.join(path, item)
        indent = "  " * level
       
        if os.path.isdir(item_path):
            print(f"{indent}📁 {item}/")
            print_directory_tree(item_path, level + 1, max_level)
        else:
            print(f"{indent}📄 {item}")

# Content generators (keeping existing ones and adding new admin-related content)

def get_required_permissions_request():
    return '''# Required AWS Permissions for Energy Forecasting MLOps

## Current Access
- ✅ Lambda function creation/management
- ✅ EventBridge rule creation
- ✅ SageMaker Processing Jobs
- ✅ SageMaker Model Registry
- ✅ S3 bucket access

## Required Additional Permissions

### 1. ECR (Elastic Container Registry) Permissions
```
ecr:GetAuthorizationToken
ecr:BatchCheckLayerAvailability
ecr:GetDownloadUrlForLayer
ecr:BatchGetImage
ecr:CreateRepository
ecr:PutImage
ecr:InitiateLayerUpload
ecr:UploadLayerPart
ecr:CompleteLayerUpload
ecr:DescribeRepositories
ecr:ListImages
ecr:DescribeImages
```

### 2. Step Functions Permissions
```
states:CreateStateMachine
states:UpdateStateMachine
states:DeleteStateMachine
states:DescribeStateMachine
states:ListStateMachines
states:StartExecution
states:DescribeExecution
states:ListExecutions
states:StopExecution
states:GetExecutionHistory
```

### 3. IAM Role Passing (for existing roles)
```
iam:PassRole (for pre-created roles only)
iam:GetRole
iam:ListRoles
```

## Required IAM Roles (to be created by admin team)
- EnergyForecastingSageMakerRole
- EnergyForecastingLambdaExecutionRole
- EnergyForecastingStepFunctionsRole
- EnergyForecastingEventBridgeRole

See required_iam_roles.md for detailed specifications.
'''

def get_iam_roles_specification():
    return '''# IAM Roles Specification for Admin Team

## Roles Required

### 1. EnergyForecastingSageMakerRole
**Purpose**: Execute SageMaker Processing Jobs
**Trusted Service**: sagemaker.amazonaws.com
**Policies**:
- AmazonSageMakerFullAccess (managed)
- Custom S3 policy for data/model buckets
- Custom Lambda invoke policy

### 2. EnergyForecastingLambdaExecutionRole
**Purpose**: Execute Lambda functions for model registry and endpoint management
**Trusted Service**: lambda.amazonaws.com
**Policies**:
- AWSLambdaBasicExecutionRole (managed)
- AmazonSageMakerFullAccess (managed)
- Custom S3 policy for data/model buckets

### 3. EnergyForecastingStepFunctionsRole
**Purpose**: Execute Step Functions workflows
**Trusted Service**: states.amazonaws.com
**Policies**:
- Custom SageMaker processing policy
- Custom Lambda invoke policy
- Custom IAM pass role policy

### 4. EnergyForecastingEventBridgeRole
**Purpose**: Trigger Step Functions from EventBridge
**Trusted Service**: events.amazonaws.com
**Policies**:
- Custom Step Functions start execution policy

## S3 Buckets Referenced
- sdcp-dev-sagemaker-energy-forecasting-data
- sdcp-dev-sagemaker-energy-forecasting-models

## ECR Repositories Required
- energy-preprocessing
- energy-training
- energy-prediction
'''

def get_jira_ticket_template():
    return '''# JIRA Ticket Template for Energy Forecasting MLOps Access Request

Copy the content below for your JIRA ticket:

---

## Title
AWS Permissions Request: Energy Forecasting MLOps Pipeline Implementation

## Type
Access Request

## Priority
Medium

## Description

### Project Overview
Implementing an automated MLOps pipeline for energy load forecasting using AWS SageMaker, Lambda, Step Functions, and ECR containers.

### Current Access
- ✅ Lambda function creation/management
- ✅ EventBridge rule creation  
- ✅ SageMaker Processing Jobs
- ✅ SageMaker Model Registry
- ✅ S3 bucket access

### Required Additional Permissions

#### 1. ECR (Container Registry) Access
**Purpose**: Build and deploy containerized ML training/preprocessing workflows
**Permissions Needed**:
- ecr:GetAuthorizationToken
- ecr:BatchCheckLayerAvailability
- ecr:GetDownloadUrlForLayer
- ecr:BatchGetImage
- ecr:CreateRepository
- ecr:PutImage
- ecr:InitiateLayerUpload
- ecr:UploadLayerPart
- ecr:CompleteLayerUpload
- ecr:DescribeRepositories

#### 2. Step Functions Access
**Purpose**: Orchestrate automated ML pipeline workflows
**Permissions Needed**:
- states:CreateStateMachine
- states:UpdateStateMachine
- states:DescribeStateMachine
- states:ListStateMachines
- states:StartExecution
- states:DescribeExecution
- states:ListExecutions
- states:StopExecution

#### 3. IAM Role Passing (Limited)
**Purpose**: Pass pre-created roles to AWS services
**Permissions Needed**:
- iam:PassRole (for specific EnergyForecasting* roles only)
- iam:GetRole
- iam:ListRoles

### Required IAM Roles Creation (Admin Team)

#### IAM Roles to Create:
1. **EnergyForecastingSageMakerRole**
   - Trusted Service: sagemaker.amazonaws.com
   - Policies: AmazonSageMakerFullAccess + Custom S3 access

2. **EnergyForecastingLambdaExecutionRole**
   - Trusted Service: lambda.amazonaws.com  
   - Policies: AWSLambdaBasicExecutionRole + AmazonSageMakerFullAccess

3. **EnergyForecastingStepFunctionsRole**
   - Trusted Service: states.amazonaws.com
   - Policies: Custom SageMaker + Lambda invoke permissions

4. **EnergyForecastingEventBridgeRole**
   - Trusted Service: events.amazonaws.com
   - Policies: Step Functions start execution

#### ECR Repositories to Create:
- energy-preprocessing
- energy-training  
- energy-prediction

### Business Justification
- **Automation**: Eliminates manual model training/deployment processes
- **Cost Optimization**: 98% reduction in ML endpoint costs through automated lifecycle management  
- **Scalability**: Supports multiple customer profiles with automated retraining
- **Governance**: Implements model versioning and approval workflows

### Timeline
- **Needed By**: [Your target date]
- **Dependencies**: MLOps pipeline implementation blocked without these permissions

### Additional Context
This implements a sophisticated MLOps pipeline with:
- Weekly automated model retraining
- Model registry with performance validation
- Cost-optimized endpoint management
- Automated daily predictions

### Attachments
- Detailed IAM role specifications (if needed)
- Architecture diagram (if available)

---

## Acceptance Criteria
- [ ] ECR permissions granted for container management
- [ ] Step Functions permissions granted for workflow orchestration  
- [ ] IAM roles created with specified policies
- [ ] ECR repositories created
- [ ] Permissions verified through test deployment

## Contact
[Your contact information for questions]
'''

def get_role_verification_script():
    return '''#!/usr/bin/env python3
"""
Verify that required IAM roles exist and are accessible
Run this after admin team completes setup
"""

import boto3
import json
from datetime import datetime

def verify_roles():
    """Verify all required IAM roles exist"""
   
    iam_client = boto3.client('iam')
    account_id = boto3.client('sts').get_caller_identity()['Account']
   
    required_roles = [
        'EnergyForecastingSageMakerRole',
        'EnergyForecastingLambdaExecutionRole',
        'EnergyForecastingStepFunctionsRole',
        'EnergyForecastingEventBridgeRole'
    ]
   
    print("🔍 Verifying IAM Roles...")
   
    verification_results = {}
   
    for role_name in required_roles:
        try:
            response = iam_client.get_role(RoleName=role_name)
            verification_results[role_name] = {
                'exists': True,
                'arn': response['Role']['Arn'],
                'created': response['Role']['CreateDate'].isoformat()
            }
            print(f"✅ {role_name}: EXISTS")
           
        except iam_client.exceptions.NoSuchEntityException:
            verification_results[role_name] = {
                'exists': False,
                'error': 'Role does not exist'
            }
            print(f"❌ {role_name}: NOT FOUND")
        except Exception as e:
            verification_results[role_name] = {
                'exists': False,
                'error': str(e)
            }
            print(f"❌ {role_name}: ERROR - {str(e)}")
   
    # Check ECR repositories
    print("\\n🔍 Verifying ECR Repositories...")
   
    ecr_client = boto3.client('ecr')
    required_repos = ['energy-preprocessing', 'energy-training', 'energy-prediction']
   
    for repo_name in required_repos:
        try:
            response = ecr_client.describe_repositories(repositoryNames=[repo_name])
            print(f"✅ {repo_name}: EXISTS")
            verification_results[f"ecr_{repo_name}"] = {
                'exists': True,
                'uri': response['repositories'][0]['repositoryUri']
            }
        except ecr_client.exceptions.RepositoryNotFoundException:
            print(f"❌ {repo_name}: NOT FOUND")
            verification_results[f"ecr_{repo_name}"] = {
                'exists': False,
                'error': 'Repository does not exist'
            }
        except Exception as e:
            print(f"❌ {repo_name}: ERROR - {str(e)}")
            verification_results[f"ecr_{repo_name}"] = {
                'exists': False,
                'error': str(e)
            }
   
    # Save verification results
    with open('verification_results.json', 'w') as f:
        json.dump(verification_results, f, indent=2, default=str)
   
    # Summary
    role_count = sum(1 for role in required_roles if verification_results[role]['exists'])
    repo_count = sum(1 for repo in required_repos if verification_results[f"ecr_{repo}"]["exists"])
   
    print(f"\\n📊 VERIFICATION SUMMARY:")
    print(f"IAM Roles: {role_count}/{len(required_roles)} ✅")
    print(f"ECR Repos: {repo_count}/{len(required_repos)} ✅")
   
    if role_count == len(required_roles) and repo_count == len(required_repos):
        print("\\n🎉 ALL REQUIREMENTS MET! Ready to proceed with deployment.")
        return True
    else:
        print("\\n⚠️  MISSING REQUIREMENTS! Contact admin team to complete setup.")
        return False

if __name__ == "__main__":
    success = verify_roles()
    exit(0 if success else 1)
'''

def get_admin_setup_doc():
    return '''# Admin Setup Required

## Overview
This document outlines what needs to be set up by the AWS admin team before the MLOps pipeline can be deployed.

## Prerequisites  
The admin team must create IAM roles and ECR repositories before the development team can proceed.

## Required Setup

### 1. IAM Roles
Four IAM roles must be created with specific trusted services and policies.

### 2. ECR Repositories  
Three ECR repositories must be created for container images.

### 3. Permissions
Data scientist role must be granted additional ECR and Step Functions permissions.

## Verification
After admin setup, run: `python scripts/verify_roles.py`

## Next Steps
Once verification passes:
1. Deploy infrastructure: `python infrastructure/setup_infrastructure.py`
2. Deploy Lambda functions: `python deployment/lambda_deployer.py`  
3. Build containers: `bash scripts/deploy_all.sh`
4. Test pipeline: `python deployment/test_pipeline.py`

## Support
Contact [admin team] for IAM role and repository creation.
Contact [your team] for MLOps pipeline questions.
'''

# Keep all the existing content generators from the original script
def get_readme_content():
    return '''# Energy Forecasting MLOps Pipeline

Complete MLOps pipeline for energy load forecasting with model registry and cost optimization.

## Architecture

- **Step 1**: Containerized Training (Preprocessing + Model Training)
- **Step 2**: Model Registry (Lambda function for model validation and registration)
- **Step 3**: Endpoint Management (Lambda function for cost-optimized endpoint lifecycle)

## Prerequisites

⚠️ **IMPORTANT**: Admin team must create IAM roles first!

1. Create JIRA ticket using `admin-requests/jira_ticket_template.md`
2. Wait for admin team to create IAM roles and ECR repositories
3. Verify setup: `python scripts/verify_roles.py`

## Quick Start

1. **Verify Admin Setup**:
   ```bash
   python scripts/verify_roles.py
   ```

2. **Setup Infrastructure**:
   ```bash
   python infrastructure/setup_infrastructure.py
   ```

3. **Deploy Lambda Functions**:
   ```bash
   python deployment/lambda_deployer.py
   ```

4. **Build and Deploy Containers**:
   ```bash
   bash scripts/deploy_all.sh
   ```

5. **Test Pipeline**:
   ```bash
   bash scripts/test_pipeline.sh
   ```

## Cost Optimization

- Endpoints are created, tested, and deleted automatically
- 98% cost reduction compared to always-on endpoints
- Configurations saved to S3 for recreation during predictions

## File Mapping

See [FILE_MAPPING_GUIDE.md](FILE_MAPPING_GUIDE.md) for which files to copy from artifacts.
'''

def get_file_mapping_guide():
    return '''# File Mapping Guide

This guide shows which files to copy from the artifacts in our conversation.

## ⚠️ PREREQUISITES
**FIRST**: Create JIRA ticket using `admin-requests/jira_ticket_template.md`
**WAIT**: For admin team to create IAM roles  
**VERIFY**: Run `python scripts/verify_roles.py`

## ✅ Files to Copy Directly (No Changes)

### Preprocessing Container
- `containers/preprocessing/src/main.py` ← **Preprocessing Container (main.py)** artifact
- `containers/preprocessing/src/process_data.py` ← **Your original process_data.py**
- `containers/preprocessing/src/feature_engineering.py` ← **Your original feature_engineering.py**
- `containers/preprocessing/src/data_split.py` ← **Your original data_split.py**
- `containers/preprocessing/src/utils.py` ← **Common Utilities Module (utils.py)** artifact

### Training Container Core Logic
- `containers/training/src/data_preparation.py` ← **Your original data_preparation.py**
- `containers/training/src/train_model.py` ← **Your original train_model.py**
- `containers/training/src/model_evaluation.py` ← **Your original model_evaluation.py**
- `containers/training/src/utils.py` ← **Common Utilities Module (utils.py)** artifact

## 🔧 Files to Copy with Modifications

### Modified Training Container
- `containers/training/src/main.py` ← **Modified Training Container Main (main.py)** artifact
  - **IMPORTANT**: This is the MODIFIED version that includes model metadata generation and Lambda triggers

### Lambda Functions
- `lambda-functions/model-registry/lambda_function.py` ← **Model Registry Lambda Function** artifact
- `lambda-functions/endpoint-management/lambda_function.py` ← **Endpoint Management Lambda Function** artifact

### Infrastructure (NO IAM ROLE CREATION)
- `infrastructure/setup_infrastructure.py` ← **Modified Infrastructure Setup (No Role Creation)** artifact
- `infrastructure/step_functions_definitions.py` ← **Modified Step Functions Definition** artifact

### Configuration
- `config/mlops_config.json` ← **MLOps Configuration File** artifact

### Deployment (NO IAM ROLE CREATION)
- `deployment/lambda_deployer.py` ← **Modified Lambda Deployer (No Role Creation)** artifact

## 📋 Action Items

1. **Create JIRA ticket** using `admin-requests/jira_ticket_template.md`
2. **Wait for admin setup** (IAM roles + ECR repositories)
3. **Verify setup**: `python scripts/verify_roles.py`
4. **Copy files marked above** from the respective artifacts
5. **Replace placeholder comments** in generated files with actual code
6. **Update account IDs and regions** in configuration files
7. **Test each component** after copying

## 🚀 Deployment Order

1. Verify admin setup completed
2. Copy and update files  
3. Run: `python infrastructure/setup_infrastructure.py`
4. Run: `python deployment/lambda_deployer.py`
5. Run: `bash scripts/deploy_all.sh`
6. Test: `python deployment/test_pipeline.py`
'''

# Include all other existing content generators (get_preprocessing_dockerfile, etc.)
# ... (keeping all the existing functions from the original script)

def get_preprocessing_dockerfile():
    return '''FROM python:3.9-slim

WORKDIR /opt/ml/processing

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt code/
RUN pip install --no-cache-dir -r code/requirements.txt

# Copy source code
COPY src/ code/src/

# Set Python path
ENV PYTHONPATH="/opt/ml/processing/code:${PYTHONPATH}"

# Set entrypoint
ENTRYPOINT ["python", "/opt/ml/processing/code/src/main.py"]
'''

def get_training_dockerfile():
    return '''FROM python:3.9-slim

WORKDIR /opt/ml/processing

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt code/
RUN pip install --no-cache-dir -r code/requirements.txt

# Copy source code
COPY src/ code/src/

# Set Python path
ENV PYTHONPATH="/opt/ml/processing/code:${PYTHONPATH}"

# Set entrypoint
ENTRYPOINT ["python", "/opt/ml/processing/code/src/main.py"]
'''

def get_prediction_dockerfile():
    return '''FROM python:3.9-slim

WORKDIR /opt/ml/processing

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt code/
RUN pip install --no-cache-dir -r code/requirements.txt

# Copy source code
COPY src/ code/src/

# Set Python path
ENV PYTHONPATH="/opt/ml/processing/code:${PYTHONPATH}"

# Set entrypoint
ENTRYPOINT ["python", "/opt/ml/processing/code/src/main.py"]
'''

# Include all other existing functions...
# (All the get_* functions from the original script should be included here)

def get_gitignore_content():
    return '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# AWS
.aws/
*.pem

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
logs/

# Deployment packages
*.zip
deployment_package/

# Temporary files
tmp/
temp/
.cache/

# Model artifacts
models/
*.pkl
*.joblib

# Data files
data/
*.csv
*.json

# Infrastructure state
terraform.tfstate*
.terraform/

# Verification results
verification_results.json

# OS
.DS_Store
Thumbs.db
'''

def get_root_requirements():
    return '''boto3>=1.26.137
sagemaker>=2.156.0
pandas>=1.5.3
numpy>=1.24.3
'''

def get_preprocessing_requirements():
    return '''boto3>=1.26.0
pandas>=1.5.0
numpy>=1.21.0
pytz>=2022.1
'''

def get_training_requirements():
    return '''boto3>=1.26.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
xgboost>=1.6.0
matplotlib>=3.5.0
plotly>=5.10.0
joblib>=1.1.0
pytz>=2022.1
'''

def get_prediction_requirements():
    return '''boto3>=1.26.0
pandas>=1.5.0
numpy>=1.21.0
joblib>=1.1.0
requests>=2.28.0
plotly>=5.10.0
openmeteo-requests>=1.0.0
requests-cache>=1.0.0
retry-requests>=1.0.0
pytz>=2022.1
'''

def get_lambda_model_registry_requirements():
    return '''boto3>=1.26.137
sagemaker>=2.156.0
pandas>=1.5.3
numpy>=1.24.3
'''

def get_lambda_endpoint_requirements():
    return '''boto3>=1.26.137
sagemaker>=2.156.0
'''

def get_build_script(repo_name):
    return f'''#!/bin/bash
# Build and push script for {repo_name}

# Set variables
REGION=us-west-2
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPOSITORY_NAME={repo_name}
IMAGE_TAG=latest

echo "Building and pushing $REPOSITORY_NAME..."

# Get ECR login
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Build image
echo "Building Docker image..."
docker build -t $REPOSITORY_NAME .

# Tag image
docker tag $REPOSITORY_NAME:$IMAGE_TAG $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:$IMAGE_TAG

# Push image
echo "Pushing image to ECR..."
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:$IMAGE_TAG

echo "Successfully pushed $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:$IMAGE_TAG"
'''

def get_model_config():
    return '''{
  "profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"],
  "model_type": "XGBoost",
  "version": "1.0.0"
}'''

def get_data_paths_config():
    return '''{
  "data_bucket": "sdcp-dev-sagemaker-energy-forecasting-data",
  "model_bucket": "sdcp-dev-sagemaker-energy-forecasting-models",
  "raw_data_path": "archived_folders/forecasting/data/raw/",
  "processed_data_path": "archived_folders/forecasting/data/xgboost/processed/",
  "model_input_path": "archived_folders/forecasting/data/xgboost/input/",
  "predictions_output_path": "archived_folders/forecasting/data/xgboost/output/"
}'''

def get_endpoint_config_template():
    return '''{
  "instance_types": {
    "RNN": "ml.t2.medium",
    "RN": "ml.t2.medium",
    "M": "ml.t2.small",
    "S": "ml.t2.small",
    "AGR": "ml.t2.small",
    "L": "ml.t2.micro",
    "A6": "ml.t2.small"
  },
  "auto_scaling": {
    "enabled": false,
    "min_capacity": 1,
    "max_capacity": 2
  }
}'''

def get_test_pipeline_script():
    return '''#!/usr/bin/env python3
"""
Test script for the complete MLOps pipeline
"""

import boto3
import json
from datetime import datetime

def test_training_pipeline():
    """Test the training pipeline"""
   
    stepfunctions = boto3.client('stepfunctions')
   
    # Get account ID
    account_id = boto3.client('sts').get_caller_identity()['Account']
    region = 'us-west-2'
   
    # Start training pipeline execution
    response = stepfunctions.start_execution(
        stateMachineArn=f'arn:aws:states:{region}:{account_id}:stateMachine:energy-forecasting-training-pipeline',
        name=f'test-execution-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        input=json.dumps({
            "PreprocessingJobName": f"test-preprocessing-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "TrainingJobName": f"test-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "PreprocessingImageUri": f"{account_id}.dkr.ecr.{region}.amazonaws.com/energy-preprocessing:latest",
            "TrainingImageUri": f"{account_id}.dkr.ecr.{region}.amazonaws.com/energy-training:latest"
        })
    )
   
    print(f"Started test execution: {response['executionArn']}")
    return response['executionArn']

if __name__ == "__main__":
    execution_arn = test_training_pipeline()
    print("Test pipeline started successfully!")
'''

def get_deployment_guide():
    return '''# Deployment Guide

## Prerequisites

1. AWS CLI configured with appropriate permissions
2. Docker installed and running
3. Python 3.9+ installed
4. ⚠️ **ADMIN TEAM MUST COMPLETE IAM ROLE SETUP FIRST**

## Step-by-Step Deployment

### 0. Admin Prerequisites (REQUIRED FIRST)

**Create JIRA ticket** using `admin-requests/jira_ticket_template.md`

Admin team must create:
- IAM roles (4 roles with specific policies)
- ECR repositories (3 repositories)
- Grant additional permissions to data scientist role

### 1. Verify Admin Setup

```bash
python scripts/verify_roles.py
```

This verifies:
- All required IAM roles exist
- ECR repositories are created
- Permissions are properly configured

### 2. Setup Infrastructure (No Role Creation)

```bash
python infrastructure/setup_infrastructure.py
```

This creates:
- ECR repositories (if not already created)
- Step Functions state machines
- EventBridge schedules

### 3. Deploy Lambda Functions

```bash
python deployment/lambda_deployer.py
```

This deploys:
- Model Registry Lambda (Step 2)
- Endpoint Management Lambda (Step 3)

### 4. Build and Push Containers

```bash
# All containers
bash scripts/deploy_all.sh

# Individual containers
cd containers/preprocessing && bash build_and_push.sh
cd containers/training && bash build_and_push.sh
```

### 5. Test the Pipeline

```bash
python deployment/test_pipeline.py
```

## Verification

1. Check AWS Console:
   - ECR repositories have images
   - Lambda functions are deployed
   - Step Functions exist
   - EventBridge rules are enabled

2. Monitor first execution:
   - CloudWatch logs for detailed output
   - S3 buckets for artifacts
   - Model Registry for registered models

## Troubleshooting

If deployment fails:
1. Run `python scripts/verify_roles.py` first
2. Check CloudWatch logs
3. Verify AWS credentials
4. Contact admin team if role issues persist
'''

def get_architecture_doc():
    return '''# MLOps Architecture

## Overview

The Energy Forecasting MLOps pipeline consists of three main steps:

1. **Containerized Training** (Step 1)
2. **Model Registry** (Step 2)
3. **Endpoint Management** (Step 3)

## Step 1: Containerized Training

- **Preprocessing Container**: Data cleaning, feature engineering
- **Training Container**: XGBoost model training with hyperparameter tuning
- **Triggers**: Weekly via EventBridge
- **Output**: Models and metadata to S3

## Step 2: Model Registry

- **Lambda Function**: Validates and registers models
- **Triggers**: Automatically after training completion
- **Process**: Performance validation → Registry → Approval
- **Output**: Versioned models in SageMaker Model Registry

## Step 3: Endpoint Management

- **Lambda Function**: Creates, tests, and deletes endpoints
- **Triggers**: Automatically after model registration
- **Process**: Create → Test → Save Config → Delete
- **Output**: Endpoint configurations saved to S3

## Cost Optimization

- Endpoints deleted after configuration save
- 98% cost reduction vs always-on endpoints
- Pay only for actual compute time

## Integration Points

- Training container triggers Model Registry Lambda
- Model Registry Lambda triggers Endpoint Management Lambda
- All configurations saved for daily prediction recreation

## Admin Dependencies

- IAM roles must be pre-created by admin team
- ECR repositories created by admin team
- Additional permissions granted to data scientist role
'''

def get_aws_setup_script():
    return '''#!/bin/bash
# AWS Setup Script

echo "Setting up AWS credentials and region..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI not found. Please install it first."
    exit 1
fi

# Configure AWS credentials if not already done
echo "Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "AWS credentials not configured. Running aws configure..."
    aws configure
else
    echo "✓ AWS credentials already configured"
    aws sts get-caller-identity
fi

# Set default region
export AWS_DEFAULT_REGION=us-west-2
echo "✓ AWS region set to us-west-2"

# Verify admin setup
echo "Verifying admin setup..."
python scripts/verify_roles.py

echo "AWS setup complete!"
'''

def get_deploy_all_script():
    return '''#!/bin/bash
# Deploy all containers script

echo "🚀 Deploying all Energy Forecasting containers..."

# Verify prerequisites first
echo "Verifying prerequisites..."
python scripts/verify_roles.py
if [ $? -ne 0 ]; then
    echo "❌ Prerequisites not met. Please complete admin setup first."
    exit 1
fi

# Build and push preprocessing container
echo "Building preprocessing container..."
cd containers/preprocessing
bash build_and_push.sh
cd ../..

# Build and push training container
echo "Building training container..."
cd containers/training
bash build_and_push.sh
cd ../..

# Note: Prediction container is for future use
echo "Note: Prediction container deployment skipped (for future daily predictions)"

echo "✅ All containers deployed successfully!"
'''

def get_test_script():
    return '''#!/bin/bash
# Test pipeline script

echo "🧪 Testing Energy Forecasting MLOps Pipeline..."

# Verify prerequisites
echo "Verifying prerequisites..."
python scripts/verify_roles.py
if [ $? -ne 0 ]; then
    echo "❌ Prerequisites not met. Please complete admin setup first."
    exit 1
fi

# Test infrastructure
echo "Testing infrastructure..."
python infrastructure/setup_infrastructure.py --test-only

# Test Lambda functions
echo "Testing Lambda functions..."
python deployment/lambda_deployer.py --test-only

# Test training pipeline
echo "Testing training pipeline..."
python deployment/test_pipeline.py

echo "✅ Pipeline testing complete!"
'''

def get_cleanup_script():
    return '''#!/bin/bash
# Cleanup script for removing AWS resources

echo "🧹 Cleaning up Energy Forecasting resources..."

# Delete Step Functions
aws stepfunctions delete-state-machine --state-machine-arn $(aws stepfunctions list-state-machines --query 'stateMachines[?name==`energy-forecasting-training-pipeline`].stateMachineArn' --output text)

# Delete Lambda functions
aws lambda delete-function --function-name energy-forecasting-model-registry
aws lambda delete-function --function-name energy-forecasting-endpoint-management

# Delete EventBridge rules
aws events delete-rule --name energy-forecasting-weekly-training
aws events delete-rule --name energy-forecasting-daily-predictions

echo "⚠️  Note: ECR repositories and S3 buckets not deleted (contains data)"
echo "⚠️  Note: IAM roles not deleted (managed by admin team)"
echo "✅ Cleanup complete!"
'''

if __name__ == "__main__":
    main()
