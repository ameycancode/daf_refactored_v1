# Energy Forecasting MLOps Pipeline

Complete MLOps pipeline for energy load forecasting with model registry and cost optimization.

## Architecture

- **Step 1**: Containerized Training (Preprocessing + Model Training)
- **Step 2**: Model Registry (Lambda function for model validation and registration)
- **Step 3**: Endpoint Management (Lambda function for cost-optimized endpoint lifecycle)

## Prerequisites

⚠️ **IMPORTANT**: Admin team must create IAM roles first!

1. Verify setup: `python scripts/verify_roles.py`

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
