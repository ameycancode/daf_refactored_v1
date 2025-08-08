#!/usr/bin/env python3
"""
Step 3: Endpoint Management Implementation
This script handles endpoint creation, configuration storage, and deletion
Following the same pattern as the working implementation
"""

import os
import sys
import boto3
import json
import time
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EndpointManager:
    def __init__(self, region="us-west-2"):
        self.region = region
        self.account_id = boto3.client('sts').get_caller_identity()['Account']
        
        # Initialize clients
        self.sagemaker_client = boto3.client('sagemaker')
        self.s3_client = boto3.client('s3')
        
        # Configuration
        self.config = {
            "model_bucket": "sdcp-dev-sagemaker-energy-forecasting-models",
            "config_prefix": "endpoint-configs/",
            "customer_profile": "energy-forecasting",
            "customer_segment": "load-prediction",
            "profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"],
            "role_arn": f"arn:aws:iam::{self.account_id}:role/sdcp-dev-sagemaker-energy-forecasting-datascientist-role",
            "instance_type": "ml.m5.large",
            "instance_count": 1
        }
        
        # Get current date
        self.current_date = datetime.now().strftime("%Y%m%d")
        
    def run_endpoint_management_pipeline(self):
        """Main pipeline for endpoint management operations"""
        try:
            logger.info("Starting Step 3: Endpoint Management Pipeline...")
            logger.info(f"Processing endpoints for date: {self.current_date}")
            
            # Step 1: Get registered models from Model Registry
            logger.info("Step 1: Finding registered models...")
            registered_models = self._find_registered_models()
            
            if not registered_models:
                raise Exception("No registered models found in Model Registry")
            
            # Step 2: Process each model - Create, Store Config, Delete
            logger.info("Step 2: Creating and managing endpoints...")
            endpoint_results = {}
            
            for profile, model_info in registered_models.items():
                try:
                    logger.info(f"Processing endpoint for profile: {profile}")
                    result = self._process_endpoint_lifecycle(profile, model_info)
                    endpoint_results[profile] = result
                    logger.info(f"✓ Successfully processed {profile} endpoint")
                    
                except Exception as e:
                    logger.error(f"✗ Failed to process {profile} endpoint: {str(e)}")
                    endpoint_results[profile] = {"error": str(e)}
            
            # Step 3: Generate summary
            logger.info("Step 3: Generating endpoint management summary...")
            summary = self._generate_endpoint_summary(endpoint_results)
            
            logger.info("Endpoint Management Pipeline completed successfully!")
            return summary
            
        except Exception as e:
            logger.error(f"Endpoint Management Pipeline failed: {str(e)}")
            raise
    
    def _find_registered_models(self):
        """Find registered models from Model Registry for each profile"""
        registered_models = {}
        
        try:
            for profile in self.config['profiles']:
                group_name = f"EnergyForecast-{profile}-{self.config['customer_profile']}-{self.config['customer_segment']}"
                
                try:
                    # Get latest model package from group
                    response = self.sagemaker_client.list_model_packages(
                        ModelPackageGroupName=group_name,
                        SortBy='CreationTime',
                        SortOrder='Descending',
                        MaxResults=1
                    )
                    
                    model_packages = response.get('ModelPackageSummaryList', [])
                    
                    if model_packages:
                        latest_model = model_packages[0]
                        
                        # Only use approved models
                        if latest_model['ModelApprovalStatus'] == 'Approved':
                            registered_models[profile] = {
                                'model_package_arn': latest_model['ModelPackageArn'],
                                'version': latest_model['ModelPackageVersion'],
                                'status': latest_model['ModelPackageStatus'],
                                'creation_time': latest_model['CreationTime']
                            }
                            logger.info(f"Found approved model for {profile}: version {latest_model['ModelPackageVersion']}")
                        else:
                            logger.warning(f"Model for {profile} not approved: {latest_model['ModelApprovalStatus']}")
                    else:
                        logger.warning(f"No model packages found for profile: {profile}")
                        
                except Exception as e:
                    logger.error(f"Error finding model for {profile}: {str(e)}")
                    continue
            
            logger.info(f"Found {len(registered_models)} approved models for endpoint creation")
            return registered_models
            
        except Exception as e:
            logger.error(f"Failed to find registered models: {str(e)}")
            raise
    
    def _process_endpoint_lifecycle(self, profile, model_info):
        """Process complete endpoint lifecycle: Create → Store Config → Delete"""
        try:
            model_package_arn = model_info['model_package_arn']
            
            # Step 1: Create Model
            model_name = self._create_model(profile, model_package_arn)
            
            # Step 2: Create Endpoint Configuration
            endpoint_config_name = self._create_endpoint_config(profile, model_name)
            
            # Step 3: Create Endpoint
            endpoint_name = self._create_endpoint(profile, endpoint_config_name)
            
            # Step 4: Wait for Endpoint to be InService
            self._wait_for_endpoint_ready(endpoint_name)
            
            # Step 5: Store Endpoint Configuration
            config_s3_key = self._store_endpoint_configuration(
                profile, endpoint_name, endpoint_config_name, model_name, model_package_arn
            )
            
            # Step 6: Delete Endpoint (keep config and model for recreation)
            self._delete_endpoint_for_cost_optimization(endpoint_name)
            
            return {
                "status": "success",
                "profile": profile,
                "model_package_arn": model_package_arn,
                "model_name": model_name,
                "endpoint_config_name": endpoint_config_name,
                "endpoint_name": endpoint_name,
                "config_s3_key": config_s3_key,
                "endpoint_deleted": True
            }
            
        except Exception as e:
            logger.error(f"Failed to process endpoint lifecycle for {profile}: {str(e)}")
            raise
    
    def _create_model(self, profile, model_package_arn):
        """Create SageMaker Model from Model Package"""
        try:
            model_name = f"energy-forecast-{profile}-{self.current_date}-{int(time.time())}"
            
            # Get model package details to extract container info
            package_details = self.sagemaker_client.describe_model_package(
                ModelPackageName=model_package_arn
            )
            
            # Extract container specification
            containers = package_details['InferenceSpecification']['Containers']
            primary_container = containers[0]  # Use first container
            
            # Create model
            self.sagemaker_client.create_model(
                ModelName=model_name,
                ExecutionRoleArn=self.config['role_arn'],
                PrimaryContainer={
                    'ModelDataUrl': primary_container['ModelDataUrl'],
                    'Image': primary_container['Image'],
                    'Environment': {}
                },
                Tags=[
                    {'Key': 'Profile', 'Value': profile},
                    {'Key': 'CreatedDate', 'Value': self.current_date},
                    {'Key': 'UseCase', 'Value': 'energy-load-forecasting'}
                ]
            )
            
            logger.info(f"✓ Created model: {model_name}")
            return model_name
            
        except Exception as e:
            logger.error(f"Failed to create model for {profile}: {str(e)}")
            raise
    
    def _create_endpoint_config(self, profile, model_name):
        """Create Endpoint Configuration"""
        try:
            endpoint_config_name = f"energy-forecast-{profile}-config-{self.current_date}-{int(time.time())}"
            
            self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'AllTraffic',
                        'ModelName': model_name,
                        'InstanceType': self.config['instance_type'],
                        'InitialInstanceCount': self.config['instance_count'],
                        'InitialVariantWeight': 1.0
                    }
                ],
                Tags=[
                    {'Key': 'Profile', 'Value': profile},
                    {'Key': 'CreatedDate', 'Value': self.current_date},
                    {'Key': 'UseCase', 'Value': 'energy-load-forecasting'}
                ]
            )
            
            logger.info(f"✓ Created endpoint config: {endpoint_config_name}")
            return endpoint_config_name
            
        except Exception as e:
            logger.error(f"Failed to create endpoint config for {profile}: {str(e)}")
            raise
    
    def _create_endpoint(self, profile, endpoint_config_name):
        """Create SageMaker Endpoint"""
        try:
            endpoint_name = f"energy-forecast-{profile}-{self.current_date}"
            
            # Check if endpoint already exists
            try:
                self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                logger.warning(f"Endpoint {endpoint_name} already exists, deleting first...")
                self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                time.sleep(30)  # Wait for deletion
            except self.sagemaker_client.exceptions.ClientError:
                pass  # Endpoint doesn't exist, proceed
            
            # Create endpoint
            self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
                Tags=[
                    {'Key': 'Profile', 'Value': profile},
                    {'Key': 'CreatedDate', 'Value': self.current_date},
                    {'Key': 'UseCase', 'Value': 'energy-load-forecasting'},
                    {'Key': 'CostOptimized', 'Value': 'true'}
                ]
            )
            
            logger.info(f"✓ Created endpoint: {endpoint_name}")
            return endpoint_name
            
        except Exception as e:
            logger.error(f"Failed to create endpoint for {profile}: {str(e)}")
            raise
    
    def _wait_for_endpoint_ready(self, endpoint_name, max_wait_time=1200):
        """Wait for endpoint to be InService"""
        try:
            logger.info(f"Waiting for endpoint {endpoint_name} to be ready...")
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                response = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                status = response['EndpointStatus']
                
                if status == 'InService':
                    logger.info(f"✓ Endpoint {endpoint_name} is ready ({status})")
                    return True
                elif status in ['Failed', 'OutOfService']:
                    error_msg = response.get('FailureReason', 'Unknown error')
                    raise Exception(f"Endpoint creation failed: {status} - {error_msg}")
                else:
                    logger.info(f"Endpoint status: {status}, waiting...")
                    time.sleep(30)
            
            raise Exception(f"Timeout waiting for endpoint {endpoint_name} to be ready")
            
        except Exception as e:
            logger.error(f"Error waiting for endpoint {endpoint_name}: {str(e)}")
            raise
    
    def _store_endpoint_configuration(self, profile, endpoint_name, endpoint_config_name, model_name, model_package_arn):
        """Store endpoint configuration to S3 for later recreation"""
        try:
            logger.info(f"Storing endpoint configuration for {profile}...")
            
            # Get detailed configuration information
            endpoint_details = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            config_details = self.sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
            model_details = self.sagemaker_client.describe_model(ModelName=model_name)
            
            # Create comprehensive configuration
            endpoint_configuration = {
                "profile": profile,
                "created_date": self.current_date,
                "created_timestamp": datetime.now().isoformat(),
                
                # Endpoint Information
                "endpoint_name": endpoint_name,
                "endpoint_arn": endpoint_details['EndpointArn'],
                "endpoint_status": endpoint_details['EndpointStatus'],
                
                # Model Information
                "model_name": model_name,
                "model_package_arn": model_package_arn,
                "model_arn": model_details['ModelArn'],
                
                # Configuration Information
                "endpoint_config_name": endpoint_config_name,
                "endpoint_config_arn": config_details['EndpointConfigArn'],
                "production_variants": config_details['ProductionVariants'],
                
                # Recreation Information
                "recreation_config": {
                    "instance_type": self.config['instance_type'],
                    "instance_count": self.config['instance_count'],
                    "execution_role_arn": self.config['role_arn'],
                    "model_data_url": model_details['PrimaryContainer']['ModelDataUrl'],
                    "image": model_details['PrimaryContainer']['Image']
                },
                
                # Metadata
                "cost_optimized": True,
                "auto_delete_enabled": True,
                "recreation_enabled": True,
                "tags": {
                    "Profile": profile,
                    "UseCase": "energy-load-forecasting",
                    "Environment": "production",
                    "CostOptimized": "true"
                }
            }
            
            # Save to S3
            s3_key = f"{self.config['config_prefix']}{profile}_endpoint_config_{self.current_date}.json"
            
            self.s3_client.put_object(
                Bucket=self.config['model_bucket'],
                Key=s3_key,
                Body=json.dumps(endpoint_configuration, indent=2, default=str),
                ContentType='application/json',
                Metadata={
                    'profile': profile,
                    'created_date': self.current_date,
                    'config_type': 'endpoint_configuration'
                }
            )
            
            logger.info(f"✓ Stored endpoint configuration: s3://{self.config['model_bucket']}/{s3_key}")
            return s3_key
            
        except Exception as e:
            logger.error(f"Failed to store endpoint configuration for {profile}: {str(e)}")
            raise
    
    def _delete_endpoint_for_cost_optimization(self, endpoint_name):
        """Delete endpoint to save costs while keeping configuration for recreation"""
        try:
            logger.info(f"Deleting endpoint {endpoint_name} for cost optimization...")
            
            # Verify endpoint exists and is InService
            response = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            if response['EndpointStatus'] != 'InService':
                logger.warning(f"Endpoint {endpoint_name} is not InService, status: {response['EndpointStatus']}")
            
            # Delete endpoint
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            
            logger.info(f"✓ Deleted endpoint {endpoint_name} - configuration saved for recreation")
            logger.info(f" Cost optimization: Endpoint now incurs ZERO costs")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete endpoint {endpoint_name}: {str(e)}")
            raise
    
    def _generate_endpoint_summary(self, endpoint_results):
        """Generate comprehensive endpoint management summary"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "processing_date": self.current_date,
            "total_profiles": len(self.config['profiles']),
            "successful_endpoints": len([r for r in endpoint_results.values() if 'error' not in r]),
            "failed_endpoints": len([r for r in endpoint_results.values() if 'error' in r]),
            "configurations_stored": len([r for r in endpoint_results.values() if 'error' not in r and r.get('config_s3_key')]),
            "endpoints_deleted": len([r for r in endpoint_results.values() if 'error' not in r and r.get('endpoint_deleted')]),
            "results": endpoint_results
        }
        
        # Save summary to file
        summary_file = f"endpoint_management_summary_{self.current_date}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Upload summary to S3
        s3_key = f"{self.config['config_prefix']}summaries/endpoint_summary_{self.current_date}.json"
        self.s3_client.upload_file(
            summary_file,
            self.config['model_bucket'],
            s3_key
        )
        
        logger.info(f"Endpoint management summary saved: {summary_file}")
        
        # Print summary
        logger.info("="*60)
        logger.info("ENDPOINT MANAGEMENT SUMMARY")
        logger.info("="*60)
        logger.info(f"Total profiles: {summary['total_profiles']}")
        logger.info(f"Successful endpoints: {summary['successful_endpoints']}")
        logger.info(f"Failed endpoints: {summary['failed_endpoints']}")
        logger.info(f"Configurations stored: {summary['configurations_stored']}")
        logger.info(f"Endpoints deleted for cost optimization: {summary['endpoints_deleted']}")
        
        for profile, result in endpoint_results.items():
            if 'error' in result:
                logger.info(f" {profile}: FAILED - {result['error']}")
            else:
                logger.info(f" {profile}: SUCCESS")
                logger.info(f"   ├── Model: {result['model_name']}")
                logger.info(f"   ├── Endpoint: {result['endpoint_name']} (DELETED)")
                logger.info(f"   └── Config: {result['config_s3_key']}")
        
        # Cost optimization summary
        if summary['endpoints_deleted'] > 0:
            logger.info(f"\n COST OPTIMIZATION SUCCESS!")
            logger.info(f"   └── {summary['endpoints_deleted']} endpoints deleted")
            logger.info(f"   └── Configurations saved for recreation")
            logger.info(f"   └── Current endpoint costs: $0.00/hour")
        
        return summary

def main():
    """Main function to run Step 3: Endpoint Management"""
    try:
        logger.info(" Starting Step 3: Endpoint Management")
        
        # Initialize manager
        endpoint_manager = EndpointManager()
        
        # Run the pipeline
        summary = endpoint_manager.run_endpoint_management_pipeline()
        
        logger.info(" Step 3 completed successfully!")
        logger.info(f"Check endpoint_management_summary_{endpoint_manager.current_date}.json for details")
        
        return summary
        
    except Exception as e:
        logger.error(f" Step 3 failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
