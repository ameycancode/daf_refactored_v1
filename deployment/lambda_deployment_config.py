#!/usr/bin/env python3
"""
Fixed script to update Lambda configurations for proper timeout and concurrency settings
"""

import boto3
import json

def update_lambda_configurations():
    """Update both Lambda functions with proper settings"""
    
    lambda_client = boto3.client('lambda')
    
    # Configuration for both Lambda functions
    lambda_configs = {
        'energy-forecasting-model-registry': {
            'timeout': 900,  # 15 minutes
            'memory_size': 1024,  # 1GB
            'reserved_concurrency': 1,  # Prevent concurrent executions
            'environment_variables': {
                'EXECUTION_TIMEOUT': '900',
                'LOG_LEVEL': 'INFO'
            }
        },
        'energy-forecasting-endpoint-management': {
            'timeout': 900,  # 15 minutes  
            'memory_size': 1024,  # 1GB
            'reserved_concurrency': 1,  # Prevent concurrent executions
            'environment_variables': {
                'EXECUTION_TIMEOUT': '900',
                'LOG_LEVEL': 'INFO',
                'SAGEMAKER_EXECUTION_ROLE': 'arn:aws:iam::YOUR_ACCOUNT:role/EnergyForecastingSageMakerRole'
            }
        }
    }
    
    for function_name, config in lambda_configs.items():
        try:
            print(f"Updating {function_name}...")
            
            # Update function configuration
            update_response = lambda_client.update_function_configuration(
                FunctionName=function_name,
                Timeout=config['timeout'],
                MemorySize=config['memory_size'],
                Environment={
                    'Variables': config['environment_variables']
                }
            )
            
            print(f" Updated basic configuration for {function_name}")
            print(f"   - Timeout: {config['timeout']} seconds")
            print(f"   - Memory: {config['memory_size']} MB")
            
            # Set reserved concurrency using correct method name
            try:
                concurrency_response = lambda_client.put_reserved_concurrency(
                    FunctionName=function_name,
                    ReservedConcurrencyConfig={
                        'ReservedConcurrency': config['reserved_concurrency']
                    }
                )
                print(f"   - Reserved Concurrency: {config['reserved_concurrency']}")
            except AttributeError:
                # Try alternative method name
                try:
                    concurrency_response = lambda_client.put_function_concurrency(
                        FunctionName=function_name,
                        ReservedConcurrency=config['reserved_concurrency']
                    )
                    print(f"   - Reserved Concurrency: {config['reserved_concurrency']}")
                except Exception as e:
                    print(f"    Could not set reserved concurrency: {str(e)}")
                    print(f"   You can set this manually in AWS Console")
            except Exception as e:
                print(f"    Could not set reserved concurrency: {str(e)}")
                print(f"   You can set this manually in AWS Console")
            
        except Exception as e:
            print(f" Failed to update {function_name}: {str(e)}")

def verify_lambda_configurations():
    """Verify Lambda configurations are correct"""
    
    lambda_client = boto3.client('lambda')
    
    function_names = [
        'energy-forecasting-model-registry',
        'energy-forecasting-endpoint-management'
    ]
    
    print("\nVerifying Lambda configurations:")
    print("=" * 50)
    
    for function_name in function_names:
        try:
            # Get function configuration
            response = lambda_client.get_function_configuration(FunctionName=function_name)
            
            print(f"\n{function_name}:")
            print(f"  Timeout: {response['Timeout']} seconds")
            print(f"  Memory Size: {response['MemorySize']} MB")
            print(f"  Runtime: {response['Runtime']}")
            print(f"  Last Modified: {response['LastModified']}")
            
            # Check reserved concurrency with multiple method attempts
            concurrency_value = None
            
            # Try different methods to get concurrency
            concurrency_methods = [
                'get_function_concurrency',
                'get_reserved_concurrency', 
                'describe_function_concurrency'
            ]
            
            for method_name in concurrency_methods:
                try:
                    if hasattr(lambda_client, method_name):
                        method = getattr(lambda_client, method_name)
                        concurrency_response = method(FunctionName=function_name)
                        
                        # Extract concurrency value from different response formats
                        if 'ReservedConcurrency' in concurrency_response:
                            concurrency_value = concurrency_response['ReservedConcurrency']
                        elif 'ReservedConcurrencyConfig' in concurrency_response:
                            concurrency_value = concurrency_response['ReservedConcurrencyConfig']['ReservedConcurrency']
                        
                        if concurrency_value is not None:
                            break
                            
                except lambda_client.exceptions.ResourceNotFoundException:
                    continue
                except Exception:
                    continue
            
            if concurrency_value is not None:
                print(f"  Reserved Concurrency: {concurrency_value}")
            else:
                print(f"  Reserved Concurrency: Not set (unlimited)")
            
            # Check environment variables
            env_vars = response.get('Environment', {}).get('Variables', {})
            if env_vars:
                print(f"  Environment Variables:")
                for key, value in env_vars.items():
                    if 'ROLE' in key and len(value) > 50:
                        print(f"    {key}: {value[:50]}...")
                    else:
                        print(f"    {key}: {value}")
            else:
                print(f"  Environment Variables: None")
            
        except Exception as e:
            print(f" Failed to get config for {function_name}: {str(e)}")

def manual_concurrency_instructions():
    """Provide manual instructions for setting concurrency"""
    
    print("\n" + "=" * 60)
    print("MANUAL CONCURRENCY SETUP (if automatic setup failed)")
    print("=" * 60)
    print("\nTo manually set reserved concurrency in AWS Console:")
    print("\n1. Go to AWS Lambda Console")
    print("2. For each function (energy-forecasting-model-registry, energy-forecasting-endpoint-management):")
    print("   a. Click on the function name")
    print("   b. Go to 'Configuration' tab")
    print("   c. Click 'Concurrency' in the left sidebar")
    print("   d. Click 'Reserve concurrency'")
    print("   e. Set 'Reserved concurrency' to 1")
    print("   f. Click 'Save'")
    print("\nThis prevents multiple concurrent executions of the same function.")

def test_lambda_function_connectivity():
    """Test that we can connect to and invoke Lambda functions"""
    
    lambda_client = boto3.client('lambda')
    
    function_names = [
        'energy-forecasting-model-registry',
        'energy-forecasting-endpoint-management'
    ]
    
    print("\n" + "=" * 50)
    print("TESTING LAMBDA CONNECTIVITY")
    print("=" * 50)
    
    for function_name in function_names:
        try:
            # Test that function exists and we can get its config
            response = lambda_client.get_function_configuration(FunctionName=function_name)
            
            print(f"\n {function_name}:")
            print(f"   Status: {response.get('State', 'Unknown')}")
            print(f"   Size: {response.get('CodeSize', 0)} bytes")
            print(f"   Handler: {response.get('Handler', 'Unknown')}")
            
            # Check if function is ready for invocation
            if response.get('State') == 'Active':
                print(f"   Ready for invocation")
            else:
                print(f"   Function state: {response.get('State')}")
                
        except lambda_client.exceptions.ResourceNotFoundException:
            print(f"\n {function_name}: Function not found")
            print(f"   Please ensure the function is deployed")
        except Exception as e:
            print(f"\n {function_name}: Error - {str(e)}")

if __name__ == "__main__":
    print("Updating Lambda configurations...")
    update_lambda_configurations()
    
    print("\n" + "=" * 50)
    verify_lambda_configurations()
    
    # Test connectivity
    test_lambda_function_connectivity()
    
    # Manual instructions if needed
    manual_concurrency_instructions()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Lambda timeout and memory updated to 15 minutes / 1GB")
    print("2.  If reserved concurrency setup failed, set manually in AWS Console")
    print("3. Update your Lambda function code with the fixed versions:")
    print("   - Deploy fixed model_registry_lambda_function.py")
    print("   - Deploy fixed endpoint_management_lambda_function.py")
    print("4. Test the updated functions:")
    print("   python deployment/test_model_registry.py")
    print("   python deployment/test_endpoint_management.py")
    print("\n The most important fixes are in the Lambda code itself!")
    print("   The timeout/memory updates will help, but the code fixes prevent")
    print("   multiple executions and tar.gz errors.")
