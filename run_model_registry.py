#!/usr/bin/env python3
"""
Execute Step 2: Model Registry & Versioning
Simple script to run the model registry pipeline
"""

import sys
import os
from datetime import datetime

# Add the current directory to path for importing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deployment.model_registry_manager import ModelRegistryManager

def main():
    """Execute Step 2: Model Registry & Versioning"""
    print("="*60)
    print("STEP 2: MODEL REGISTRY & VERSIONING")
    print("="*60)
    print(f"Started at: {datetime.now().isoformat()}")
    print()
   
    try:
        # Initialize the model registry manager
        registry_manager = ModelRegistryManager()
       
        # Run the complete pipeline
        result = registry_manager.run_model_registry_pipeline()
       
        if result:
            print("\n" + "="*60)
            print("STEP 2 COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("✅ Models found and registered in SageMaker Model Registry")
            print("✅ Model performance validated")
            print("✅ Models approved for deployment")
            print("✅ Summary report generated")
            print("\nNext Steps:")
            print("- Run Step 3: Endpoint Management")
            print("- Check model_registry_summary_*.json for detailed results")
            print("- Verify models in SageMaker Model Registry console")
            return True
        else:
            print("\n" + "="*60)
            print("STEP 2 FAILED!")
            print("="*60)
            print("❌ Check logs for detailed error information")
            print("❌ Ensure training step completed successfully")
            print("❌ Verify S3 permissions and model files exist")
            return False
           
    except Exception as e:
        print(f"\nSTEP 2 EXECUTION ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
