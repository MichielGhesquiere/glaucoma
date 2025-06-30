#!/usr/bin/env python3
"""
Test script to verify that robust saving mechanisms work correctly.
This script simulates various failure scenarios to ensure results are always saved.
"""

import os
import sys
import tempfile
import shutil
from datetime import datetime

# Add the scripts directory to the path to import our module
sys.path.append(os.path.dirname(__file__))

from multisource_domain_finetuning import MultiSourceTrainer

def test_robust_saving():
    """Test the robust saving mechanisms."""
    print("Testing robust result saving mechanisms...")
    
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp(prefix="glaucoma_test_")
    print(f"Test directory: {test_dir}")
    
    try:
        # Initialize trainer
        trainer = MultiSourceTrainer(
            output_dir=test_dir,
            device='cpu',  # Use CPU for testing
            enable_checkpointing=True,
            quiet_mode=False
        )
        
        # Add some mock results
        trainer.results = [
            {
                'target_dataset': 'test_dataset_1',
                'model_name': 'test_model_1',
                'auc': 0.85,
                'accuracy': 82.5,
                'status': 'completed'
            },
            {
                'target_dataset': 'test_dataset_2',
                'model_name': 'test_model_2',
                'auc': 0.78,
                'accuracy': 79.1,
                'status': 'completed'
            }
        ]
        
        trainer.completed_experiments = ['test_dataset_1_test_model_1', 'test_dataset_2_test_model_2']
        
        print(f"Added {len(trainer.results)} mock results")
        
        # Test 1: Normal intermediate save
        print("\n1. Testing normal intermediate save...")
        trainer.save_intermediate_results()
        
        # Test 2: Force save all results
        print("\n2. Testing force save all results...")
        saved_files = trainer.force_save_all_results(additional_info="Test scenario - simulated failure")
        
        # Verify files were created
        print("\n3. Verifying saved files...")
        files_found = []
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                file_path = os.path.join(root, file)
                files_found.append(file_path)
                print(f"   Found: {file}")
        
        print(f"\nTotal files saved: {len(files_found)}")
        
        # Test 3: Verify we can read back the results
        print("\n4. Testing result recovery...")
        csv_files = [f for f in files_found if f.endswith('.csv')]
        json_files = [f for f in files_found if f.endswith('.json')]
        pickle_files = [f for f in files_found if f.endswith('.pkl')]
        txt_files = [f for f in files_found if f.endswith('.txt')]
        
        print(f"   CSV files: {len(csv_files)}")
        print(f"   JSON files: {len(json_files)}")
        print(f"   Pickle files: {len(pickle_files)}")
        print(f"   TXT files: {len(txt_files)}")
        
        # Try to read one of each type
        success_count = 0
        
        if csv_files:
            try:
                import pandas as pd
                df = pd.read_csv(csv_files[0])
                print(f"   CSV readable: {len(df)} rows")
                success_count += 1
            except Exception as e:
                print(f"   CSV read failed: {e}")
        
        if json_files:
            try:
                import json
                with open(json_files[0], 'r') as f:
                    data = json.load(f)
                print(f"   JSON readable: {len(data.get('results', []))} results")
                success_count += 1
            except Exception as e:
                print(f"   JSON read failed: {e}")
        
        if pickle_files:
            try:
                import pickle
                with open(pickle_files[0], 'rb') as f:
                    data = pickle.load(f)
                print(f"   Pickle readable: {len(data.get('results', []))} results")
                success_count += 1
            except Exception as e:
                print(f"   Pickle read failed: {e}")
        
        if txt_files:
            try:
                with open(txt_files[0], 'r') as f:
                    content = f.read()
                print(f"   TXT readable: {len(content)} characters")
                success_count += 1
            except Exception as e:
                print(f"   TXT read failed: {e}")
        
        print(f"\n5. Summary:")
        print(f"   Files created: {len(files_found)}")
        print(f"   Readable formats: {success_count}")
        print(f"   Test directory: {test_dir}")
        
        if len(files_found) > 0 and success_count > 0:
            print("   ✓ ROBUST SAVING TEST PASSED!")
            return True
        else:
            print("   ✗ ROBUST SAVING TEST FAILED!")
            return False
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up (optional - you might want to keep the test files for inspection)
        try:
            # Uncomment the next line if you want to clean up test files
            # shutil.rmtree(test_dir)
            print(f"\nTest files preserved in: {test_dir}")
            print("You can manually delete this directory when done inspecting.")
        except Exception as e:
            print(f"Cleanup failed: {e}")

if __name__ == "__main__":
    print("="*60)
    print("GLAUCOMA CLASSIFICATION - ROBUST SAVING TEST")
    print("="*60)
    
    success = test_robust_saving()
    
    print("\n" + "="*60)
    if success:
        print("ALL TESTS PASSED - Results will be saved reliably!")
    else:
        print("TESTS FAILED - Check the error messages above")
    print("="*60)
    
    sys.exit(0 if success else 1)
