#!/usr/bin/env python3
"""
Result Recovery Script for Glaucoma Classification Experiments
This script helps you find and recover results from experiments,
even if they were interrupted or failed.
"""

import os
import glob
import json
import pickle
import pandas as pd
from datetime import datetime
import argparse


def find_all_result_files(base_dir):
    """Find all result files in the base directory."""
    result_files = {
        'csv': [],
        'json': [],
        'pickle': [],
        'txt': [],
        'models': []
    }
    
    # Search patterns
    patterns = {
        'csv': ['*results*.csv', '*EMERGENCY*.csv'],
        'json': ['*results*.json', '*EMERGENCY*.json'],
        'pickle': ['*results*.pkl', '*EMERGENCY*.pkl'],
        'txt': ['*results*.txt', '*EMERGENCY*.txt', '*INDEX*.txt'],
        'models': ['model_*.pth', '*.pth']
    }
    
    for root, dirs, files in os.walk(base_dir):
        for file_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = glob.glob(os.path.join(root, pattern))
                result_files[file_type].extend(matches)
    
    # Remove duplicates
    for file_type in result_files:
        result_files[file_type] = list(set(result_files[file_type]))
        result_files[file_type].sort()
    
    return result_files


def analyze_csv_results(csv_file):
    """Analyze CSV results file."""
    try:
        df = pd.read_csv(csv_file)
        
        analysis = {
            'file': csv_file,
            'total_experiments': len(df),
            'unique_models': df['model_name'].nunique() if 'model_name' in df.columns else 0,
            'unique_datasets': df['target_dataset'].nunique() if 'target_dataset' in df.columns else 0,
            'avg_auc': df['auc'].mean() if 'auc' in df.columns else None,
            'best_auc': df['auc'].max() if 'auc' in df.columns else None,
            'columns': list(df.columns)
        }
        
        if 'model_name' in df.columns:
            analysis['models'] = df['model_name'].unique().tolist()
        if 'target_dataset' in df.columns:
            analysis['datasets'] = df['target_dataset'].unique().tolist()
            
        return analysis, df
    except Exception as e:
        return {'file': csv_file, 'error': str(e)}, None


def analyze_json_results(json_file):
    """Analyze JSON results file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        analysis = {
            'file': json_file,
            'total_experiments': len(results),
            'completed_experiments': len(data.get('completed_experiments', [])),
            'timestamp': data.get('timestamp', 'Unknown'),
            'additional_info': data.get('additional_info', None)
        }
        
        if results:
            aucs = [r.get('auc', 0) for r in results if 'auc' in r and r['auc'] is not None]
            if aucs:
                analysis['avg_auc'] = sum(aucs) / len(aucs)
                analysis['best_auc'] = max(aucs)
            
            models = list(set(r.get('model_name', 'Unknown') for r in results))
            datasets = list(set(r.get('target_dataset', 'Unknown') for r in results))
            analysis['models'] = models
            analysis['datasets'] = datasets
        
        return analysis, data
    except Exception as e:
        return {'file': json_file, 'error': str(e)}, None


def analyze_pickle_results(pickle_file):
    """Analyze pickle results file."""
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        results = data.get('results', [])
        analysis = {
            'file': pickle_file,
            'total_experiments': len(results),
            'completed_experiments': len(data.get('completed_experiments', [])),
            'timestamp': data.get('timestamp', 'Unknown')
        }
        
        if results:
            aucs = [r.get('auc', 0) for r in results if 'auc' in r and r['auc'] is not None]
            if aucs:
                analysis['avg_auc'] = sum(aucs) / len(aucs)
                analysis['best_auc'] = max(aucs)
        
        return analysis, data
    except Exception as e:
        return {'file': pickle_file, 'error': str(e)}, None


def print_summary(result_files):
    """Print a summary of all found files."""
    print("="*80)
    print("GLAUCOMA CLASSIFICATION EXPERIMENT - RESULT RECOVERY")
    print("="*80)
    
    total_files = sum(len(files) for files in result_files.values())
    print(f"Total files found: {total_files}")
    
    for file_type, files in result_files.items():
        if files:
            print(f"\n{file_type.upper()} files ({len(files)}):")
            for file in files:
                print(f"  {file}")
    
    if not total_files:
        print("No result files found.")
        return
    
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    # Analyze CSV files
    best_results = []
    
    for csv_file in result_files['csv']:
        print(f"\nAnalyzing CSV: {os.path.basename(csv_file)}")
        analysis, df = analyze_csv_results(csv_file)
        
        if 'error' in analysis:
            print(f"  Error: {analysis['error']}")
            continue
        
        print(f"  Total experiments: {analysis['total_experiments']}")
        print(f"  Unique models: {analysis['unique_models']}")
        print(f"  Unique datasets: {analysis['unique_datasets']}")
        
        if analysis['avg_auc'] is not None:
            print(f"  Average AUC: {analysis['avg_auc']:.4f}")
            print(f"  Best AUC: {analysis['best_auc']:.4f}")
            best_results.append({
                'file': csv_file,
                'best_auc': analysis['best_auc'],
                'type': 'CSV'
            })
        
        if 'models' in analysis:
            print(f"  Models: {', '.join(analysis['models'])}")
        if 'datasets' in analysis:
            print(f"  Datasets: {', '.join(analysis['datasets'])}")
    
    # Analyze JSON files
    for json_file in result_files['json']:
        print(f"\nAnalyzing JSON: {os.path.basename(json_file)}")
        analysis, data = analyze_json_results(json_file)
        
        if 'error' in analysis:
            print(f"  Error: {analysis['error']}")
            continue
        
        print(f"  Total experiments: {analysis['total_experiments']}")
        print(f"  Completed experiments: {analysis['completed_experiments']}")
        print(f"  Timestamp: {analysis['timestamp']}")
        
        if analysis.get('additional_info'):
            print(f"  Additional info: {analysis['additional_info']}")
        
        if 'avg_auc' in analysis:
            print(f"  Average AUC: {analysis['avg_auc']:.4f}")
            print(f"  Best AUC: {analysis['best_auc']:.4f}")
            best_results.append({
                'file': json_file,
                'best_auc': analysis['best_auc'],
                'type': 'JSON'
            })
    
    # Show best results
    if best_results:
        print(f"\n" + "="*80)
        print("BEST RESULTS SUMMARY")
        print("="*80)
        
        best_results.sort(key=lambda x: x['best_auc'], reverse=True)
        
        for i, result in enumerate(best_results):
            print(f"{i+1}. {result['type']}: {os.path.basename(result['file'])}")
            print(f"   Best AUC: {result['best_auc']:.4f}")
            print(f"   Full path: {result['file']}")
    
    # Show model files
    if result_files['models']:
        print(f"\n" + "="*80)
        print(f"SAVED MODELS ({len(result_files['models'])})")
        print("="*80)
        
        for model_file in result_files['models']:
            print(f"  {model_file}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if result_files['csv']:
        print("1. For analysis, use the CSV files with pandas:")
        print(f"   import pandas as pd")
        print(f"   df = pd.read_csv('{result_files['csv'][0]}')")
        print(f"   print(df.head())")
    
    if result_files['json']:
        print("2. For detailed data, use the JSON files:")
        print(f"   import json")
        print(f"   with open('{result_files['json'][0]}', 'r') as f:")
        print(f"       data = json.load(f)")
    
    if result_files['models']:
        print("3. To load trained models:")
        print(f"   import torch")
        print(f"   checkpoint = torch.load('{result_files['models'][0]}')")
        print(f"   model.load_state_dict(checkpoint['model_state_dict'])")


def main():
    parser = argparse.ArgumentParser(description="Recover and analyze glaucoma classification experiment results")
    parser.add_argument('--search_dir', type=str, default='./multisource_results', 
                       help="Directory to search for results")
    parser.add_argument('--output_dir', type=str, default='.',
                       help="Alternative search directory (for compatibility)")
    
    args = parser.parse_args()
    
    # Try both search directories
    search_dirs = []
    if os.path.exists(args.search_dir):
        search_dirs.append(args.search_dir)
    if os.path.exists(args.output_dir) and args.output_dir != args.search_dir:
        search_dirs.append(args.output_dir)
    
    # Also try common result directories
    common_dirs = [
        './multisource_results',
        './vfm_quick_experiments', 
        '.',
        'D:/glaucoma/vfm_quick_experiments'
    ]
    
    for common_dir in common_dirs:
        if os.path.exists(common_dir) and common_dir not in search_dirs:
            search_dirs.append(common_dir)
    
    if not search_dirs:
        print("No valid search directories found.")
        print("Please specify a valid directory with --search_dir")
        return
    
    all_result_files = {
        'csv': [],
        'json': [],
        'pickle': [],
        'txt': [],
        'models': []
    }
    
    for search_dir in search_dirs:
        print(f"Searching in: {search_dir}")
        result_files = find_all_result_files(search_dir)
        
        for file_type in all_result_files:
            all_result_files[file_type].extend(result_files[file_type])
    
    # Remove duplicates
    for file_type in all_result_files:
        all_result_files[file_type] = list(set(all_result_files[file_type]))
        all_result_files[file_type].sort()
    
    print_summary(all_result_files)


if __name__ == "__main__":
    main()
