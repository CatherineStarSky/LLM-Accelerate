#!/usr/bin/env python
"""Verify project configuration and dependencies"""

import sys
import os

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  Warning: Python 3.8+ recommended")
        return False
    return True

def check_dependencies():
    """Check dependency packages"""
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'yaml': 'PyYAML'
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            all_ok = False
    
    return all_ok

def check_data():
    """Check the dataset"""
    data_dir = "ml-100k"
    required_files = ['u.item', 'u.data', 'u.user']
    
    if not os.path.exists(data_dir):
        print(f"✗ Data directory not found: {data_dir}")
        return False
    
    all_ok = True
    for file in required_files:
        path = os.path.join(data_dir, file)
        if os.path.exists(path):
            print(f"✓ {file} found")
        else:
            print(f"✗ {file} NOT found")
            all_ok = False
    
    return all_ok

def check_simd_lib():
    """Check SIMD library"""
    lib_path = "build/simd_ops.dylib"
    
    if os.path.exists(lib_path):
        size_mb = os.path.getsize(lib_path) / (1024 * 1024)
        print(f"✓ SIMD library found ({size_mb:.2f} MB)")
        return True
    else:
        print(f"✗ SIMD library NOT found at {lib_path}")
        print("  Run: mkdir -p build && cd build && cmake .. && cmake --build .")
        return False

def check_structure():
    """Check directory structure"""
    required_dirs = ['src', 'benchmark', 'scripts', 'results']
    required_files = [
        'src/main_rerank.py',
        'benchmark/data_loader.py',
        'benchmark/benchmark_runner.py',
        'benchmark/configs.yaml',
        'run_demo.sh'
    ]
    
    all_ok = True
    
    print("\nDirectory structure:")
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ NOT found")
            all_ok = False
    
    print("\nKey files:")
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} NOT found")
            all_ok = False
    
    return all_ok

def main():
    print("="*60)
    print("LLM Re-ranking Setup Verification")
    print("="*60)
    print()
    
    checks = [
        ("Python version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Data files", check_data),
        ("SIMD library", check_simd_lib),
        ("Project structure", check_structure)
    ]
    
    results = {}
    for name, check_fn in checks:
        print(f"\n--- {name} ---")
        results[name] = check_fn()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_pass = all(results.values())
    
    print()
    if all_pass:
        print("✓ All checks passed! You can run:")
        print("  ./run_demo.sh")
        print("  或")
        print("  python src/main_rerank.py --mode single --n-candidates 10")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nQuick fixes:")
        if not results["Dependencies"]:
            print("  pip install -r requirements.txt")
        if not results["SIMD library"]:
            print("  mkdir -p build && cd build && cmake .. && cmake --build .")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())

