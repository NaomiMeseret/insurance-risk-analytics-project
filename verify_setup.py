#!/usr/bin/env python3
"""
Setup verification script
Checks if all required components are properly configured
"""

import sys
from pathlib import Path
import subprocess

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version {version.major}.{version.minor}.{version.micro} is too old. Need 3.8+")
        return False

def check_git():
    """Check if Git is initialized"""
    try:
        result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git repository initialized")
            return True
        else:
            print("⚠️  Git repository not initialized. Run: git init")
            return False
    except FileNotFoundError:
        print("❌ Git not found. Please install Git")
        return False

def check_dvc():
    """Check if DVC is installed and initialized"""
    try:
        result = subprocess.run(['dvc', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ DVC installed: {version}")
            
            # Check if DVC is initialized
            dvc_dir = Path('.dvc')
            if dvc_dir.exists():
                print("✅ DVC initialized")
                return True
            else:
                print("⚠️  DVC not initialized. Run: dvc init")
                return False
        else:
            print("❌ DVC not installed. Run: pip install dvc")
            return False
    except FileNotFoundError:
        print("❌ DVC not installed. Run: pip install dvc")
        return False

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'scipy', 'scikit-learn', 'pytest'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    return True

def check_project_structure():
    """Check if project structure is correct"""
    required_dirs = [
        'src', 'src/data', 'src/eda', 'src/models', 
        'src/utils', 'src/hypothesis_testing',
        'tests', 'data', 'notebooks', 'reports'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}/ exists")
        else:
            print(f"❌ {dir_path}/ missing")
            all_exist = False
    
    return all_exist

def check_key_files():
    """Check if key files exist"""
    key_files = [
        'README.md', 'requirements.txt', '.gitignore', 
        'setup_dvc.sh', '.github/workflows/ci.yml'
    ]
    
    all_exist = True
    for file_path in key_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all checks"""
    print("="*60)
    print("PROJECT SETUP VERIFICATION")
    print("="*60)
    print()
    
    checks = {
        'Python Version': check_python_version,
        'Git': check_git,
        'DVC': check_dvc,
        'Dependencies': check_dependencies,
        'Project Structure': check_project_structure,
        'Key Files': check_key_files
    }
    
    results = {}
    for name, check_func in checks.items():
        print(f"\n{name}:")
        print("-" * 60)
        results[name] = check_func()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Project is ready to use.")
    else:
        print("\n⚠️  Some checks failed. Please review the output above.")
        print("See SETUP_GUIDE.md for detailed setup instructions.")

if __name__ == "__main__":
    main()

