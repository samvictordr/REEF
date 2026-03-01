#!/usr/bin/env python3
"""
Environment Verification Script for Coral VFL Multi-Omics Project.

Run this script to verify all required dependencies are installed
and the environment is correctly configured.

Usage:
    python scripts/verify_environment.py

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
"""

import sys
import importlib
from typing import List, Tuple

# =============================================================================
# Required packages with minimum versions
# =============================================================================

REQUIRED_PACKAGES = [
    ("numpy", "1.24.0"),
    ("scipy", "1.10.0"),
    ("pandas", "2.0.0"),
    ("sklearn", "1.2.0"),  # scikit-learn
    ("torch", "2.0.0"),
    ("matplotlib", "3.7.0"),
    ("seaborn", "0.12.0"),
    ("yaml", None),  # PyYAML - no version check
    ("statsmodels", "0.14.0"),
]

OPTIONAL_PACKAGES = [
    ("shap", "0.42.0"),
    ("nvflare", "2.4.0"),
    ("mlflow", "2.3.0"),
    ("tensorboard", None),
    ("plotly", "5.14.0"),
]


def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse version string to tuple for comparison."""
    try:
        parts = version_str.split(".")
        return tuple(int(p.split("+")[0].split("a")[0].split("b")[0].split("rc")[0]) 
                     for p in parts[:3])
    except (ValueError, AttributeError):
        return (0, 0, 0)


def check_package(package_name: str, min_version: str = None) -> Tuple[bool, str]:
    """
    Check if a package is installed and meets minimum version.
    
    Returns:
        (success, message)
    """
    try:
        module = importlib.import_module(package_name)
        
        # Get version
        version = getattr(module, "__version__", None)
        if version is None:
            version = getattr(module, "VERSION", "unknown")
        
        # Check version if required
        if min_version and version != "unknown":
            current = parse_version(str(version))
            required = parse_version(min_version)
            
            if current < required:
                return False, f"v{version} (requires >= {min_version})"
        
        return True, f"v{version}"
        
    except ImportError:
        return False, "NOT INSTALLED"


def check_cuda() -> Tuple[bool, str]:
    """Check CUDA availability for PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            return True, f"CUDA available ({device_count} device(s): {device_name})"
        else:
            return True, "CPU only (CUDA not available - this is OK for testing)"
    except Exception as e:
        return False, f"Error checking CUDA: {e}"


def check_data_files() -> Tuple[bool, str]:
    """Check if processed data files exist."""
    from pathlib import Path
    
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    required_files = [
        "transcriptomics.tsv",
        "proteomics.tsv", 
        "metabolomics.tsv",
        "microbiome.tsv",
    ]
    
    missing = []
    for fname in required_files:
        fpath = data_dir / fname
        if not fpath.exists() or fpath.stat().st_size == 0:
            missing.append(fname)
    
    if missing:
        return False, f"Missing/empty: {', '.join(missing)}"
    else:
        return True, "All data files present"


def main():
    """Run all environment checks."""
    print("=" * 70)
    print("CORAL VFL MULTI-OMICS - ENVIRONMENT VERIFICATION")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # Check Python version
    print("Python Version:")
    py_version = sys.version_info
    if py_version >= (3, 10):
        print(f"  ✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print(f"  ✗ Python {py_version.major}.{py_version.minor} (requires >= 3.10)")
        all_passed = False
    print()
    
    # Check required packages
    print("Required Packages:")
    for package, min_version in REQUIRED_PACKAGES:
        success, msg = check_package(package, min_version)
        status = "✓" if success else "✗"
        print(f"  {status} {package}: {msg}")
        if not success:
            all_passed = False
    print()
    
    # Check optional packages
    print("Optional Packages:")
    for package, min_version in OPTIONAL_PACKAGES:
        success, msg = check_package(package, min_version)
        status = "✓" if success else "○"
        print(f"  {status} {package}: {msg}")
    print()
    
    # Check CUDA
    print("GPU/CUDA:")
    success, msg = check_cuda()
    status = "✓" if success else "○"
    print(f"  {status} {msg}")
    print()
    
    # Check data files
    print("Data Files:")
    success, msg = check_data_files()
    status = "✓" if success else "✗"
    print(f"  {status} {msg}")
    if not success:
        all_passed = False
    print()
    
    # Summary
    print("=" * 70)
    if all_passed:
        print("✓ All environment checks PASSED")
        print("  You can now run: python benchmarks/run_comprehensive_benchmark.py")
        return 0
    else:
        print("✗ Some environment checks FAILED")
        print("  Please install missing dependencies:")
        print("    conda env create -f environment.yml")
        print("    conda activate coral-vfl")
        return 1


if __name__ == "__main__":
    sys.exit(main())
