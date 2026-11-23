"""Rebuild local arithmetic/search kernels without reassociation or FMA."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
subprocess.run(['clang++', '-O3', '-std=c++17', '-ffp-contract=off', '-fno-fast-math',
                '-dynamiclib', str(root/'realapps_kernels.cpp'),
                '-o', str(root/'librealapps_kernels.dylib')], check=True)
