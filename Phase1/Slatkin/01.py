# How to find your python version:
import subprocess

subprocess.run(["python3", "--version"])

# or:
import sys
print(sys.version_info)
print(sys.version)
