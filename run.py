"""Run the pipeline without installing the package: python run.py <step> [REGION]."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from inail_severity.cli import main

if __name__ == "__main__":
    main()
