#!/usr/bin/env python3
"""Verify deployment files are present"""
import os
from pathlib import Path

def verify_deployment():
    required_paths = [
        "web_app/db_utils.py",
        "web_app/schemas.py", 
        "web_app/static",
        "web_app/templates",
        "config/schema_maps",
        "processed_data/final_linked_data"
    ]
    
    missing = []
    for path in required_paths:
        if not Path(path).exists():
            missing.append(path)
    
    if missing:
        print("Missing required files:")
        for path in missing:
            print(f"   - {path}")
        return False
    
    print("All required files present")
    return True

if __name__ == "__main__":
    verify_deployment()
