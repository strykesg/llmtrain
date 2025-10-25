#!/usr/bin/env python3
"""
Syntax check script for Python files.
"""

import ast
import sys

def check_syntax(filename):
    """Check Python file syntax."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source = f.read()

        # Parse the AST
        ast.parse(source, filename=filename)
        print(f"✅ {filename}: Syntax is valid")
        return True

    except SyntaxError as e:
        print(f"❌ {filename}: Syntax error at line {e.lineno}: {e.msg}")
        print(f"   {e.text}")
        return False
    except Exception as e:
        print(f"❌ {filename}: Error checking syntax: {e}")
        return False

def main():
    """Check syntax of training script."""
    files_to_check = [
        "train_qwen3.py",
        "test_env.py",
        "test_wandb.py",
        "test_formatting.py",
        "test_training_args.py"
    ]

    print("🔍 Checking Python syntax...")
    all_valid = True

    for filename in files_to_check:
        if not check_syntax(filename):
            all_valid = False

    if all_valid:
        print("🎉 All Python files have valid syntax!")
    else:
        print("❌ Some files have syntax errors")
        sys.exit(1)

if __name__ == "__main__":
    main()
