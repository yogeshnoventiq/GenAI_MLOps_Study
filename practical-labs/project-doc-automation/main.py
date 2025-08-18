#!/usr/bin/env python3
"""
SDD Generator - Main Script
Generate SDD from SOW using local Ollama AI
"""

import os
import sys
import argparse
from local_sow_parser import LocalSOWParser
from sdd_generator import SDDGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate SDD from SOW using local AI')
    parser.add_argument('--sow', required=True, help='Path to SOW file (PDF, DOCX, or TXT)')
    parser.add_argument('--customer', help='Override customer name')
    parser.add_argument('--template', help='Path to custom SDD template')
    parser.add_argument('--output', help='Output SDD file path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.sow):
        print(f"❌ Error: SOW file not found: {args.sow}")
        sys.exit(1)
    
    print("🚀 Starting SDD Generation")
    print("=" * 40)
    
    # Initialize components
    sow_parser = LocalSOWParser()
    sdd_generator = SDDGenerator(args.template)
    
    # Parse SOW
    print("📄 Parsing SOW document...")
    try:
        sow_text = sow_parser.parse_sow_file(args.sow)
        print(f"   ✅ SOW loaded ({len(sow_text)} characters)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        sys.exit(1)
    
    # Analyze with AI
    print("🤖 Analyzing with local AI...")
    try:
        requirements = sow_parser.analyze_sow(sow_text)
        
        if args.customer:
            requirements['customer_name'] = args.customer
        
        print(f"   ✅ Customer: {requirements['customer_name']}")
        print(f"   🏗️ Project: {requirements['project_name']}")
        print(f"   ⚙️ Services: {', '.join(requirements['aws_services'])}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        sys.exit(1)
    
    # Generate SDD
    print("📝 Generating SDD...")
    try:
        if args.output:
            output_path = args.output
        else:
            project_name = requirements['project_name'].replace(' ', '_')
            output_path = f"output/{project_name}_SDD.md"
        
        sdd_generator.generate(requirements, output_path)
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 40)
    print("🎉 SDD Generation Complete!")
    print(f"📁 Output: {output_path}")

if __name__ == "__main__":
    main()