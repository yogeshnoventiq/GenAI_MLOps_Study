#!/usr/bin/env python3
"""
Enhanced SDD Generator - Main Script
Generate comprehensive SDD from detailed SOW PDF using local Ollama AI
"""

import os
import sys
import argparse
import json
from enhanced_sow_parser import EnhancedSOWParser
from custom_sdd_generator import CustomSDDGenerator

def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive SDD from detailed SOW PDF using local AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_main.py --sow comprehensive_sow.pdf --customer "ACME Corp"
  python enhanced_main.py --sow detailed_sow.pdf --template my_sdd_template.md
  python enhanced_main.py --sow sow.pdf --output custom_sdd.md --save-data
        """
    )
    
    parser.add_argument('--sow', required=True, help='Path to comprehensive SOW PDF file')
    parser.add_argument('--customer', help='Override customer name')
    parser.add_argument('--template', help='Path to custom SDD template file')
    parser.add_argument('--output', help='Output SDD file path')
    parser.add_argument('--save-data', action='store_true', help='Save extracted data as JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.sow):
        print(f"❌ Error: SOW file not found: {args.sow}")
        sys.exit(1)
    
    if not args.sow.lower().endswith('.pdf'):
        print(f"❌ Error: Only PDF files are supported for comprehensive analysis")
        sys.exit(1)
    
    print("🚀 Starting Comprehensive SDD Generation")
    print("=" * 60)
    
    # Initialize components
    sow_parser = EnhancedSOWParser()
    sdd_generator = CustomSDDGenerator(args.template)
    
    # Step 1: Parse comprehensive SOW PDF
    print("📄 Step 1: Parsing comprehensive SOW PDF...")
    print("   📋 Extracting text content...")
    print("   🖼️  Analyzing images and diagrams...")
    
    try:
        comprehensive_data = sow_parser.parse_comprehensive_sow(args.sow)
        
        # Override customer if provided
        if args.customer:
            comprehensive_data['project_overview']['customer_name'] = args.customer
        
        print(f"   ✅ SOW analysis complete")
        
        if args.verbose:
            print(f"   📊 Extracted data summary:")
            print(f"      Customer: {comprehensive_data['project_overview']['customer_name']}")
            print(f"      Project: {comprehensive_data['project_overview']['project_name']}")
            print(f"      Type: {comprehensive_data['project_overview']['project_type']}")
            print(f"      Duration: {comprehensive_data['project_overview']['project_duration']}")
            print(f"      AWS Services: {len(comprehensive_data['technical_requirements']['aws_services'])}")
            print(f"      Business Objectives: {len(comprehensive_data['business_requirements']['business_objectives'])}")
        
    except Exception as e:
        print(f"   ❌ Error parsing SOW: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Step 2: Save extracted data if requested
    if args.save_data:
        print("💾 Step 2: Saving extracted data...")
        try:
            project_name = comprehensive_data['project_overview']['project_name'].replace(' ', '_')
            data_path = f"output/{project_name}_extracted_data.json"
            
            os.makedirs(os.path.dirname(data_path) if os.path.dirname(data_path) else '.', exist_ok=True)
            
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ Extracted data saved: {data_path}")
            
        except Exception as e:
            print(f"   ⚠️  Warning: Could not save extracted data: {e}")
    
    # Step 3: Generate comprehensive SDD
    print("📝 Step 3: Generating comprehensive SDD document...")
    try:
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            project_name = comprehensive_data['project_overview']['project_name'].replace(' ', '_')
            customer_name = comprehensive_data['project_overview']['customer_name'].replace(' ', '_')
            output_path = f"output/{customer_name}_{project_name}_SDD.md"
        
        # Generate SDD
        sdd_generator.generate_comprehensive_sdd(comprehensive_data, output_path)
        
        print(f"   ✅ SDD generation complete")
        
    except Exception as e:
        print(f"   ❌ Error generating SDD: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Summary and next steps
    print("\n" + "=" * 60)
    print("🎉 Comprehensive SDD Generation Complete!")
    print(f"📁 Output file: {output_path}")
    print(f"📊 Customer: {comprehensive_data['project_overview']['customer_name']}")
    print(f"🏗️ Project: {comprehensive_data['project_overview']['project_name']}")
    print(f"⏱️ Duration: {comprehensive_data['project_overview']['project_duration']}")
    print(f"💰 Budget: {comprehensive_data['project_overview']['total_budget']}")
    
    if args.save_data:
        print(f"💾 Extracted data: {data_path}")
    
    print("\n📋 Generated SDD includes:")
    print("   • Executive Summary with business context")
    print("   • Detailed technical requirements and architecture")
    print("   • Comprehensive implementation approach")
    print("   • Risk assessment and mitigation strategies")
    print("   • Cost analysis and resource requirements")
    print("   • Operational model and support procedures")
    print("   • Compliance and governance framework")
    
    print("\n💡 Next steps:")
    print("   1. Review the generated SDD document")
    print("   2. Customize sections with company-specific details")
    print("   3. Add actual cost estimates and timelines")
    print("   4. Include architecture diagrams from SOW")
    print("   5. Validate technical specifications")
    print("   6. Share with stakeholders for review and approval")
    
    print(f"\n📖 Open the SDD: {output_path}")

def validate_environment():
    """Validate that required dependencies are available"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("❌ Error: PyMuPDF not installed. Install with: pip install PyMuPDF")
        return False
    
    try:
        import requests
    except ImportError:
        print("❌ Error: requests not installed. Install with: pip install requests")
        return False
    
    try:
        import jinja2
    except ImportError:
        print("❌ Error: jinja2 not installed. Install with: pip install jinja2")
        return False
    
    return True

if __name__ == "__main__":
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Check if Ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("⚠️  Warning: Ollama server not responding. Make sure it's running:")
            print("   ollama serve")
    except Exception:
        print("⚠️  Warning: Cannot connect to Ollama. Make sure it's running:")
        print("   ollama serve")
    
    main()