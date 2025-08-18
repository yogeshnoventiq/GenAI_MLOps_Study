# 🚀 SDD Generator with Ollama
*Generate Solution Design Documents from SOW using local AI*

## 🎯 **What This Does**
- Reads your SOW document (PDF, DOCX, TXT)
- Extracts project details using local Ollama AI
- Generates professional SDD document
- **100% Private** - No data sent to internet

## ⚡ **Quick Start**

### **1. Install Dependencies**
```bash
pip install requests jinja2 PyPDF2 python-docx
```

### **2. Start Ollama**
```bash
ollama serve
ollama pull llama2:7b  # if not already done
```

### **3. Create Sample SOW**
```bash
cat > sample_sow.txt << 'EOF'
Client: ACME Corporation
Project: AWS Cloud Migration
Services: EC2, S3, RDS, Lambda
Timeline: 12 weeks
Compliance: SOC2, HIPAA
Business Goals: Cost reduction, improved performance
EOF
```

### **4. Generate SDD**
```bash
python main.py --sow sample_sow.txt --customer "ACME Corp"
```

## 📁 **Files**
- `main.py` - Main script
- `local_sow_parser.py` - SOW parser with Ollama
- `sdd_generator.py` - SDD document generator

## 🎯 **Usage**
```bash
# Basic usage
python main.py --sow your_sow.pdf

# With custom customer name
python main.py --sow sow.docx --customer "Client Name"

# With custom template
python main.py --sow sow.txt --template my_template.md

# Custom output path
python main.py --sow sow.pdf --output custom_sdd.md
```

## 🔒 **Privacy**
- ✅ Uses local Ollama AI only
- ✅ No internet connection required
- ✅ Customer data stays private
- ✅ Compliance friendly

## 📊 **Output**
Generated SDD includes:
- Executive Summary
- Business Objectives  
- Technical Requirements
- AWS Services
- Implementation Plan
- Cost Estimation
- Risk Assessment

Ready to use! 🚀