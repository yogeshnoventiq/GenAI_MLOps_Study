import os
from datetime import datetime
from jinja2 import Template

class SDDGenerator:
    def __init__(self, template_path: str = None):
        self.template = self._load_template(template_path)
    
    def _load_template(self, template_path: str) -> str:
        """Load SDD template"""
        if template_path and os.path.exists(template_path):
            with open(template_path, 'r') as f:
                return f.read()
        
        return """# Solution Design Document
## {{ project_name }}

**Client:** {{ customer_name }}  
**Date:** {{ current_date }}  
**Prepared by:** Your Company Name

## Executive Summary
This document outlines the AWS solution design for {{ customer_name }}'s {{ project_name }} project.

## Business Objectives
{% for objective in business_objectives %}
- {{ objective }}
{% endfor %}

## Technical Requirements
{% for requirement in technical_requirements %}
- {{ requirement }}
{% endfor %}

## AWS Services
{% for service in aws_services %}
- **{{ service }}**: Core infrastructure component
{% endfor %}

## Implementation Timeline
{{ timeline }}

## Compliance Requirements
{% for req in compliance_requirements %}
- {{ req }}
{% endfor %}

## Architecture Overview
The proposed solution follows AWS Well-Architected Framework principles:
- Security: Multi-layered security approach
- Reliability: High availability design
- Performance: Optimized for workload requirements
- Cost Optimization: Right-sized resources

## Implementation Plan
1. Assessment and Planning
2. Infrastructure Setup
3. Migration/Deployment
4. Testing and Validation
5. Go-live and Support

## Cost Estimation
- Monthly operational costs: To be determined
- One-time setup costs: To be determined

## Risk Assessment
- Technical risks and mitigation strategies
- Business continuity planning
- Change management approach

---
*Document generated on {{ current_date }}*"""
    
    def generate(self, requirements: dict, output_path: str):
        """Generate SDD document"""
        template_vars = {
            'project_name': requirements.get('project_name', 'AWS Project'),
            'customer_name': requirements.get('customer_name', 'Customer'),
            'current_date': datetime.now().strftime('%Y-%m-%d'),
            'timeline': requirements.get('timeline', 'Not specified'),
            'aws_services': requirements.get('aws_services', []),
            'business_objectives': requirements.get('business_objectives', []),
            'technical_requirements': requirements.get('technical_requirements', []),
            'compliance_requirements': requirements.get('compliance_requirements', [])
        }
        
        template = Template(self.template)
        content = template.render(**template_vars)
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ SDD generated: {output_path}")
        return output_path