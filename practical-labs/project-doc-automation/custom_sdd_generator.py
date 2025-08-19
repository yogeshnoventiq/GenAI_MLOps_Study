import os
from datetime import datetime
from jinja2 import Template

class CustomSDDGenerator:
    def __init__(self, template_path: str = None):
        self.template = self._load_template(template_path)
    
    def _load_template(self, template_path: str) -> str:
        """Load custom SDD template"""
        if template_path and os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        # Comprehensive SDD template - customize this with your company format
        return """# Solution Design Document
## {{ project_overview.project_name }}

---

**Client:** {{ project_overview.customer_name }}  
**Document Version:** 1.0  
**Date:** {{ current_date }}  
**Project Duration:** {{ project_overview.project_duration }}  
**Total Budget:** {{ project_overview.total_budget }}  
**Prepared by:** [Your Company Name]  
**Contact:** [your-email@company.com]

---

## 1. Executive Summary

### 1.1 Project Overview
{{ project_overview.project_scope }}

This Solution Design Document outlines the comprehensive AWS cloud solution for {{ project_overview.customer_name }}'s {{ project_overview.project_name }}. The project involves {{ project_overview.project_type }} of existing infrastructure to leverage AWS cloud services for enhanced scalability, security, and operational efficiency.

### 1.2 Key Benefits
{% for objective in business_requirements.business_objectives %}
- {{ objective }}
{% endfor %}

### 1.3 Success Criteria
{% for criteria in business_requirements.success_criteria %}
- {{ criteria }}
{% endfor %}

---

## 2. Business Requirements

### 2.1 Business Objectives
{% for objective in business_requirements.business_objectives %}
- **{{ objective }}**: Strategic business goal driving the project
{% endfor %}

### 2.2 Business Drivers
{% for driver in business_requirements.business_drivers %}
- {{ driver }}
{% endfor %}

### 2.3 Stakeholders
{% for stakeholder in business_requirements.stakeholders %}
- {{ stakeholder }}
{% endfor %}

---

## 3. Current State Assessment

### 3.1 Existing Infrastructure
{% for item in technical_requirements.current_infrastructure %}
- {{ item }}
{% endfor %}

### 3.2 Current Challenges
- Legacy system limitations
- Scalability constraints
- Maintenance overhead
- Security vulnerabilities

---

## 4. Target Architecture

### 4.1 Proposed AWS Architecture
{% for item in technical_requirements.target_architecture %}
- {{ item }}
{% endfor %}

### 4.2 AWS Services Utilization

{% for service in technical_requirements.aws_services %}
#### {{ service }}
- **Purpose**: Core infrastructure component for {{ project_overview.project_name }}
- **Configuration**: To be detailed during implementation phase
- **Integration**: Seamless integration with other AWS services
{% endfor %}

### 4.3 Architecture Principles
- **Security First**: Multi-layered security approach with defense in depth
- **High Availability**: {{ technical_requirements.availability_requirements | join(', ') }}
- **Scalability**: {{ technical_requirements.scalability_requirements | join(', ') }}
- **Performance**: {{ technical_requirements.performance_requirements | join(', ') }}

---

## 5. Detailed Technical Design

### 5.1 Network Architecture
{% for item in architecture_details.network_architecture %}
- {{ item }}
{% endfor %}

### 5.2 Security Architecture
{% for item in architecture_details.security_architecture %}
- {{ item }}
{% endfor %}

### 5.3 Data Architecture
{% for item in architecture_details.data_architecture %}
- {{ item }}
{% endfor %}

### 5.4 Application Architecture
{% for item in architecture_details.application_architecture %}
- {{ item }}
{% endfor %}

---

## 6. Security and Compliance

### 6.1 Security Requirements
{% for req in technical_requirements.security_requirements %}
- {{ req }}
{% endfor %}

### 6.2 Compliance Standards
{% for standard in compliance_and_governance.compliance_standards %}
- **{{ standard }}**: Implementation approach and controls
{% endfor %}

### 6.3 Data Governance
{% for item in compliance_and_governance.data_governance %}
- {{ item }}
{% endfor %}

### 6.4 Privacy Requirements
{% for req in compliance_and_governance.privacy_requirements %}
- {{ req }}
{% endfor %}

---

## 7. Implementation Approach

### 7.1 Migration Strategy
{% for strategy in implementation_approach.migration_strategy %}
- {{ strategy }}
{% endfor %}

### 7.2 Implementation Phases
{% for phase in implementation_approach.phases %}
- **{{ phase }}**: Detailed planning and execution
{% endfor %}

### 7.3 Dependencies
{% for dependency in implementation_approach.dependencies %}
- {{ dependency }}
{% endfor %}

### 7.4 Testing Strategy
{% for test in implementation_approach.testing_strategy %}
- {{ test }}
{% endfor %}

---

## 8. Risk Assessment and Mitigation

### 8.1 Identified Risks
{% for risk in implementation_approach.risks_and_mitigation %}
- **Risk**: {{ risk }}
- **Mitigation**: Comprehensive planning and contingency procedures
{% endfor %}

### 8.2 Risk Mitigation Strategies
- **Technical Risks**: Thorough testing and validation procedures
- **Timeline Risks**: Phased implementation with buffer time
- **Resource Risks**: Dedicated team allocation and backup resources
- **Business Risks**: Regular stakeholder communication and approval gates

---

## 9. Operational Model

### 9.1 Monitoring and Alerting
{% for req in operational_requirements.monitoring_requirements %}
- {{ req }}
{% endfor %}

### 9.2 Maintenance and Support
{% for req in operational_requirements.maintenance_requirements %}
- {{ req }}
{% endfor %}

### 9.3 Support Model
{% for item in operational_requirements.support_model %}
- {{ item }}
{% endfor %}

### 9.4 Training Requirements
{% for training in operational_requirements.training_requirements %}
- {{ training }}
{% endfor %}

---

## 10. Disaster Recovery and Business Continuity

### 10.1 Disaster Recovery Strategy
{% for item in architecture_details.disaster_recovery %}
- {{ item }}
{% endfor %}

### 10.2 Backup and Recovery Procedures
- **Recovery Time Objective (RTO)**: 4 hours
- **Recovery Point Objective (RPO)**: 1 hour
- **Backup Frequency**: Daily automated backups
- **Cross-Region Replication**: Enabled for critical data

---

## 11. Cost Analysis

### 11.1 Cost Breakdown
{% for cost in cost_and_resources.cost_breakdown %}
- {{ cost }}
{% endfor %}

### 11.2 Monthly Operational Costs (Estimated)
- **Compute Services**: $X,XXX/month
- **Storage Services**: $XXX/month
- **Network Services**: $XXX/month
- **Security Services**: $XXX/month
- **Management Services**: $XXX/month
- **Total Monthly**: $X,XXX/month

### 11.3 One-time Implementation Costs
- **Migration Services**: $XX,XXX
- **Professional Services**: $XX,XXX
- **Training and Certification**: $X,XXX
- **Total Implementation**: $XX,XXX

### 11.4 Cost Optimization Strategies
- Reserved Instance utilization for predictable workloads
- Spot Instance usage for non-critical batch processing
- Automated scaling to optimize resource utilization
- Regular cost review and optimization

---

## 12. Project Timeline and Milestones

### 12.1 Project Phases and Timeline
**Total Duration**: {{ project_overview.project_duration }}

{% for milestone in cost_and_resources.timeline_milestones %}
- **{{ milestone }}**: Key project deliverable
{% endfor %}

### 12.2 Critical Path Activities
- Infrastructure design and approval
- Security assessment and compliance validation
- Data migration and application deployment
- Testing and performance validation
- Go-live and production support

---

## 13. Resource Requirements

### 13.1 Project Team Structure
{% for resource in cost_and_resources.resource_requirements %}
- {{ resource }}
{% endfor %}

### 13.2 Skills and Expertise Required
- AWS Certified Solutions Architect
- DevOps and automation expertise
- Security and compliance specialists
- Application migration specialists
- Project management and coordination

---

## 14. Integration Requirements

### 14.1 System Integrations
{% for integration in technical_requirements.integration_requirements %}
- {{ integration }}
{% endfor %}

### 14.2 Data Integration
- Real-time data synchronization
- Batch data processing
- API-based integrations
- Legacy system connectivity

---

## 15. Quality Assurance and Testing

### 15.1 Testing Approach
- **Unit Testing**: Individual component validation
- **Integration Testing**: End-to-end system testing
- **Performance Testing**: Load and stress testing
- **Security Testing**: Vulnerability assessment and penetration testing
- **User Acceptance Testing**: Business validation and sign-off

### 15.2 Quality Gates
- Architecture review and approval
- Security assessment completion
- Performance benchmark achievement
- Compliance validation
- Stakeholder acceptance

---

## 16. Go-Live and Transition

### 16.1 Go-Live Strategy
- Phased rollout approach
- Pilot user group validation
- Production deployment
- Post-go-live monitoring and support

### 16.2 Transition to Operations
- Knowledge transfer to operations team
- Documentation handover
- Support procedures activation
- Continuous improvement process

---

## 17. Success Metrics and KPIs

### 17.1 Technical KPIs
- System availability: 99.9%
- Response time improvement: 50%
- Infrastructure cost reduction: 30%
- Security incident reduction: 90%

### 17.2 Business KPIs
- Time to market improvement
- Operational efficiency gains
- Customer satisfaction scores
- Revenue impact metrics

---

## 18. Assumptions and Constraints

### 18.1 Project Assumptions
{% for assumption in cost_and_resources.assumptions %}
- {{ assumption }}
{% endfor %}

### 18.2 Constraints
- Budget limitations and approval processes
- Timeline constraints and business deadlines
- Resource availability and skill requirements
- Regulatory and compliance requirements

---

## 19. Next Steps and Recommendations

### 19.1 Immediate Actions
1. **Stakeholder Approval**: Obtain formal approval for this design document
2. **Project Kickoff**: Schedule project initiation meeting
3. **Team Assembly**: Confirm project team members and roles
4. **Detailed Planning**: Develop comprehensive project plan

### 19.2 Recommendations
- Conduct proof of concept for critical components
- Establish regular stakeholder communication cadence
- Implement robust change management processes
- Plan for comprehensive team training and knowledge transfer

---

## 20. Appendices

### Appendix A: Technical Specifications
*Detailed technical specifications will be provided during implementation phase*

### Appendix B: Architecture Diagrams
*Comprehensive architecture diagrams including:*
- High-level solution architecture
- Network topology and security zones
- Data flow and integration patterns
- Disaster recovery architecture

### Appendix C: Compliance Mapping
*Detailed mapping of compliance requirements to technical controls*

### Appendix D: Cost Models and Calculations
*Detailed cost breakdown and ROI analysis*

---

**Document Control Information:**
- **Version**: 1.0
- **Created**: {{ current_date }}
- **Last Modified**: {{ current_date }}
- **Next Review Date**: {{ next_review_date }}
- **Document Owner**: [Your Company Name]
- **Approval Status**: Draft

---

**Contact Information:**
**[Your Company Name]**  
Address: [Your Company Address]  
Phone: [Your Phone Number]  
Email: [Your Email Address]  
Website: [Your Website]

---

*This document contains confidential and proprietary information. Distribution is restricted to authorized personnel only.*
"""
    
    def generate_comprehensive_sdd(self, comprehensive_data: dict, output_path: str):
        """Generate comprehensive SDD from extracted data"""
        
        # Prepare template variables
        template_vars = {
            'current_date': datetime.now().strftime('%Y-%m-%d'),
            'next_review_date': datetime.now().strftime('%Y-%m-%d'),
            'project_overview': comprehensive_data.get('project_overview', {}),
            'business_requirements': comprehensive_data.get('business_requirements', {}),
            'technical_requirements': comprehensive_data.get('technical_requirements', {}),
            'compliance_and_governance': comprehensive_data.get('compliance_and_governance', {}),
            'architecture_details': comprehensive_data.get('architecture_details', {}),
            'implementation_approach': comprehensive_data.get('implementation_approach', {}),
            'operational_requirements': comprehensive_data.get('operational_requirements', {}),
            'cost_and_resources': comprehensive_data.get('cost_and_resources', {})
        }
        
        # Render template
        template = Template(self.template)
        sdd_content = template.render(**template_vars)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Save document
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(sdd_content)
        
        print(f"✅ Comprehensive SDD generated: {output_path}")
        return output_path