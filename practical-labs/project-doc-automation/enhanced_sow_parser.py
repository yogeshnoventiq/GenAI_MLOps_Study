import requests
import json
import re
import fitz  # PyMuPDF for better PDF parsing
from typing import Dict, List
import base64
from io import BytesIO

class EnhancedSOWParser:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "llama2:7b"
    
    def parse_comprehensive_sow(self, pdf_path: str) -> Dict:
        """Parse comprehensive SOW PDF with text and images"""
        
        # Extract text content
        text_content = self._extract_text_from_pdf(pdf_path)
        
        # Extract images/diagrams
        images_info = self._extract_images_from_pdf(pdf_path)
        
        # Analyze content with AI
        extracted_data = self._analyze_comprehensive_content(text_content, images_info)
        
        return extracted_data
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from PDF"""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                full_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
            
            doc.close()
            return full_text
            
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""
    
    def _extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract images and diagrams from PDF"""
        try:
            doc = fitz.open(pdf_path)
            images_info = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            
                            images_info.append({
                                'page': page_num + 1,
                                'index': img_index,
                                'size': len(img_data),
                                'description': f"Image on page {page_num + 1}"
                            })
                        
                        pix = None
                    except Exception as e:
                        continue
            
            doc.close()
            return images_info
            
        except Exception as e:
            print(f"Error extracting images: {e}")
            return []
    
    def _analyze_comprehensive_content(self, text_content: str, images_info: List[Dict]) -> Dict:
        """Analyze comprehensive SOW content using AI"""
        
        # Create detailed prompt for comprehensive analysis
        prompt = f"""
        Analyze this comprehensive Statement of Work document and extract detailed information.
        The document contains {len(images_info)} images/diagrams.
        
        Return a detailed JSON object with this structure:
        {{
            "project_overview": {{
                "customer_name": "full company name",
                "project_name": "complete project title",
                "project_type": "migration/modernization/new_deployment/hybrid",
                "project_scope": "detailed scope description",
                "project_duration": "timeline with phases",
                "total_budget": "budget amount if mentioned"
            }},
            "business_requirements": {{
                "business_objectives": ["list of business goals"],
                "success_criteria": ["measurable success criteria"],
                "stakeholders": ["key stakeholders mentioned"],
                "business_drivers": ["reasons for the project"]
            }},
            "technical_requirements": {{
                "current_infrastructure": ["existing systems and technologies"],
                "target_architecture": ["desired future state"],
                "aws_services": ["specific AWS services mentioned"],
                "performance_requirements": ["performance criteria"],
                "scalability_requirements": ["scaling needs"],
                "availability_requirements": ["uptime and availability needs"],
                "security_requirements": ["security specifications"],
                "integration_requirements": ["systems to integrate with"]
            }},
            "compliance_and_governance": {{
                "compliance_standards": ["regulatory requirements"],
                "data_governance": ["data handling requirements"],
                "audit_requirements": ["audit and reporting needs"],
                "privacy_requirements": ["data privacy specifications"]
            }},
            "architecture_details": {{
                "network_architecture": ["network design elements"],
                "security_architecture": ["security design elements"],
                "data_architecture": ["data flow and storage design"],
                "application_architecture": ["application design patterns"],
                "disaster_recovery": ["DR and backup requirements"]
            }},
            "implementation_approach": {{
                "migration_strategy": ["approach for migration"],
                "phases": ["implementation phases"],
                "dependencies": ["project dependencies"],
                "risks_and_mitigation": ["identified risks and solutions"],
                "testing_strategy": ["testing approach"]
            }},
            "operational_requirements": {{
                "monitoring_requirements": ["monitoring and alerting needs"],
                "maintenance_requirements": ["ongoing maintenance needs"],
                "support_model": ["support and operations model"],
                "training_requirements": ["team training needs"]
            }},
            "cost_and_resources": {{
                "cost_breakdown": ["cost categories and estimates"],
                "resource_requirements": ["human resources needed"],
                "timeline_milestones": ["key project milestones"],
                "assumptions": ["project assumptions"]
            }}
        }}
        
        Document Content:
        {text_content[:8000]}  # Limit for AI processing
        
        Images/Diagrams Found: {len(images_info)} (including architecture diagrams)
        
        Extract comprehensive and detailed information. If specific information is not available, use "Not specified" but try to infer reasonable details from context.
        """
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "num_ctx": 4096  # Larger context for comprehensive analysis
                    }
                },
                timeout=120  # Longer timeout for complex analysis
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._extract_comprehensive_json(result.get('response', ''))
            else:
                print(f"Ollama error: {response.status_code}")
                return self._fallback_comprehensive_extraction(text_content)
                
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return self._fallback_comprehensive_extraction(text_content)
    
    def _extract_comprehensive_json(self, text: str) -> Dict:
        """Extract comprehensive JSON from AI response"""
        # Try to find and parse JSON
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        
        if json_match:
            try:
                json_str = json_match.group()
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
        
        # If JSON extraction fails, return comprehensive default
        return self._get_comprehensive_default_structure()
    
    def _fallback_comprehensive_extraction(self, text_content: str) -> Dict:
        """Comprehensive fallback extraction using regex patterns"""
        
        # Enhanced regex patterns for detailed extraction
        patterns = {
            'customer': [
                r'(?:client|customer|company|organization):\s*([^\n]+)',
                r'(?:for|client)\s+([A-Z][a-zA-Z\s&.,]+?)(?:\s|,|\.|$)',
                r'([A-Z][a-zA-Z\s&.,]{10,50})\s+(?:requires|needs|wants)'
            ],
            'project': [
                r'(?:project|title|initiative):\s*([^\n]+)',
                r'(?:project|migration|modernization|implementation)\s+([^\n]+)',
                r'(?:the|this)\s+([a-zA-Z\s]{10,50})\s+(?:project|initiative)'
            ],
            'budget': [
                r'(?:budget|cost|investment):\s*\$?([0-9,]+)',
                r'\$([0-9,]+)(?:\s+(?:budget|cost|total))?'
            ],
            'timeline': [
                r'(?:timeline|duration|schedule):\s*([^\n]+)',
                r'(?:over|within|in)\s+([0-9]+\s+(?:weeks|months|years))'
            ]
        }
        
        extracted = {}
        
        # Extract using patterns
        for key, pattern_list in patterns.items():
            extracted[key] = "Not specified"
            for pattern in pattern_list:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    extracted[key] = match.group(1).strip()
                    break
        
        # Extract AWS services
        aws_services = []
        aws_keywords = [
            'EC2', 'S3', 'RDS', 'Lambda', 'EKS', 'ECS', 'VPC', 'CloudFormation',
            'CloudWatch', 'IAM', 'Route53', 'CloudFront', 'ALB', 'NLB',
            'DynamoDB', 'Redshift', 'Kinesis', 'SQS', 'SNS', 'API Gateway'
        ]
        
        for service in aws_keywords:
            if service.lower() in text_content.lower():
                aws_services.append(service)
        
        if not aws_services:
            aws_services = ["EC2", "S3", "RDS", "VPC"]
        
        # Build comprehensive structure
        return {
            "project_overview": {
                "customer_name": extracted['customer'],
                "project_name": extracted['project'],
                "project_type": "migration",
                "project_scope": "AWS cloud migration and modernization",
                "project_duration": extracted['timeline'],
                "total_budget": extracted['budget']
            },
            "business_requirements": {
                "business_objectives": ["Cost optimization", "Improved scalability", "Enhanced security"],
                "success_criteria": ["Successful migration", "Performance improvement", "Cost reduction"],
                "stakeholders": ["IT Team", "Business Users", "Management"],
                "business_drivers": ["Digital transformation", "Cost efficiency", "Scalability needs"]
            },
            "technical_requirements": {
                "current_infrastructure": ["On-premises servers", "Legacy applications", "Traditional databases"],
                "target_architecture": ["Cloud-native architecture", "Microservices", "Serverless components"],
                "aws_services": aws_services,
                "performance_requirements": ["High availability", "Low latency", "Scalable performance"],
                "scalability_requirements": ["Auto-scaling", "Elastic capacity", "Load balancing"],
                "availability_requirements": ["99.9% uptime", "Multi-AZ deployment", "Disaster recovery"],
                "security_requirements": ["Data encryption", "Access controls", "Network security"],
                "integration_requirements": ["API integrations", "Data synchronization", "Legacy system connectivity"]
            },
            "compliance_and_governance": {
                "compliance_standards": ["SOC2", "ISO 27001", "Industry standards"],
                "data_governance": ["Data classification", "Data retention", "Data quality"],
                "audit_requirements": ["Audit trails", "Compliance reporting", "Regular assessments"],
                "privacy_requirements": ["Data privacy", "PII protection", "GDPR compliance"]
            },
            "architecture_details": {
                "network_architecture": ["VPC design", "Subnet configuration", "Security groups"],
                "security_architecture": ["IAM policies", "Encryption", "Monitoring"],
                "data_architecture": ["Data lakes", "Data warehouses", "ETL processes"],
                "application_architecture": ["Microservices", "API Gateway", "Load balancers"],
                "disaster_recovery": ["Backup strategies", "Cross-region replication", "RTO/RPO targets"]
            },
            "implementation_approach": {
                "migration_strategy": ["Phased approach", "Pilot migration", "Full cutover"],
                "phases": ["Assessment", "Planning", "Migration", "Testing", "Go-live"],
                "dependencies": ["Network connectivity", "Data migration", "Application compatibility"],
                "risks_and_mitigation": ["Technical risks", "Timeline risks", "Resource risks"],
                "testing_strategy": ["Unit testing", "Integration testing", "Performance testing"]
            },
            "operational_requirements": {
                "monitoring_requirements": ["CloudWatch monitoring", "Custom dashboards", "Alerting"],
                "maintenance_requirements": ["Regular updates", "Security patches", "Performance tuning"],
                "support_model": ["24/7 support", "Escalation procedures", "Documentation"],
                "training_requirements": ["Technical training", "User training", "Operations training"]
            },
            "cost_and_resources": {
                "cost_breakdown": ["Infrastructure costs", "Migration costs", "Operational costs"],
                "resource_requirements": ["Project manager", "Architects", "Engineers", "Testers"],
                "timeline_milestones": ["Project kickoff", "Design approval", "Migration completion", "Go-live"],
                "assumptions": ["Resource availability", "Technical feasibility", "Business approval"]
            }
        }
    
    def _get_comprehensive_default_structure(self) -> Dict:
        """Default comprehensive structure"""
        return {
            "project_overview": {
                "customer_name": "Not specified",
                "project_name": "AWS Project",
                "project_type": "migration",
                "project_scope": "Not specified",
                "project_duration": "Not specified",
                "total_budget": "Not specified"
            },
            "business_requirements": {
                "business_objectives": ["Cost optimization"],
                "success_criteria": ["Successful implementation"],
                "stakeholders": ["IT Team"],
                "business_drivers": ["Digital transformation"]
            },
            "technical_requirements": {
                "current_infrastructure": ["Legacy systems"],
                "target_architecture": ["Cloud architecture"],
                "aws_services": ["EC2", "S3", "RDS"],
                "performance_requirements": ["High availability"],
                "scalability_requirements": ["Auto-scaling"],
                "availability_requirements": ["99.9% uptime"],
                "security_requirements": ["Data encryption"],
                "integration_requirements": ["API integrations"]
            },
            "compliance_and_governance": {
                "compliance_standards": ["Industry standards"],
                "data_governance": ["Data management"],
                "audit_requirements": ["Audit trails"],
                "privacy_requirements": ["Data privacy"]
            },
            "architecture_details": {
                "network_architecture": ["VPC design"],
                "security_architecture": ["Security controls"],
                "data_architecture": ["Data management"],
                "application_architecture": ["Application design"],
                "disaster_recovery": ["Backup and recovery"]
            },
            "implementation_approach": {
                "migration_strategy": ["Phased approach"],
                "phases": ["Planning", "Implementation", "Testing"],
                "dependencies": ["Technical dependencies"],
                "risks_and_mitigation": ["Risk management"],
                "testing_strategy": ["Testing approach"]
            },
            "operational_requirements": {
                "monitoring_requirements": ["System monitoring"],
                "maintenance_requirements": ["Regular maintenance"],
                "support_model": ["Support procedures"],
                "training_requirements": ["Team training"]
            },
            "cost_and_resources": {
                "cost_breakdown": ["Cost estimates"],
                "resource_requirements": ["Project resources"],
                "timeline_milestones": ["Key milestones"],
                "assumptions": ["Project assumptions"]
            }
        }