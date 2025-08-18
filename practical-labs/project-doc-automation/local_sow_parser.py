import requests
import json
import re
from typing import Dict

class LocalSOWParser:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "llama2:7b"
    
    def parse_sow_file(self, file_path: str) -> str:
        """Extract text from SOW file"""
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_path.endswith('.pdf'):
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        elif file_path.endswith('.docx'):
            from docx import Document
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def analyze_sow(self, sow_text: str) -> Dict:
        """Analyze SOW using local Ollama"""
        
        prompt = f"""
        Analyze this Statement of Work document and extract key information.
        Return ONLY a JSON object with this exact structure:
        
        {{
            "customer_name": "company or client name",
            "project_name": "project title",
            "project_type": "migration or modernization or new_deployment",
            "aws_services": ["list of AWS services mentioned"],
            "timeline": "project duration",
            "budget": "budget information if mentioned",
            "compliance_requirements": ["compliance standards mentioned"],
            "technical_requirements": ["technical needs mentioned"],
            "business_objectives": ["business goals mentioned"]
        }}
        
        SOW Document:
        {sow_text[:3000]}
        
        Extract only factual information. If not found, use "Not specified".
        """
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._extract_json(result.get('response', ''))
            else:
                print(f"Ollama error: {response.status_code}")
                return self._fallback_extraction(sow_text)
                
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return self._fallback_extraction(sow_text)
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from AI response"""
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        
        if json_match:
            try:
                json_str = json_match.group()
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        return self._get_default_structure()
    
    def _fallback_extraction(self, sow_text: str) -> Dict:
        """Basic regex extraction as fallback"""
        
        customer_patterns = [
            r'(?:client|customer|company):\s*([^\n]+)',
            r'(?:for|client)\s+([A-Z][a-zA-Z\s&]+?)(?:\s|,|\.)',
        ]
        
        project_patterns = [
            r'(?:project|title):\s*([^\n]+)',
            r'(?:project|migration|modernization)\s+([^\n]+)',
        ]
        
        customer_name = "Not specified"
        for pattern in customer_patterns:
            match = re.search(pattern, sow_text, re.IGNORECASE)
            if match:
                customer_name = match.group(1).strip()
                break
        
        project_name = "Not specified"
        for pattern in project_patterns:
            match = re.search(pattern, sow_text, re.IGNORECASE)
            if match:
                project_name = match.group(1).strip()
                break
        
        aws_services = []
        aws_keywords = ['EC2', 'S3', 'RDS', 'Lambda', 'EKS', 'ECS', 'VPC', 'CloudFormation']
        for service in aws_keywords:
            if service.lower() in sow_text.lower():
                aws_services.append(service)
        
        if not aws_services:
            aws_services = ["EC2", "S3", "RDS"]
        
        return {
            "customer_name": customer_name,
            "project_name": project_name,
            "project_type": "migration",
            "aws_services": aws_services,
            "timeline": "Not specified",
            "budget": "Not specified",
            "compliance_requirements": ["Standard"],
            "technical_requirements": ["High availability", "Scalability"],
            "business_objectives": ["Cost optimization", "Improved performance"]
        }
    
    def _get_default_structure(self) -> Dict:
        """Default structure if all else fails"""
        return {
            "customer_name": "Not specified",
            "project_name": "AWS Project",
            "project_type": "migration",
            "aws_services": ["EC2", "S3", "RDS"],
            "timeline": "Not specified",
            "budget": "Not specified",
            "compliance_requirements": ["Standard"],
            "technical_requirements": ["High availability"],
            "business_objectives": ["Cost optimization"]
        }