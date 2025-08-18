# Lab 2.2: AI-Powered DevOps Assistant
*Build intelligent automation with OpenAI and AWS Bedrock*

## 🎯 **Objectives**
- Create Slack bot with AI integration
- Automate documentation generation
- Build code review assistant

---

## 📋 **Prerequisites**
```bash
pip install openai langchain boto3 slack-sdk flask
export OPENAI_API_KEY="your-key"
export SLACK_BOT_TOKEN="your-token"
```

---

## 🛠️ **Implementation**

### **Step 1: OpenAI Integration**
Create `ai_helper.py`:

```python
import openai
import boto3
from typing import List, Dict
import json

class AIHelper:
    def __init__(self, use_bedrock=False):
        self.use_bedrock = use_bedrock
        if use_bedrock:
            self.bedrock = boto3.client('bedrock-runtime')
        else:
            self.openai_client = openai.OpenAI()
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate AI response using OpenAI or Bedrock"""
        full_prompt = f"Context: {context}\n\nQuery: {prompt}\n\nResponse:"
        
        if self.use_bedrock:
            return self._bedrock_response(full_prompt)
        else:
            return self._openai_response(full_prompt)
    
    def _openai_response(self, prompt: str) -> str:
        """Generate response using OpenAI"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a DevOps expert assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _bedrock_response(self, prompt: str) -> str:
        """Generate response using AWS Bedrock"""
        try:
            body = json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": 500,
                "temperature": 0.7
            })
            
            response = self.bedrock.invoke_model(
                body=body,
                modelId="anthropic.claude-v2",
                accept="application/json",
                contentType="application/json"
            )
            
            response_body = json.loads(response.get('body').read())
            return response_body.get('completion', 'No response generated')
        except Exception as e:
            return f"Error: {str(e)}"
    
    def analyze_logs(self, log_content: str) -> str:
        """Analyze log content for issues"""
        prompt = f"""
        Analyze the following log content and provide:
        1. Summary of key events
        2. Any errors or warnings found
        3. Recommendations for investigation
        
        Log content:
        {log_content[:2000]}  # Limit content
        """
        return self.generate_response(prompt)
    
    def generate_documentation(self, code: str, description: str) -> str:
        """Generate documentation for code"""
        prompt = f"""
        Generate comprehensive documentation for the following code:
        
        Description: {description}
        
        Code:
        {code}
        
        Include:
        - Purpose and functionality
        - Parameters and return values
        - Usage examples
        - Best practices
        """
        return self.generate_response(prompt)
    
    def review_code(self, code: str, language: str) -> str:
        """Review code for best practices"""
        prompt = f"""
        Review the following {language} code and provide:
        1. Code quality assessment
        2. Security considerations
        3. Performance improvements
        4. Best practice recommendations
        
        Code:
        {code}
        """
        return self.generate_response(prompt)

# Usage example
if __name__ == "__main__":
    ai = AIHelper(use_bedrock=False)
    
    # Test log analysis
    sample_log = """
    2024-01-15 10:30:15 INFO Starting application
    2024-01-15 10:30:16 ERROR Database connection failed
    2024-01-15 10:30:17 WARN Retrying connection
    2024-01-15 10:30:18 INFO Connection established
    """
    
    analysis = ai.analyze_logs(sample_log)
    print("Log Analysis:")
    print(analysis)
```

### **Step 2: Slack Bot Integration**
Create `slack_bot.py`:

```python
import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from flask import Flask, request, jsonify
import json
from ai_helper import AIHelper

app = Flask(__name__)
slack_client = WebClient(token=os.environ.get('SLACK_BOT_TOKEN'))
ai_helper = AIHelper()

class SlackBot:
    def __init__(self):
        self.client = slack_client
        self.ai = ai_helper
    
    def send_message(self, channel: str, text: str):
        """Send message to Slack channel"""
        try:
            response = self.client.chat_postMessage(
                channel=channel,
                text=text
            )
            return response
        except SlackApiError as e:
            print(f"Error sending message: {e}")
    
    def handle_mention(self, event_data: dict):
        """Handle bot mentions"""
        text = event_data.get('text', '')
        channel = event_data.get('channel')
        user = event_data.get('user')
        
        # Remove bot mention from text
        clean_text = text.split('>', 1)[-1].strip()
        
        # Process different commands
        if 'analyze logs' in clean_text.lower():
            response = self._handle_log_analysis(clean_text)
        elif 'review code' in clean_text.lower():
            response = self._handle_code_review(clean_text)
        elif 'help' in clean_text.lower():
            response = self._get_help_message()
        else:
            response = self.ai.generate_response(
                clean_text, 
                "You are a DevOps assistant helping with infrastructure and automation tasks."
            )
        
        self.send_message(channel, response)
    
    def _handle_log_analysis(self, text: str) -> str:
        """Handle log analysis requests"""
        # In a real implementation, you'd fetch logs from CloudWatch or other sources
        return """
        🔍 **Log Analysis Request Received**
        
        To analyze logs, please:
        1. Specify the log source (CloudWatch group, file path, etc.)
        2. Provide time range
        3. Share log content or I can fetch it
        
        Example: `@bot analyze logs from /aws/lambda/my-function for last 1 hour`
        """
    
    def _handle_code_review(self, text: str) -> str:
        """Handle code review requests"""
        return """
        📝 **Code Review Request Received**
        
        To review code, please:
        1. Share the code snippet
        2. Specify the programming language
        3. Mention any specific concerns
        
        Example: 
        ```
        @bot review code
        Language: Python
        
        def process_data(data):
            return data.upper()
        ```
        """
    
    def _get_help_message(self) -> str:
        """Get help message"""
        return """
        🤖 **DevOps AI Assistant Help**
        
        **Available Commands:**
        • `analyze logs` - Analyze log files for issues
        • `review code` - Review code for best practices
        • `generate docs` - Create documentation
        • `help` - Show this help message
        
        **Examples:**
        • `@bot analyze logs from CloudWatch`
        • `@bot review this Python code: [code]`
        • `@bot what's the best way to deploy on AWS?`
        
        **Features:**
        ✅ Log analysis and troubleshooting
        ✅ Code review and suggestions
        ✅ Documentation generation
        ✅ AWS best practices guidance
        ✅ Infrastructure automation help
        """

bot = SlackBot()

@app.route('/slack/events', methods=['POST'])
def slack_events():
    """Handle Slack events"""
    data = request.json
    
    # Handle URL verification
    if data.get('type') == 'url_verification':
        return jsonify({'challenge': data.get('challenge')})
    
    # Handle app mentions
    if data.get('event', {}).get('type') == 'app_mention':
        bot.handle_mention(data['event'])
    
    return jsonify({'status': 'ok'})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
```

### **Step 3: Documentation Generator**
Create `doc_generator.py`:

```python
import os
import ast
import boto3
from typing import List, Dict
from ai_helper import AIHelper

class DocumentationGenerator:
    def __init__(self):
        self.ai = AIHelper()
        self.s3 = boto3.client('s3')
    
    def analyze_python_file(self, file_path: str) -> Dict:
        """Analyze Python file structure"""
        with open(file_path, 'r') as file:
            content = file.read()
        
        try:
            tree = ast.parse(content)
            
            functions = []
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node),
                        'line_number': node.lineno
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        'name': node.name,
                        'docstring': ast.get_docstring(node),
                        'line_number': node.lineno
                    })
            
            return {
                'file_path': file_path,
                'functions': functions,
                'classes': classes,
                'content': content
            }
        except Exception as e:
            return {'error': str(e)}
    
    def generate_function_docs(self, function_info: Dict, code_context: str) -> str:
        """Generate documentation for a function"""
        prompt = f"""
        Generate comprehensive documentation for this Python function:
        
        Function: {function_info['name']}
        Arguments: {function_info['args']}
        Current docstring: {function_info.get('docstring', 'None')}
        
        Code context:
        {code_context}
        
        Generate:
        1. Clear description of purpose
        2. Parameter descriptions with types
        3. Return value description
        4. Usage example
        5. Any exceptions that might be raised
        """
        
        return self.ai.generate_response(prompt)
    
    def generate_readme(self, project_path: str) -> str:
        """Generate README for project"""
        # Analyze project structure
        python_files = []
        for root, dirs, files in os.walk(project_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        # Analyze main files
        project_analysis = []
        for file_path in python_files[:5]:  # Limit to first 5 files
            analysis = self.analyze_python_file(file_path)
            if 'error' not in analysis:
                project_analysis.append(analysis)
        
        # Generate README content
        prompt = f"""
        Generate a comprehensive README.md for this Python project:
        
        Project path: {project_path}
        Python files found: {len(python_files)}
        
        Key files analysis:
        {json.dumps(project_analysis, indent=2, default=str)[:2000]}
        
        Generate README with:
        1. Project title and description
        2. Installation instructions
        3. Usage examples
        4. API documentation
        5. Contributing guidelines
        6. License information
        """
        
        return self.ai.generate_response(prompt)
    
    def generate_api_docs(self, functions: List[Dict]) -> str:
        """Generate API documentation"""
        prompt = f"""
        Generate API documentation for these functions:
        
        {json.dumps(functions, indent=2, default=str)}
        
        Create documentation in markdown format with:
        1. Function signatures
        2. Parameter descriptions
        3. Return values
        4. Example usage
        5. Error handling
        """
        
        return self.ai.generate_response(prompt)
    
    def save_documentation(self, content: str, output_path: str, s3_bucket: str = None):
        """Save documentation to file and optionally S3"""
        # Save locally
        with open(output_path, 'w') as file:
            file.write(content)
        
        print(f"Documentation saved to: {output_path}")
        
        # Upload to S3 if bucket specified
        if s3_bucket:
            try:
                s3_key = f"docs/{os.path.basename(output_path)}"
                self.s3.upload_file(output_path, s3_bucket, s3_key)
                print(f"Documentation uploaded to: s3://{s3_bucket}/{s3_key}")
            except Exception as e:
                print(f"Error uploading to S3: {e}")

# Usage example
if __name__ == "__main__":
    doc_gen = DocumentationGenerator()
    
    # Generate documentation for current directory
    project_path = "."
    readme_content = doc_gen.generate_readme(project_path)
    
    # Save documentation
    doc_gen.save_documentation(
        readme_content, 
        "AI_GENERATED_README.md",
        s3_bucket="your-docs-bucket"
    )
    
    print("Documentation generation complete!")
```

### **Step 4: Code Review Assistant**
Create `code_reviewer.py`:

```python
import os
import subprocess
import json
from typing import List, Dict
from ai_helper import AIHelper

class CodeReviewer:
    def __init__(self):
        self.ai = AIHelper()
    
    def review_file(self, file_path: str) -> Dict:
        """Review a single file"""
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            
            # Determine language
            language = self._detect_language(file_path)
            
            # Get AI review
            review = self.ai.review_code(content, language)
            
            # Run static analysis if available
            static_analysis = self._run_static_analysis(file_path, language)
            
            return {
                'file_path': file_path,
                'language': language,
                'ai_review': review,
                'static_analysis': static_analysis,
                'line_count': len(content.split('\n'))
            }
        except Exception as e:
            return {'error': str(e)}
    
    def review_pull_request(self, repo_path: str, base_branch: str = 'main') -> Dict:
        """Review changes in current branch vs base branch"""
        try:
            # Get changed files
            cmd = f"cd {repo_path} && git diff --name-only {base_branch}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {'error': 'Failed to get git diff'}
            
            changed_files = result.stdout.strip().split('\n')
            changed_files = [f for f in changed_files if f.strip()]
            
            reviews = []
            for file_path in changed_files:
                full_path = os.path.join(repo_path, file_path)
                if os.path.exists(full_path):
                    review = self.review_file(full_path)
                    reviews.append(review)
            
            # Generate summary
            summary = self._generate_pr_summary(reviews)
            
            return {
                'changed_files': len(changed_files),
                'reviews': reviews,
                'summary': summary
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.go': 'Go',
            '.rs': 'Rust',
            '.cpp': 'C++',
            '.c': 'C',
            '.sh': 'Shell',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.json': 'JSON'
        }
        
        return language_map.get(ext, 'Unknown')
    
    def _run_static_analysis(self, file_path: str, language: str) -> Dict:
        """Run static analysis tools"""
        results = {}
        
        try:
            if language == 'Python':
                # Run flake8 if available
                cmd = f"flake8 {file_path}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    results['flake8'] = 'No issues found'
                else:
                    results['flake8'] = result.stdout
            
            elif language == 'JavaScript':
                # Run eslint if available
                cmd = f"eslint {file_path}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                results['eslint'] = result.stdout if result.stdout else 'No issues found'
        
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _generate_pr_summary(self, reviews: List[Dict]) -> str:
        """Generate pull request summary"""
        total_files = len(reviews)
        total_lines = sum(r.get('line_count', 0) for r in reviews)
        
        # Collect all AI reviews
        all_reviews = '\n'.join([r.get('ai_review', '') for r in reviews])
        
        prompt = f"""
        Generate a pull request review summary based on these individual file reviews:
        
        Total files changed: {total_files}
        Total lines of code: {total_lines}
        
        Individual reviews:
        {all_reviews[:3000]}  # Limit content
        
        Provide:
        1. Overall code quality assessment
        2. Key issues found across files
        3. Security considerations
        4. Performance recommendations
        5. Approval recommendation (Approve/Request Changes/Comment)
        """
        
        return self.ai.generate_response(prompt)
    
    def generate_review_report(self, review_data: Dict, output_path: str):
        """Generate formatted review report"""
        report = f"""# Code Review Report
        
## Summary
- **Files Reviewed**: {review_data.get('changed_files', 'N/A')}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Assessment
{review_data.get('summary', 'No summary available')}

## Individual File Reviews
"""
        
        for review in review_data.get('reviews', []):
            report += f"""
### {review.get('file_path', 'Unknown')}
**Language**: {review.get('language', 'Unknown')}
**Lines**: {review.get('line_count', 'Unknown')}

#### AI Review
{review.get('ai_review', 'No review available')}

#### Static Analysis
{json.dumps(review.get('static_analysis', {}), indent=2)}

---
"""
        
        with open(output_path, 'w') as file:
            file.write(report)
        
        print(f"Review report saved to: {output_path}")

# Usage example
if __name__ == "__main__":
    reviewer = CodeReviewer()
    
    # Review current directory
    review_result = reviewer.review_pull_request(".", "main")
    
    # Generate report
    reviewer.generate_review_report(review_result, "code_review_report.md")
    
    print("Code review complete!")
```

---

## 🚀 **Deployment**

### **Lambda Function for Slack Bot**
Create `lambda_function.py`:

```python
import json
import os
from slack_bot import SlackBot

bot = SlackBot()

def lambda_handler(event, context):
    """AWS Lambda handler for Slack events"""
    try:
        # Parse Slack event
        body = json.loads(event.get('body', '{}'))
        
        # Handle URL verification
        if body.get('type') == 'url_verification':
            return {
                'statusCode': 200,
                'body': json.dumps({'challenge': body.get('challenge')})
            }
        
        # Handle app mentions
        if body.get('event', {}).get('type') == 'app_mention':
            bot.handle_mention(body['event'])
        
        return {
            'statusCode': 200,
            'body': json.dumps({'status': 'ok'})
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### **Deployment Script**
Create `deploy_ai_assistant.sh`:

```bash
#!/bin/bash

# Package Lambda function
zip -r ai-assistant.zip *.py

# Deploy Lambda
aws lambda create-function \
  --function-name ai-devops-assistant \
  --runtime python3.9 \
  --role arn:aws:iam::123456789012:role/lambda-execution-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://ai-assistant.zip \
  --environment Variables='{
    "OPENAI_API_KEY":"'$OPENAI_API_KEY'",
    "SLACK_BOT_TOKEN":"'$SLACK_BOT_TOKEN'"
  }'

# Create API Gateway
aws apigatewayv2 create-api \
  --name ai-assistant-api \
  --protocol-type HTTP \
  --target arn:aws:lambda:us-east-1:123456789012:function:ai-devops-assistant

echo "Deployment complete!"
```

---

## 🧪 **Testing**

Create `test_ai_assistant.py`:

```python
import unittest
from ai_helper import AIHelper
from doc_generator import DocumentationGenerator
from code_reviewer import CodeReviewer

class TestAIAssistant(unittest.TestCase):
    
    def setUp(self):
        self.ai = AIHelper(use_bedrock=False)
        self.doc_gen = DocumentationGenerator()
        self.reviewer = CodeReviewer()
    
    def test_ai_response(self):
        response = self.ai.generate_response("What is DevOps?")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 10)
    
    def test_log_analysis(self):
        sample_log = "ERROR: Database connection failed"
        analysis = self.ai.analyze_logs(sample_log)
        self.assertIn("error", analysis.lower())
    
    def test_documentation_generation(self):
        sample_code = "def hello(): return 'world'"
        docs = self.ai.generate_documentation(sample_code, "Simple function")
        self.assertIsInstance(docs, str)
    
    def test_code_review(self):
        sample_code = "print('hello world')"
        review = self.ai.review_code(sample_code, "Python")
        self.assertIsInstance(review, str)

if __name__ == '__main__':
    unittest.main()
```

---

## 🎯 **Expected Results**

1. **Slack Bot**: Responds to mentions with AI-generated answers
2. **Documentation**: Auto-generated README and API docs
3. **Code Review**: Automated analysis with recommendations
4. **Integration**: Seamless AWS and OpenAI integration

**Completion Time:** 8-10 hours
**Difficulty:** Advanced