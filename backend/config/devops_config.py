"""
DevOps Agent Configuration.
Externalizes constants for DevOpsAgent.
"""

DOCKER_AUDIT_RULES = [
    {"check": "USER", "message": "Security: Container runs as root (missing USER instruction)"},
    {"check": ":latest", "message": "Stability: Using ':latest' tag is discouraged"},
    {"check": "sudo", "message": "Security: Avoid using sudo in Dockerfile"}
]

CI_CD_TEMPLATE = """
name: CI/CD Pipeline
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Test
      run: python -m pytest
"""

HEALTH_CHECK_TIMEOUT = 5
