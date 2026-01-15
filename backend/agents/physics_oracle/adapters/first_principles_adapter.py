import os
import logging
import traceback
import sys
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class FirstPrinciplesAdapter:
    """
    Code Interpreter & First Principles Solver.
    Uses an LLM to generate Python code for analytical physics problems, 
    then executes it to get the result.
    """
    
    def __init__(self):
        self.name = "LLM-Code-Interpreter"
        self.llm = self._init_llm()
        
    def _init_llm(self):
        """Detect and initialize available LLM provider"""
        try:
            # Check environment for keys
            if os.getenv("OPENAI_API_KEY"):
                # Dynamic import to avoid circular deps or hard failures
                from llm.openai_provider import OpenAIProvider
                logger.info("[CODE INTERPRETER] Powered by OpenAI GPT-4")
                return OpenAIProvider()
            
            # Check for Ollama (heuristic: assume localhost if no key but requested)
            # Ideally we check a config, but for now we look for the module
            try:
                import requests
                # quick ping to ollama default port? Skip for speed.
                from llm.ollama_provider import OllamaProvider
                logger.info("[CODE INTERPRETER] Powered by Ollama (Local)")
                return OllamaProvider(model_name="llama3.2") 
            except ImportError:
                pass
                
        except Exception as e:
            logger.error(f"[CODE INTERPRETER] Failed to init LLM: {e}")
            
        logger.warning("[CODE INTERPRETER] No LLM provider found. Interpreter disabled.")
        return None

    def run_simulation(self, params: Dict[str, Any], query: str = "") -> Dict[str, Any]:
        """
        Generate and Run Solver Code.
        """
        if not query:
             return {"status": "error", "message": "Query string required for Code Interpreter."}

        if not self.llm:
             return {
                 "status": "error", 
                 "message": "LLM Provider not available. Ensure OPENAI_API_KEY is set or Ollama is running."
             }

        logger.info(f"[CODE INTERPRETER] Analyzing: {query}")

        # 1. Generate Python Code
        system_prompt = """
        You are a generic Physics Code Interpreter.
        Your goal is to WRITE A PYTHON SCRIPT to solve the user's question.
        
        GUIDELINES:
        1. Identify the physics domain, variables, and required formulas.
        2. Write clean, executable Python code.
        3. Calculate the answer.
        4. Assign the final result to a dictionary variable named `result`.
        5. `result` MUST have keys: 
           - 'answer': (float or string) the simplified value
           - 'unit': (str) the unit
           - 'explanation': (str) brief derivation steps
           - 'raw_value': (float) precise value
        6. IMPORT necessary libraries (math, numpy).
        7. OUTPUT ONLY THE CODE. Do not include markdown blocks (```python).
        """
        
        try:
            # Generate code
            code_response = self.llm.generate(query, system_prompt=system_prompt)
            
            # Cleanup markdown
            code = code_response.replace("```python", "").replace("```", "").strip()
            
            logger.info("[CODE INTERPRETER] Generated Code:\n" + code)
            
            # 2. Execute Code (Sandbox)
            # We provide a safe(ish) dict. 
            # In production, use dedicated sandbox (containers).
            local_scope = {}
            
            exec(code, {"__builtins__": __builtins__, "import": __import__}, local_scope)
            
            if "result" not in local_scope:
                 return {
                     "status": "error", 
                     "message": "Generated code failed to produce 'result' dictionary.",
                     "code": code
                 }
            
            result = local_scope["result"]
            
            return {
                "status": "solved",
                "solver": "LLM Code Interpreter",
                "intent_analysis": "Automated Derivation",
                "code_executed": code,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Code Interpreter Error: {e}")
            return {
                "status": "error", 
                "message": f"Execution Failed: {str(e)}", 
                "traceback": traceback.format_exc()
            }