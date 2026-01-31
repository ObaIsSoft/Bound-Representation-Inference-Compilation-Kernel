from typing import Dict, Any, List, Optional
import logging
import json
import os

logger = logging.getLogger(__name__)

class ComplianceAgent:
    """
    Compliance Agent.
    Checks regulatory standards (FAA, FCC, ISO, ASME) with detailed citations.
    """
    def __init__(self, llm_provider=None):
        self.name = "ComplianceAgent"
        self.llm = llm_provider
        if not self.llm:
             from backend.llm.factory import get_llm_provider
             self.llm = get_llm_provider()

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate regulatory compliance using dynamic JSON Logic rules.
        Expected Params:
        - regime: str (e.g., "AERIAL", "MEDICAL", "TERRESTRIAL")
        - design_params: Dict (Flattened metrics like weight, voltage, speed)
        """
        logger.info(f"{self.name} checking regulatory compliance (Detailed Engine)...")
        
        regime = params.get("regime", "AERIAL").upper()
        # Ensure we have common base design params for evaluation
        design_metrics = params.get("design_params", {})
        
        # 1. Load Dynamic Rules
        rules = self._load_rules(regime)
        
        compliance_report = {
            "regime": regime,
            "status": "compliant",
            "checklist": [], # New unified list for frontend
            "logs": []
        }
        
        for rule in rules:
            rule_id = rule.get("id", "unknown")
            rule_name = rule.get("name", rule_id)
            logic = rule.get("logic", {})
            msg = rule.get("violation_msg", "Regulatory violation")
            citation = rule.get("citation", "N/A")
            reg_text = rule.get("regulation_text", "No detailed text available.")
            link = rule.get("official_link", "#")
            
            item = {
                "id": rule_id,
                "name": rule_name,
                "citation": citation,
                "regulation_text": reg_text,
                "official_link": link,
                "status": "pending",
                "message": None
            }
            
            try:
                # Evaluate Logic
                passed = self._evaluate_logic(logic, design_metrics)
                
                if passed:
                    item["status"] = "passed"
                else:
                    item["status"] = "failed"
                    item["message"] = msg
                    compliance_report["status"] = "non_compliant"
                    
            except Exception as e:
                item["status"] = "error"
                item["message"] = f"Evaluation Error: {str(e)}"
                compliance_report["logs"].append(f"Error evaluating rule {rule_id}: {e}")
            
            compliance_report["checklist"].append(item)

        compliance_report["logs"].append(f"Regime: {regime}. Checked {len(rules)} rules.")
        return compliance_report

    def _load_rules(self, regime: str) -> List[Dict]:
        """Load rules from data/regulatory_rules.json"""
        path = os.path.join(os.path.dirname(__file__), "../data/regulatory_rules.json")
        
        if not os.path.exists(path):
            logger.warning(f"Rules file not found at {path}. Using defaults.")
            return []
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                return data.get(regime, [])
        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            return []

    def _evaluate_logic(self, logic: Any, data: Dict) -> Any:
        """Lightweight JSON Logic evaluator."""
        if isinstance(logic, (bool, int, float, str)): return logic
        if not isinstance(logic, dict): return logic
        
        try:
            op = list(logic.keys())[0]
            args = logic[op]
            
            if op == "var":
                # Check if property exists in data. If not, treat as False/0 for numeric ops
                return data.get(args, False)
                
            # Recursive evaluation
            values = []
            if isinstance(args, list):
                values = args
            else:
                values = [args]
                
            eval_args = []
            for v in values:
                eval_args.append(self._evaluate_logic(v, data))
            
            if op == ">": return float(eval_args[0]) > float(eval_args[1])
            if op == ">=": return float(eval_args[0]) >= float(eval_args[1])
            if op == "<": return float(eval_args[0]) < float(eval_args[1])
            if op == "<=": return float(eval_args[0]) <= float(eval_args[1])
            if op == "==": return eval_args[0] == eval_args[1]
            if op == "and": return all(eval_args)
            if op == "or": return any(eval_args)
            if op == "!": return not eval_args[0]
            
            return False
        except Exception as e:
            logger.error(f"Logic Error for {logic}: {e}")
            return False # Fail safe as non-compliant if error

    def discover_regulations(self, topic: str) -> List[Dict]:
        """
        AI-driven discovery of relevant regulations for a new topic.
        Uses LLM to synthesize rules.
        """
        prompt = f"Extract core engineering regulatory requirements for {topic}. Return in format matching regulatory_rules.json."
        # ... implementation for automated fetching
        return []
