from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ComplianceAgent:
    """
    Compliance Agent.
    Checks regulatory standards (FAA, ISO, ASME).
    """
    def __init__(self):
        self.name = "ComplianceAgent"

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate regulatory compliance using dynamic JSON Logic rules.
        Expected Params:
        - regime: str (e.g., "AERIAL", "MEDICAL")
        - design_params: Dict (Flattened metrics like weight, voltage, speed)
        """
        logger.info(f"{self.name} checking regulatory compliance (Dynamic Engine)...")
        
        regime = params.get("regime", "AERIAL")
        design_metrics = params.get("design_params", {})
        
        # 1. Load Dynamic Rules
        rules = self._load_rules(regime)
        
        # 2. Evaluate Logic
        # We use a lightweight implementation of JsonLogic or simple python eval for MVP
        # Format: {"rule_id": "FAA_PART_107", "logic": {"<": [{"var": "mass_kg"}, 25.0]}, "violation_msg": "Too heavy"}
        
        compliance_report = {
            "status": "compliant",
            "passed_rules": [],
            "failed_rules": [],
            "logs": []
        }
        
        for rule in rules:
            rule_id = rule.get("id", "unknown")
            logic = rule.get("logic", {})
            msg = rule.get("violation_msg", "Regulatory violation")
            
            try:
                # Simple recursive evaluator for basic operators (>, <, ==, and, or)
                # In production, use `json-logic-py` library
                result = self._evaluate_logic(logic, design_metrics)
                
                if result:
                    compliance_report["passed_rules"].append(rule_id)
                else:
                    compliance_report["failed_rules"].append({
                        "id": rule_id,
                        "description": msg
                    })
                    compliance_report["status"] = "non_compliant"
            except Exception as e:
                compliance_report["logs"].append(f"Error evaluating rule {rule_id}: {e}")

        compliance_report["logs"].append(f"Regime: {regime}. Checked {len(rules)} dynamic rules.")
        return compliance_report

    def _load_rules(self, regime: str) -> List[Dict]:
        """Load rules from data/regulatory_rules.json"""
        import json
        import os
        path = os.path.join(os.path.dirname(__file__), "../data/regulatory_rules.json")
        default_rules = []
        
        if not os.path.exists(path):
            # Create default if missing
            default_data = {
                "AERIAL": [
                    {
                        "id": "FAA_PART_107_WEIGHT", 
                        "logic": {"<=": [{"var": "mass_kg"}, 25.0]}, 
                        "violation_msg": "Exceeds FAA Part 107 max weight (25kg)"
                    }
                ]
            }
            try:
                with open(path, 'w') as f: json.dump(default_data, f, indent=2)
            except: pass
            return default_data.get(regime, [])
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                return data.get(regime, [])
        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            return []

    def _evaluate_logic(self, logic: Any, data: Dict) -> bool:
        """Lightweight JSON Logic evaluator."""
        # logger.info(f"DEBUG: Logic={logic}")
        if isinstance(logic, bool): return logic
        if not isinstance(logic, dict): return bool(logic)
        
        try:
            op = list(logic.keys())[0]
            args = logic[op]
            
            if op == "var":
                return data.get(args, 0.0)
                
            # Recursive evaluation
            values = []
            if isinstance(args, list):
                values = args
            else:
                values = [args]
                
            # Explicitly recurse
            eval_args = []
            for v in values:
                eval_args.append(self._evaluate_logic(v, data))
            
            if op == ">": return eval_args[0] > eval_args[1]
            if op == ">=": return eval_args[0] >= eval_args[1]
            if op == "<": return eval_args[0] < eval_args[1]
            if op == "<=": return eval_args[0] <= eval_args[1]
            if op == "==": return eval_args[0] == eval_args[1]
            if op == "and": return all(eval_args)
            if op == "or": return any(eval_args)
            if op == "!": return not eval_args[0]
            
            return False
        except Exception as e:
            logger.error(f"Logic Error for {logic}: {e}")
            raise e

    def update_rules(self, regime: str, new_rules: List[Dict]):
        """
        Called by ConversationalAgent when new regulations are extracted.
        """
        import json
        import os
        path = os.path.join(os.path.dirname(__file__), "../data/regulatory_rules.json")
        
        current_data = {}
        if os.path.exists(path):
            try:
                with open(path, 'r') as f: current_data = json.load(f)
            except: pass
            
        current_data[regime] = new_rules # Overwrite or Merge? Overwrite for now.
        
        with open(path, 'w') as f:
            json.dump(current_data, f, indent=2)
        logger.info(f"Updated compliance rules for {regime}")
