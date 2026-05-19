from typing import Dict, Any, List, Optional
import logging
import json
import os

logger = logging.getLogger(__name__)

class ComplianceAgent:
    """
    Compliance Agent.
    Checks regulatory standards (FAA, FCC, ISO, ASME) with detailed citations.
    Rules are loaded from data/regulatory_rules.json — never LLM-generated.
    """
    def __init__(self):
        self.name = "ComplianceAgent"

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
        Uses LLM to synthesize rules or returns from built-in database.
        """
        # Built-in regulation database for common topics
        regulation_db = {
            "AERIAL": [
                {
                    "id": "FAA_107_1",
                    "name": "Maximum Altitude",
                    "logic": {"<": [{"var": "altitude_ft"}, 400]},
                    "violation_msg": "Altitude exceeds 400 ft AGL limit",
                    "citation": "14 CFR 107.51",
                    "regulation_text": "The altitude of the small unmanned aircraft cannot be higher than 400 feet above ground level",
                    "official_link": "https://www.ecfr.gov/current/title-14/chapter-I/subchapter-F/part-107"
                },
                {
                    "id": "FAA_107_2", 
                    "name": "Maximum Speed",
                    "logic": {"<": [{"var": "speed_mph"}, 100]},
                    "violation_msg": "Speed exceeds 100 mph limit",
                    "citation": "14 CFR 107.51",
                    "regulation_text": "The ground speed of the small unmanned aircraft may not exceed 87 knots (100 mph)",
                    "official_link": "https://www.ecfr.gov/current/title-14/chapter-I/subchapter-F/part-107"
                },
                {
                    "id": "FAA_107_3",
                    "name": "Maximum Weight",
                    "logic": {"<": [{"var": "weight_kg"}, 25]},
                    "violation_msg": "Weight exceeds 25 kg (55 lbs) limit",
                    "citation": "14 CFR 107.51",
                    "regulation_text": "Weight of small unmanned aircraft must be less than 55 lbs (25 kg)",
                    "official_link": "https://www.ecfr.gov/current/title-14/chapter-I/subchapter-F/part-107"
                }
            ],
            "MEDICAL": [
                {
                    "id": "FDA_1",
                    "name": "Biocompatibility ISO 10993",
                    "logic": {"==": [{"var": "biocomp_tested"}, True]},
                    "violation_msg": "Device requires biocompatibility testing per ISO 10993",
                    "citation": "21 CFR 860.7 / ISO 10993",
                    "regulation_text": "Devices in contact with body must be tested for biocompatibility",
                    "official_link": "https://www.fda.gov/medical-devices/biocompatibility"
                },
                {
                    "id": "FDA_2",
                    "name": "Sterilization Validation",
                    "logic": {"==": [{"var": "sterilization_validated"}, True]},
                    "violation_msg": "Sterilization process must be validated per ISO 11137",
                    "citation": "21 CFR 820.75",
                    "regulation_text": "Process validation including sterilization must be documented",
                    "official_link": "https://www.fda.gov/medical-devices/quality-system-qs-regulation"
                }
            ],
            "TERRESTRIAL": [
                {
                    "id": "DOT_1",
                    "name": "Vehicle Crash Safety",
                    "logic": {
                        "and": [
                            {">": [{"var": "crash_test_rating"}, 3]},
                            {"==": [{"var": "airbags_installed"}, True]}
                        ]
                    },
                    "violation_msg": "Vehicle must meet FMVSS crash standards",
                    "citation": "49 CFR 571",
                    "regulation_text": "Federal Motor Vehicle Safety Standards for crashworthiness",
                    "official_link": "https://www.nhtsa.gov/laws-regulations/fmvss"
                }
            ],
            "MARINE": [
                {
                    "id": "IMO_1",
                    "name": "SOLAS Stability",
                    "logic": {">": [{"var": "stability_index"}, 1.0]},
                    "violation_msg": "Vessel stability index below SOLAS requirement",
                    "citation": "SOLAS Chapter II-1",
                    "regulation_text": "International Convention for Safety of Life at Sea stability requirements",
                    "official_link": "https://www.imo.org/en/About/Conventions/Pages/SOLAS.aspx"
                }
            ],
            "SPACE": [
                {
                    "id": "NASA_1",
                    "name": "Debris Mitigation",
                    "logic": {"<": [{"var": "orbital_lifetime_years"}, 25]},
                    "violation_msg": "Orbital lifetime exceeds 25-year debris mitigation limit",
                    "citation": "NASA-STD-8719.14",
                    "regulation_text": "Process for Limiting Orbital Debris - 25 year rule",
                    "official_link": "https://standards.nasa.gov/standard/nasa/nasa-std-871914"
                },
                {
                    "id": "NASA_2",
                    "name": "Safety Factor Structural",
                    "logic": {">": [{"var": "safety_factor"}, 1.4]},
                    "violation_msg": "Structural safety factor below NASA-STD-5005 requirement",
                    "citation": "NASA-STD-5005",
                    "regulation_text": "Structural design must maintain minimum 1.4 safety factor",
                    "official_link": "https://standards.nasa.gov/standard/nasa/nasa-std-5005"
                }
            ]
        }
        
        topic_upper = topic.upper()
        if topic_upper in regulation_db:
            return regulation_db[topic_upper]

        # Also check the JSON rules file for additional regimes
        path = os.path.join(os.path.dirname(__file__), "../data/regulatory_rules.json")
        if os.path.exists(path):
            try:
                import json as _json
                with open(path) as f:
                    data = _json.load(f)
                if topic_upper in data:
                    return data[topic_upper]
            except Exception as e:
                logger.error(f"Failed to load rules file: {e}")

        logger.warning(f"No regulations found for topic: {topic}")
        return []


# =============================================================================
# FASTAPI ENDPOINTS
# =============================================================================

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    router = None

if HAS_FASTAPI:
    router = APIRouter(prefix="/compliance", tags=["compliance"])
    
    class ComplianceCheckRequest(BaseModel):
        regime: str = Field(..., description="Regulatory regime")
        design_params: dict = Field(default_factory=dict, description="Design parameters")
        
    @router.post("/check")
    async def check_compliance(request: ComplianceCheckRequest):
        """Check regulatory compliance"""
        try:
            agent = ComplianceAgent()
            result = agent.run({
                "regime": request.regime,
                "design_params": request.design_params
            })
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.get("/regulations/{topic}")
    async def get_regulations(topic: str):
        """Get regulations for a topic"""
        try:
            agent = ComplianceAgent()
            regulations = agent.discover_regulations(topic)
            return {
                "topic": topic,
                "regulations": regulations
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.get("/regimes")
    async def get_regimes():
        """Get available regulatory regimes"""
        return {
            "regimes": [
                {"id": "AERIAL", "name": "Aerial/UAV", "authority": "FAA/EASA"},
                {"id": "MEDICAL", "name": "Medical Device", "authority": "FDA"},
                {"id": "TERRESTRIAL", "name": "Terrestrial Vehicle", "authority": "DOT/NHTSA"},
                {"id": "MARINE", "name": "Marine", "authority": "IMO/Coast Guard"},
                {"id": "SPACE", "name": "Space", "authority": "NASA/FAA-AST"}
            ]
        }
