from typing import List, Dict, Any, Optional
from .base_ingestor import BaseIngestor
from llm.provider import LLMProvider
from llm.mock_dreamer import MockDreamer

# Schema definition for the LLM
COMPONENT_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["motor", "internal_combustion_engine", "battery", "fastener", "unknown"]},
        "name": {"type": "string"},
        "description": {"type": "string"},
        "mass_g": {"type": "number"},
        "specs": {"type": "object"},
        "geometry_def": {
            "type": "object",
            "properties": {
                "generator": {"type": "string"},
                "params": {"type": "object"}
            }
        },
        "behavior_model": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "reliability": {
                    "type": "object",
                    "properties": {
                        "mtbf_hours": {"type": "integer"},
                        "failure_modes": {"type": "array"}
                    }
                }
            }
        }
    },
    "required": ["category", "name", "specs"]
}

class DatasheetIngestor(BaseIngestor):
    """
    Ingests components by parsing PDF Datasheets or Product Pages using AI.
    Features:
    - VLM extraction of Tech Specs via LLMProvider.
    - Mapping to Universal Component Schema.
    """
    
    def __init__(self, provider: Optional[LLMProvider] = None):
        super().__init__()
        self.source_name = "ai_datasheet_parser"
        # Default to Mock if not provided, or load from Env in production
        self.llm_provider = provider if provider else MockDreamer()

    def fetch_candidates(self, query: str = None) -> List[Dict[str, Any]]:
        # ... (Same as before, placeholder)
        return []

    def parse_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Parses raw text using the LLMProvider.
        """
        system_prompt = """
        You are an expert Data Ingestion Agent.
        Extract component specifications from the provided Datasheet Text.
        Map the data to the provided JSON Schema.
        
        For 'geometry_def', infer the best 'generator' (e.g. 'motor_outrunner', 'battery_pack_prismatic')
        and extract dimensions into 'params'.
        
        For 'behavior_model', extract reliability data like MTBF or Life Expectancy.
        """
        
        try:
            # The core LLM Call
            response_data = self.llm_provider.generate_json(
                prompt=f"Extract data from this text:\n\n{text}",
                schema=COMPONENT_SCHEMA,
                system_prompt=system_prompt
            )
            
            # Post-process: Add Metadata
            response_data["id"] = f"ingested_{response_data.get('name', 'unknown').replace(' ', '_').lower()}"
            response_data["mass_g"] = {"nominal": response_data.get("mass_g", 0.0)}
            response_data["cost_usd"] = {"nominal": 0.0}
            response_data["metadata"] = {"source": "datasheet_llm", "original_text_snippet": text[:50] + "..."}
            
            # Wrap in list as fetch_candidates returns list
            return [response_data]
            
        except Exception as e:
            print(f"LLM Parsing Error: {e}")
            return []

    def parse_url(self, url: str) -> Dict[str, Any]:
        return {}

