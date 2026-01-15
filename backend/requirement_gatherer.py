"""
Enhanced Conversational Agent with Multi-Turn Requirement Gathering

This module extends the base ConversationalAgent with intelligent requirement
gathering capabilities. It asks clarifying questions before proceeding to design.
"""

from typing import Dict, Any, List, Optional
import logging
from agents.conversational_agent import ConversationalAgent
from conversation_state import ConversationState
from requirement_templates import template_registry, RequirementPriority

logger = logging.getLogger(__name__)

class RequirementGatherer:
    """Handles intelligent requirement gathering through conversation"""
    
    def __init__(self, conversational_agent: ConversationalAgent):
        self.agent = conversational_agent
        self.template_registry = template_registry
    
    def identify_design_type(self, user_input: str, intent: str, entities: Dict) -> Optional[str]:
        """Identify the type of design from user input"""
        # Use LLM to classify design type
        # Get list of available templates
        templates = self.template_registry.templates
        categories = "\n".join([f"        - {name} ({tmpl.description})" for name, tmpl in templates.items()])
        
        prompt = f"""
        Analyze this design request and classify it into one of these categories:
{categories}
        - custom (anything else)
        
        User request: "{user_input}"
        
        Return only the category name.
        """
        
        try:
            response = self.agent.provider.generate(
                prompt=prompt,
                system_prompt="You are a design classifier. Be concise."
            )
            design_type = response.strip().lower()
            
            # Validate it's a known template
            if design_type in self.template_registry.templates:
                return design_type
            return "custom"
        except Exception as e:
            logger.error(f"Failed to identify design type: {e}")
            return None
    
    def generate_questions(self, 
                          conversation_state: ConversationState,
                          max_questions: int = 3) -> List[str]:
        """Generate clarifying questions based on missing requirements"""
        
        if not conversation_state.design_type:
            return ["What type of system are you designing? (e.g., drone, rover, robot arm, etc.)"]
        
        template = self.template_registry.get(conversation_state.design_type)
        if not template:
            return ["Could you provide more details about your design requirements?"]
        
        missing = template.get_missing_requirements(conversation_state.gathered_requirements)
        
        if not missing:
            return []  # All requirements gathered
        
        # Prioritize critical requirements first
        critical = [f for f in missing if f.priority == RequirementPriority.CRITICAL]
        important = [f for f in missing if f.priority == RequirementPriority.IMPORTANT]
        optional = [f for f in missing if f.priority == RequirementPriority.OPTIONAL]
        
        # Select questions to ask
        questions_to_ask = []
        
        # Always ask critical first
        for field in critical[:max_questions]:
            question = field.question
            if field.unit:
                question += f" ({field.unit})"
            if field.choices:
                question += f" Options: {', '.join(field.choices)}"
            if field.help_text:
                question += f" ({field.help_text})"
            questions_to_ask.append(question)
        
        # Fill remaining slots with important
        remaining = max_questions - len(questions_to_ask)
        for field in important[:remaining]:
            question = field.question
            if field.unit:
                question += f" ({field.unit})"
            questions_to_ask.append(question)
        
        return questions_to_ask
    
    def extract_answers(self, 
                       user_response: str,
                       conversation_state: ConversationState) -> Dict[str, Any]:
        """Extract requirement values from user's response using LLM"""
        
        template = self.template_registry.get(conversation_state.design_type)
        if not template:
            return {}
        
        missing = template.get_missing_requirements(conversation_state.gathered_requirements)
        
        # Build extraction prompt
        fields_desc = "\n".join([
            f"- {f.key}: {f.question} (type: {f.data_type})"
            for f in missing
        ])
        
        prompt = f"""
        Extract requirement values from the user's response.
        
        Expected fields:
        {fields_desc}
        
        User response: "{user_response}"
        
        Return a JSON object with extracted values. Only include fields you can confidently extract.
        Use null for fields you cannot determine.
        """
        
        schema = {
            "type": "object",
            "properties": {
                field.key: {"type": "string"}  # We'll parse types later
                for field in missing
            }
        }
        
        try:
            extracted = self.agent.provider.generate_json(
                prompt=prompt,
                schema=schema,
                system_prompt="You are a requirement extractor. Be precise."
            )
            
            # Filter out null values and convert types
            result = {}
            for key, value in extracted.items():
                if value is not None and value != "null":
                    # Find the field to get its type
                    field = next((f for f in missing if f.key == key), None)
                    if field:
                        result[key] = self._convert_value(value, field.data_type)
            
            return result
        except Exception as e:
            logger.error(f"Failed to extract answers: {e}")
            return {}
    
    def _convert_value(self, value: Any, data_type: str) -> Any:
        """Convert extracted value to proper type"""
        if data_type == "number":
            try:
                return float(value)
            except:
                # Try to extract number from string (e.g. "250kmh", "80 kg")
                import re
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", str(value))
                if matches:
                    try:
                        return float(matches[0])
                    except:
                        return value
                return value
        elif data_type == "boolean":
            if isinstance(value, str):
                return value.lower() in ['true', 'yes', '1']
            return bool(value)
        return value
    
    def check_readiness(self, conversation_state: ConversationState) -> bool:
        """Check if we have enough information to proceed with planning"""
        if not conversation_state.design_type:
            logger.debug("Readiness Check: No design type")
            return False
        
        template = self.template_registry.get(conversation_state.design_type)
        if not template:
            logger.debug(f"Readiness Check: No template for {conversation_state.design_type}")
            return False
            
        critical = template.get_critical_fields()
        missing = [f.key for f in critical if f.key not in conversation_state.gathered_requirements]
        
        if missing:
            logger.info(f"Readiness Check: Missing critical fields: {missing}")
            # Also log what we HAVE for debugging
            logger.info(f"Have: {list(conversation_state.gathered_requirements.keys())}")
            return False
            
        return True
    
    def generate_summary(self, conversation_state: ConversationState) -> str:
        """Generate a summary of gathered requirements"""
        if not conversation_state.gathered_requirements:
            return "No requirements gathered yet."
        
        lines = ["Here's what I understand so far:"]
        for key, value in conversation_state.gathered_requirements.items():
            # Make key human-readable
            readable_key = key.replace('_', ' ').title()
            lines.append(f"- {readable_key}: {value}")
        
        return "\n".join(lines)
