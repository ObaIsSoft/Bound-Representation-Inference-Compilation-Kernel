"""
Conversation State Management for Multi-Turn Requirement Gathering

Tracks conversation history, gathered requirements, and determines when
enough information has been collected to proceed with design planning.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class Message:
    """Single message in conversation"""
    role: str  # 'user' or 'agent'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationState:
    """State of an ongoing conversation"""
    conversation_id: str
    title: str = "New Conversation"
    branch_name: str = "Main"
    parent_id: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    design_type: Optional[str] = None  # aerial, ground, marine, space, robotics, etc.
    gathered_requirements: Dict[str, Any] = field(default_factory=dict)
    missing_requirements: List[str] = field(default_factory=list)
    ready_for_planning: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    language: str = "en"  # Multi-language support
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add a message to conversation history"""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        self.updated_at = datetime.now()
    
    def update_requirement(self, key: str, value: Any):
        """Update a gathered requirement"""
        self.gathered_requirements[key] = value
        if key in self.missing_requirements:
            self.missing_requirements.remove(key)
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary for storage"""
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "branch_name": self.branch_name,
            "parent_id": self.parent_id,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "metadata": m.metadata
                }
                for m in self.messages
            ],
            "design_type": self.design_type,
            "gathered_requirements": self.gathered_requirements,
            "missing_requirements": self.missing_requirements,
            "ready_for_planning": self.ready_for_planning,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "language": self.language
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationState':
        """Deserialize from dictionary"""
        state = cls(
            conversation_id=data["conversation_id"],
            title=data.get("title", "New Conversation"),
            branch_name=data.get("branch_name", "Main"),
            parent_id=data.get("parent_id")
        )
        state.design_type = data.get("design_type")
        state.gathered_requirements = data.get("gathered_requirements", {})
        state.missing_requirements = data.get("missing_requirements", [])
        state.ready_for_planning = data.get("ready_for_planning", False)
        state.language = data.get("language", "en")
        
        # Restore messages
        for msg_data in data.get("messages", []):
            state.messages.append(Message(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                metadata=msg_data.get("metadata", {})
            ))
        
        return state


class ConversationManager:
    """Manages multiple conversation states"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.conversations: Dict[str, ConversationState] = {}
        self.storage_path = storage_path
        if storage_path:
            self.load_from_file(storage_path)
    
    def create_conversation(self, conversation_id: str, title: str = "New Conversation", language: str = "en") -> ConversationState:
        """Create a new conversation"""
        state = ConversationState(conversation_id=conversation_id, title=title, language=language)
        self.conversations[conversation_id] = state
        self._auto_save()
        return state
    
    def branch_session(self, parent_id: str) -> Optional[ConversationState]:
        """Clone a session into a new branch"""
        parent = self.conversations.get(parent_id)
        if not parent:
            return None
        
        branch_id = f"session-branch-{int(datetime.now().timestamp())}"
        branch = ConversationState(
            conversation_id=branch_id,
            title=f"Branch: {parent.title}",
            branch_name=parent.branch_name,
            parent_id=parent_id,
            messages=[Message(m.role, m.content, m.timestamp, m.metadata) for m in parent.messages],
            design_type=parent.design_type,
            gathered_requirements=dict(parent.gathered_requirements),
            missing_requirements=list(parent.missing_requirements),
            language=parent.language
        )
        self.conversations[branch_id] = branch
        self._auto_save()
        return branch

    def merge_session(self, branch_id: str) -> Optional[ConversationState]:
        """Merge a branch back into its parent and delete the branch"""
        branch = self.conversations.get(branch_id)
        if not branch or not branch.parent_id:
            return None
        
        parent = self.conversations.get(branch.parent_id)
        if not parent:
            return None
        
        # Find new messages in branch (messages added after branching)
        new_messages = branch.messages[len(parent.messages):]
        parent.messages.extend(new_messages)
        parent.updated_at = datetime.now()
        
        # Clean up the branch
        del self.conversations[branch_id]
        self._auto_save()
        return parent

    def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """Get existing conversation"""
        return self.conversations.get(conversation_id)
    
    def get_or_create(self, conversation_id: str, language: str = "en") -> ConversationState:
        """Get existing or create new conversation"""
        if conversation_id in self.conversations:
            return self.conversations[conversation_id]
        return self.create_conversation(conversation_id, language=language)
    
    def delete_conversation(self, conversation_id: str):
        """Delete a conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            self._auto_save()
    
    def _auto_save(self):
        """Auto-save if storage path is set"""
        if self.storage_path:
            self.save_to_file(self.storage_path)

    def save_to_file(self, filepath: str):
        """Persist conversations to file"""
        data = {
            conv_id: state.to_dict()
            for conv_id, state in self.conversations.items()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load conversations from file"""
        try:
            if not os.path.exists(filepath):
                return
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.conversations = {
                conv_id: ConversationState.from_dict(conv_data)
                for conv_id, conv_data in data.items()
            }
        except Exception as e:
            print(f"Error loading conversations: {e}")


# Global conversation manager instance
import os
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "data")
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)
STORAGE_FILE = os.path.join(STORAGE_DIR, "conversations.json")

conversation_manager = ConversationManager(STORAGE_FILE)
