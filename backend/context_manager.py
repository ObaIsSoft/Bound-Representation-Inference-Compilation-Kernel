from typing import List, Dict, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import time
import asyncio
import hashlib
from collections import deque
import numpy as np

# Optional: For production
try:
    import aiosqlite
    import openai  # For summarization
    HAS_ASYNC = True
except ImportError:
    HAS_ASYNC = False


class ContextScope(Enum):
    """Hierarchical isolation levels"""
    EPHEMERAL = auto()      # This turn only (tool outputs)
    SCENE = auto()          # This design session (hours)
    CAMPAIGN = auto()       # This project (days/weeks)
    AGENT_IDENTITY = auto() # Persistent agent personality
    UNIVERSAL = auto()      # Shared physics knowledge


@dataclass
class MemoryFragment:
    """Atomic unit of context with rich metadata"""
    content: str
    role: str  # "user", "assistant", "system", "tool", "physics"
    scope: ContextScope
    agent_id: Optional[str] = None  # Which agent created this
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0  # Pre-computed
    embedding: Optional[List[float]] = None  # For semantic retrieval
    tags: Set[str] = field(default_factory=set)  # "thermal", "critical", "error"
    references: Dict[str, Any] = field(default_factory=dict)
    priority: float = 1.0  # For eviction (higher = keep longer)
    
    def __post_init__(self):
        if self.token_count == 0:
            # Rough estimate: 4 chars ~ 1 token
            self.token_count = len(self.content) // 4


class HierarchicalSummarizer:
    """
    Multi-level summarization: 
    - L0: Raw messages (recent)
    - L1: Scene summaries (hourly)
    - L2: Campaign summaries (daily)
    - L3: Agent identity (persistent)
    """
    
    def __init__(self, llm_callback: Optional[Callable] = None):
        self.llm = llm_callback or self._default_summarizer
        self.levels = {
            0: deque(maxlen=50),   # Raw fragments
            1: deque(maxlen=20),   # Scene summaries
            2: deque(maxlen=10),   # Campaign summaries  
            3: None                # Identity (never evicted)
        }
    
    async def summarize_batch(self, fragments: List[MemoryFragment], level: int) -> str:
        """Compress N fragments into 1 summary"""
        if not fragments:
            return ""
        
        # Build prompt
        content = "\n".join([f"[{f.role}]: {f.content[:500]}" for f in fragments])
        prompt = f"""Summarize the following conversation for long-term memory.
Preserve: critical decisions, numerical values, errors, action items.
Discard: pleasantries, repetition, procedural details.

{content}

SUMMARY:"""

        # Async LLM call
        summary = await self.llm(prompt)
        return summary.strip()
    
    async def compress(self, current_level: int) -> Optional[MemoryFragment]:
        """Promote level N to level N+1 summary"""
        if current_level >= 3:
            return None
            
        batch = list(self.levels[current_level])
        if len(batch) < 5:  # Not enough to compress
            return None
        
        summary_text = await self.summarize_batch(batch, current_level)
        
        # Create summary fragment at next level
        summary_fragment = MemoryFragment(
            content=summary_text,
            role="system",
            scope=self._scope_for_level(current_level + 1),
            tags={"summary", f"L{current_level+1}"}
        )
        
        # Clear compressed level
        self.levels[current_level].clear()
        self.levels[current_level + 1].append(summary_fragment)
        
        return summary_fragment
    
    def _scope_for_level(self, level: int) -> ContextScope:
        mapping = {0: ContextScope.EPHEMERAL, 1: ContextScope.SCENE,
                   2: ContextScope.CAMPAIGN, 3: ContextScope.AGENT_IDENTITY}
        return mapping.get(level, ContextScope.UNIVERSAL)
    
    async def _default_summarizer(self, prompt: str) -> str:
        """Fallback: extract key sentences"""
        sentences = prompt.split(". ")
        key_sentences = [s for s in sentences if any(kw in s.lower() 
                        for kw in ["decided", "error", "value", "must", "critical"])]
        return ". ".join(key_sentences[:3]) if key_sentences else sentences[0]


class VectorMemoryIndex:
    """
    Semantic retrieval for long-term context.
    Simple implementation; replace with Chroma/Pinecone in production.
    """
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.fragments: List[MemoryFragment] = []
        # Random projection for fast similarity (LSH-style)
        self.projection = np.random.randn(dim, dim) * 0.1
    
    def _embed(self, text: str) -> np.ndarray:
        """Simple sentence embedding (replace with real model)"""
        # Production: use sentence-transformers or OpenAI embedding
        hash_vec = np.array([ord(c) for c in text[:self.dim].ljust(self.dim)])
        return hash_vec / (np.linalg.norm(hash_vec) + 1e-8)
    
    def add(self, fragment: MemoryFragment):
        fragment.embedding = self._embed(fragment.content).tolist()
        self.fragments.append(fragment)
    
    def search(self, query: str, k: int = 5, tag_filter: Optional[Set[str]] = None) -> List[MemoryFragment]:
        """Find relevant fragments by semantic similarity"""
        query_vec = self._embed(query)
        
        candidates = self.fragments
        if tag_filter:
            candidates = [f for f in candidates if tag_filter & f.tags]
        
        # Cosine similarity
        scores = []
        for f in candidates:
            if f.embedding:
                vec = np.array(f.embedding)
                sim = np.dot(query_vec, vec)
                scores.append((sim, f))
        
        scores.sort(reverse=True)
        return [f for _, f in scores[:k]]


class EnhancedContextManager:
    """
    Production-grade context for BRICK OS agent swarms.
    Features: hierarchical memory, semantic retrieval, async, persistence.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        max_tokens: int = 8000,
        enable_vector_search: bool = True,
        db_path: Optional[str] = None,
        llm_summarizer: Optional[Callable] = None
    ):
        self.agent_id = agent_id or "orchestrator"
        self.max_tokens = max_tokens
        
        # Hierarchical memory
        self.summarizer = HierarchicalSummarizer(llm_summarizer)
        self.working_memory: deque[MemoryFragment] = deque(maxlen=20)  # L0
        
        # Long-term retrieval
        self.vector_index = VectorMemoryIndex() if enable_vector_search else None
        
        # Structured knowledge (like your original, but typed)
        self.plan: Dict[str, Any] = {}
        self.constraints: Dict[str, Any] = {}  # Hardware ISA constraints
        self.references: Dict[str, Any] = {}
        
        # Persistence
        self.db_path = db_path
        self._pending_saves: List[MemoryFragment] = []
        
        # Metrics
        self.access_stats = {"retrievals": 0, "summarizations": 0}
    
    async def add_message(
        self,
        role: str,
        content: str,
        scope: ContextScope = ContextScope.SCENE,
        metadata: Optional[Dict] = None,
        tags: Optional[Set[str]] = None,
        priority: float = 1.0
    ) -> MemoryFragment:
        """Add with full metadata and async processing"""
        
        fragment = MemoryFragment(
            content=content,
            role=role,
            scope=scope,
            agent_id=self.agent_id,
            metadata=metadata or {},
            tags=tags or set(),
            priority=priority
        )
        
        # Add to working memory
        self.working_memory.append(fragment)
        
        # Index for retrieval
        if self.vector_index:
            self.vector_index.add(fragment)
        
        # Queue for persistence
        self._pending_saves.append(fragment)
        
        # Trigger compression if needed
        await self._maintain_memory()
        
        return fragment
    
    async def _maintain_memory(self):
        """Background maintenance: compress, save, index"""
        total_tokens = sum(f.token_count for f in self.working_memory)
        
        if total_tokens > self.max_tokens * 0.8:
            # Compress L0 -> L1
            summary = await self.summarizer.compress(0)
            if summary:
                self.access_stats["summarizations"] += 1
                print(f"[Context] Compressed {len(self.summarizer.levels[0])} fragments to L1 summary")
        
        # Periodic persistence
        if len(self._pending_saves) >= 5 and self.db_path:
            await self._checkpoint()
    
    async def _checkpoint(self):
        """Save to SQLite/Postgres"""
        if not HAS_ASYNC or not self.db_path:
            # TODO: IMPLEMENT PERSISTENCE
            # 1. Update main.py to pass a 'session_id' or 'project_id' to this manager.
            # 2. Hook up a real Database (SQLite/Postgres) here to persist L1/L2 summaries between restarts.
            # 3. Currently, this manager is re-hydrated from frontend history on every turn.
            return
        
        async with aiosqlite.connect(self.db_path) as db:
            for fragment in self._pending_saves:
                await db.execute("""
                    INSERT INTO memory_fragments 
                    (agent_id, content, role, scope, timestamp, tags, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    fragment.agent_id,
                    fragment.content,
                    fragment.role,
                    fragment.scope.name,
                    fragment.timestamp,
                    json.dumps(list(fragment.tags)),
                    json.dumps(fragment.embedding) if fragment.embedding else None
                ))
            await db.commit()
        
        self._pending_saves.clear()
    
    def retrieve_relevant(
        self,
        query: str,
        k: int = 3,
        scope_filter: Optional[ContextScope] = None,
        tag_filter: Optional[Set[str]] = None
    ) -> List[MemoryFragment]:
        """Semantic search across long-term memory"""
        if not self.vector_index:
            return []
        
        results = self.vector_index.search(query, k=k, tag_filter=tag_filter)
        
        if scope_filter:
            results = [r for r in results if r.scope == scope_filter]
        
        self.access_stats["retrievals"] += 1
        return results
    
    def build_prompt_context(
        self,
        include_plan: bool = True,
        include_summaries: bool = True,
        retrieval_query: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Construct LLM prompt with intelligent token budgeting.
        Priority: Plan > Critical tags > Recent > Retrieved > Summaries
        """
        max_t = max_tokens or self.max_tokens
        parts = []
        current_tokens = 0
        
        def add_section(header: str, content: str, priority: float = 1.0) -> bool:
            nonlocal current_tokens
            tokens = len(content) // 4 + len(header) // 4
            
            if current_tokens + tokens > max_t * priority:
                return False
            
            parts.append(f"\n--- {header} ---\n{content}\n")
            current_tokens += tokens
            return True
        
        # 1. Plan (highest priority)
        if include_plan and self.plan:
            plan_str = json.dumps(self.plan, indent=2)
            add_section("ACTIVE MISSION", plan_str, priority=0.9)
        
        # 2. Critical constraints (safety, physics)
        critical = [f for f in self.working_memory if "critical" in f.tags or "error" in f.tags]
        if critical:
            crit_str = "\n".join([f"[{f.role}]: {f.content[:200]}" for f in critical[-3:]])
            add_section("CRITICAL ITEMS", crit_str, priority=0.85)
        
        # 3. Retrieved relevant context
        if retrieval_query:
            retrieved = self.retrieve_relevant(retrieval_query, k=3)
            if retrieved:
                ret_str = "\n".join([f"[RELEVANT] {f.content[:300]}" for f in retrieved])
                add_section("RELEVANT HISTORY", ret_str, priority=0.7)
        
        # 4. Recent working memory
        recent = list(self.working_memory)[-10:]
        recent_str = "\n".join([f"[{f.role.upper()}]: {f.content[:400]}" for f in recent])
        add_section("RECENT MESSAGES", recent_str, priority=0.6)
        
        # 5. Summaries (if space permits)
        if include_summaries:
            for level in [1, 2]:
                summaries = list(self.summarizer.levels[level])[-2:]
                for s in summaries:
                    add_section(f"SUMMARY L{level}", s.content[:500], priority=0.3)
        
        return "\n".join(parts)
    
    def get_cross_agent_context(self, other_agents: List["EnhancedContextManager"]) -> str:
        """
        For NexusAgent: Synthesize context from multiple agents.
        Critical for swarm coordination.
        """
        consensus = {}
        conflicts = []
        
        for agent in other_agents:
            # Extract key decisions from each agent
            for fragment in agent.working_memory:
                if "decision" in fragment.tags:
                    key = hashlib.md5(fragment.content[:100].encode()).hexdigest()[:8]
                    if key in consensus:
                        if consensus[key]["content"] != fragment.content:
                            conflicts.append((self.agent_id, agent.agent_id, fragment.content))
                    else:
                        consensus[key] = {
                            "agent": agent.agent_id,
                            "content": fragment.content,
                            "timestamp": fragment.timestamp
                        }
        
        # Build synthesized context
        result = ["--- CROSS-AGENT CONSENSUS ---"]
        for item in consensus.values():
            result.append(f"[{item['agent']}]: {item['content'][:100]}")
        
        if conflicts:
            result.append("\n--- CONFLICTS DETECTED ---")
            for a1, a2, content in conflicts[:3]:
                result.append(f"{a1} vs {a2}: {content[:80]}...")
        
        return "\n".join(result)


# Convenience factory for BRICK OS
def create_agent_context(agent_type: str, agent_id: str, db_path: Optional[str] = None):
    """Factory for specific agent roles"""
    
    configs = {
        "physics": {"max_tokens": 12000, "tags": {"physics", "critical"}},
        "geometry": {"max_tokens": 8000, "tags": {"sdf", "manufacturing"}},
        "manufacturing": {"max_tokens": 6000, "tags": {"process", "cost"}},
        "orchestrator": {"max_tokens": 16000, "tags": {"coordination", "plan"}}
    }
    
    config = configs.get(agent_type, configs["orchestrator"])
    
    return EnhancedContextManager(
        agent_id=agent_id,
        max_tokens=config["max_tokens"],
        enable_vector_search=True,
        db_path=db_path
    )
