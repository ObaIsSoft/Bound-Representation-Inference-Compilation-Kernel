"""
Production Nexus Agent - Knowledge Graph & Context Navigation

Features:
- Knowledge graph construction and traversal
- Design context management
- Component relationship mapping
- Semantic search across designs
- Version history tracking
- Cross-project intelligence
- Natural language query interface
- Graph visualization export
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import logging
import os
import hashlib
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities in knowledge graph."""
    PROJECT = "project"
    COMPONENT = "component"
    MATERIAL = "material"
    PROCESS = "process"
    REQUIREMENT = "requirement"
    TEST = "test"
    PERSON = "person"
    DOCUMENT = "document"
    DECISION = "decision"
    ISSUE = "issue"


class RelationType(Enum):
    """Types of relationships between entities."""
    CONTAINS = "contains"
    DEPENDS_ON = "depends_on"
    USES = "uses"
    REQUIRES = "requires"
    PRODUCES = "produces"
    VERIFIES = "verifies"
    AUTHORED_BY = "authored_by"
    REPLACES = "replaces"
    REFERENCES = "references"
    CONFLICTS_WITH = "conflicts_with"
    IMPLEMENTS = "implements"


@dataclass
class Entity:
    """Knowledge graph entity."""
    id: str
    type: EntityType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Relation:
    """Relationship between entities."""
    id: str
    source_id: str
    target_id: str
    type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class QueryResult:
    """Query result with context."""
    entities: List[Entity]
    relations: List[Relation]
    path: Optional[List[str]] = None
    confidence: float = 1.0


class NexusAgent:
    """
    Production-grade knowledge graph and context navigation agent.
    
    Manages:
    - Knowledge graph construction from designs
    - Entity-relationship tracking
    - Semantic search and querying
    - Design context navigation
    - Cross-project intelligence
    """
    
    def __init__(self, storage_path: str = "data/nexus_graph.json"):
        self.name = "NexusAgent"
        self.storage_path = Path(storage_path)
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.entity_index: Dict[EntityType, Set[str]] = defaultdict(set)
        self.relation_index: Dict[str, Set[str]] = defaultdict(set)  # source_id -> relation_ids
        self.reverse_relation_index: Dict[str, Set[str]] = defaultdict(set)  # target_id -> relation_ids
        
        # Load existing graph
        self._load_graph()
        
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Nexus operation.
        
        Args:
            params: {
                "action": str,  # query, add_entity, add_relation, traverse,
                               # search, visualize, export, import_design
                ... action-specific parameters
            }
        """
        action = params.get("action", "query")
        
        actions = {
            "query": self._action_query,
            "add_entity": self._action_add_entity,
            "add_relation": self._action_add_relation,
            "traverse": self._action_traverse,
            "search": self._action_search,
            "visualize": self._action_visualize,
            "export": self._action_export,
            "import_design": self._action_import_design,
            "get_entity": self._action_get_entity,
            "get_neighbors": self._action_get_neighbors,
            "find_path": self._action_find_path,
            "delete_entity": self._action_delete_entity,
            "get_stats": self._action_get_stats,
            "semantic_search": self._action_semantic_search,
        }
        
        if action not in actions:
            return {
                "status": "error",
                "message": f"Unknown action: {action}",
                "available_actions": list(actions.keys())
            }
        
        return actions[action](params)
    
    def _load_graph(self):
        """Load graph from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                # Load entities
                for e_data in data.get("entities", []):
                    entity = Entity(
                        id=e_data["id"],
                        type=EntityType(e_data["type"]),
                        name=e_data["name"],
                        properties=e_data.get("properties", {}),
                        created_at=datetime.fromisoformat(e_data["created_at"]),
                        updated_at=datetime.fromisoformat(e_data["updated_at"])
                    )
                    self.entities[entity.id] = entity
                    self.entity_index[entity.type].add(entity.id)
                
                # Load relations
                for r_data in data.get("relations", []):
                    relation = Relation(
                        id=r_data["id"],
                        source_id=r_data["source_id"],
                        target_id=r_data["target_id"],
                        type=RelationType(r_data["type"]),
                        properties=r_data.get("properties", {}),
                        created_at=datetime.fromisoformat(r_data["created_at"])
                    )
                    self.relations[relation.id] = relation
                    self.relation_index[relation.source_id].add(relation.id)
                    self.reverse_relation_index[relation.target_id].add(relation.id)
                
                logger.info(f"[NEXUS] Loaded {len(self.entities)} entities, {len(self.relations)} relations")
            except Exception as e:
                logger.error(f"[NEXUS] Failed to load graph: {e}")
    
    def _save_graph(self):
        """Save graph to storage."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "entities": [
                    {
                        "id": e.id,
                        "type": e.type.value,
                        "name": e.name,
                        "properties": e.properties,
                        "created_at": e.created_at.isoformat(),
                        "updated_at": e.updated_at.isoformat()
                    }
                    for e in self.entities.values()
                ],
                "relations": [
                    {
                        "id": r.id,
                        "source_id": r.source_id,
                        "target_id": r.target_id,
                        "type": r.type.value,
                        "properties": r.properties,
                        "created_at": r.created_at.isoformat()
                    }
                    for r in self.relations.values()
                ]
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"[NEXUS] Saved graph to {self.storage_path}")
        except Exception as e:
            logger.error(f"[NEXUS] Failed to save graph: {e}")
    
    def _action_add_entity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add entity to knowledge graph."""
        entity_type_str = params.get("entity_type", "component")
        name = params.get("name")
        properties = params.get("properties", {})
        entity_id = params.get("id") or self._generate_id(name or entity_type_str)
        
        try:
            entity_type = EntityType(entity_type_str.lower())
        except ValueError:
            return {
                "status": "error",
                "message": f"Invalid entity type: {entity_type_str}",
                "valid_types": [t.value for t in EntityType]
            }
        
        entity = Entity(
            id=entity_id,
            type=entity_type,
            name=name or f"{entity_type.value}_{len(self.entities)}",
            properties=properties
        )
        
        self.entities[entity_id] = entity
        self.entity_index[entity_type].add(entity_id)
        
        self._save_graph()
        
        return {
            "status": "success",
            "entity_id": entity_id,
            "type": entity_type.value,
            "name": entity.name
        }
    
    def _action_add_relation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add relationship between entities."""
        source_id = params.get("source_id")
        target_id = params.get("target_id")
        relation_type_str = params.get("relation_type", "depends_on")
        properties = params.get("properties", {})
        
        if not source_id or not target_id:
            return {"status": "error", "message": "source_id and target_id required"}
        
        if source_id not in self.entities:
            return {"status": "error", "message": f"Source entity not found: {source_id}"}
        if target_id not in self.entities:
            return {"status": "error", "message": f"Target entity not found: {target_id}"}
        
        try:
            relation_type = RelationType(relation_type_str.lower())
        except ValueError:
            return {
                "status": "error",
                "message": f"Invalid relation type: {relation_type_str}",
                "valid_types": [t.value for t in RelationType]
            }
        
        relation_id = f"{source_id}_{relation_type.value}_{target_id}_{len(self.relations)}"
        
        relation = Relation(
            id=relation_id,
            source_id=source_id,
            target_id=target_id,
            type=relation_type,
            properties=properties
        )
        
        self.relations[relation_id] = relation
        self.relation_index[source_id].add(relation_id)
        self.reverse_relation_index[target_id].add(relation_id)
        
        self._save_graph()
        
        return {
            "status": "success",
            "relation_id": relation_id,
            "type": relation_type.value,
            "source": source_id,
            "target": target_id
        }
    
    def _action_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Query knowledge graph."""
        entity_types = params.get("entity_types", [])
        relation_types = params.get("relation_types", [])
        properties = params.get("properties", {})
        
        # Filter entities
        results = []
        for entity in self.entities.values():
            # Check type filter
            if entity_types and entity.type.value not in entity_types:
                continue
            
            # Check property filters
            match = True
            for key, value in properties.items():
                if entity.properties.get(key) != value:
                    match = False
                    break
            
            if match:
                results.append(entity)
        
        # Get related relations if requested
        related_relations = []
        if relation_types:
            for entity in results:
                for rel_id in self.relation_index.get(entity.id, []):
                    rel = self.relations.get(rel_id)
                    if rel and rel.type.value in relation_types:
                        related_relations.append(rel)
        
        return {
            "status": "success",
            "entity_count": len(results),
            "relation_count": len(related_relations),
            "entities": [
                {"id": e.id, "type": e.type.value, "name": e.name, "properties": e.properties}
                for e in results[:100]  # Limit results
            ],
            "relations": [
                {"id": r.id, "type": r.type.value, "source": r.source_id, "target": r.target_id}
                for r in related_relations[:100]
            ]
        }
    
    def _action_traverse(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Traverse graph from starting entity."""
        start_id = params.get("start_id")
        depth = params.get("depth", 2)
        relation_types = params.get("relation_types", [])
        
        if not start_id or start_id not in self.entities:
            return {"status": "error", "message": f"Start entity not found: {start_id}"}
        
        visited = {start_id}
        current_level = {start_id}
        all_entities = {start_id: self.entities[start_id]}
        all_relations = {}
        
        for d in range(depth):
            next_level = set()
            for entity_id in current_level:
                # Get outgoing relations
                for rel_id in self.relation_index.get(entity_id, []):
                    rel = self.relations.get(rel_id)
                    if rel:
                        if not relation_types or rel.type.value in relation_types:
                            all_relations[rel_id] = rel
                            if rel.target_id not in visited:
                                next_level.add(rel.target_id)
                                all_entities[rel.target_id] = self.entities.get(rel.target_id)
                                visited.add(rel.target_id)
                
                # Get incoming relations
                for rel_id in self.reverse_relation_index.get(entity_id, []):
                    rel = self.relations.get(rel_id)
                    if rel:
                        if not relation_types or rel.type.value in relation_types:
                            all_relations[rel_id] = rel
                            if rel.source_id not in visited:
                                next_level.add(rel.source_id)
                                all_entities[rel.source_id] = self.entities.get(rel.source_id)
                                visited.add(rel.source_id)
            
            current_level = next_level
            if not current_level:
                break
        
        return {
            "status": "success",
            "start_id": start_id,
            "depth": depth,
            "entities_found": len(all_entities),
            "relations_found": len(all_relations),
            "entities": [
                {"id": e.id, "type": e.type.value, "name": e.name}
                for e in all_entities.values() if e
            ],
            "relations": [
                {"id": r.id, "type": r.type.value, "source": r.source_id, "target": r.target_id}
                for r in all_relations.values()
            ]
        }
    
    def _action_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search entities by name or property."""
        query = params.get("query", "").lower()
        entity_types = params.get("entity_types", [])
        
        if not query:
            return {"status": "error", "message": "Query string required"}
        
        results = []
        for entity in self.entities.values():
            # Check type filter
            if entity_types and entity.type.value not in entity_types:
                continue
            
            # Search in name
            score = 0
            if query in entity.name.lower():
                score = 100 if entity.name.lower() == query else 80
            
            # Search in properties
            for key, value in entity.properties.items():
                if isinstance(value, str) and query in value.lower():
                    score = max(score, 50)
            
            if score > 0:
                results.append({
                    "entity": {"id": entity.id, "type": entity.type.value, "name": entity.name},
                    "score": score
                })
        
        # Sort by score
        results.sort(key=lambda x: -x["score"])
        
        return {
            "status": "success",
            "query": query,
            "results_count": len(results),
            "results": results[:50]
        }
    
    def _action_semantic_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Semantic search using LLM embeddings (placeholder)."""
        query = params.get("query", "")
        
        # This would use vector embeddings in a full implementation
        # For now, fall back to regular search
        return self._action_search(params)
    
    def _action_visualize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Export graph for visualization."""
        format_type = params.get("format", "cytoscape")  # cytoscape, d3, neo4j
        entity_types = params.get("entity_types", [])
        
        # Filter entities
        entities_to_export = []
        for entity in self.entities.values():
            if not entity_types or entity.type.value in entity_types:
                entities_to_export.append(entity)
        
        entity_ids = {e.id for e in entities_to_export}
        
        # Filter relations
        relations_to_export = []
        for relation in self.relations.values():
            if relation.source_id in entity_ids and relation.target_id in entity_ids:
                relations_to_export.append(relation)
        
        if format_type == "cytoscape":
            elements = []
            
            # Add nodes
            for entity in entities_to_export:
                elements.append({
                    "data": {
                        "id": entity.id,
                        "label": entity.name,
                        "type": entity.type.value
                    }
                })
            
            # Add edges
            for relation in relations_to_export:
                elements.append({
                    "data": {
                        "id": relation.id,
                        "source": relation.source_id,
                        "target": relation.target_id,
                        "label": relation.type.value
                    }
                })
            
            return {
                "status": "success",
                "format": "cytoscape",
                "elements": elements
            }
        
        elif format_type == "d3":
            nodes = [{"id": e.id, "name": e.name, "group": e.type.value} for e in entities_to_export]
            links = [{"source": r.source_id, "target": r.target_id, "type": r.type.value} 
                    for r in relations_to_export]
            
            return {
                "status": "success",
                "format": "d3",
                "nodes": nodes,
                "links": links
            }
        
        elif format_type == "neo4j":
            cypher_statements = []
            
            # Create nodes
            for entity in entities_to_export:
                props = ", ".join([f"{k}: '{v}'" for k, v in entity.properties.items() if isinstance(v, str)])
                cypher_statements.append(
                    f"CREATE ({entity.id}:{entity.type.value} {{name: '{entity.name}', {props}}})"
                )
            
            # Create relationships
            for relation in relations_to_export:
                cypher_statements.append(
                    f"CREATE ({relation.source_id})-[:{relation.type.value}]->({relation.target_id})"
                )
            
            return {
                "status": "success",
                "format": "neo4j",
                "cypher": cypher_statements
            }
        
        return {"status": "error", "message": f"Unknown format: {format_type}"}
    
    def _action_export(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Export graph to file."""
        format_type = params.get("format", "json")
        filepath = params.get("filepath", "nexus_export.json")
        
        if format_type == "json":
            data = {
                "entities": [
                    {"id": e.id, "type": e.type.value, "name": e.name, "properties": e.properties}
                    for e in self.entities.values()
                ],
                "relations": [
                    {"id": r.id, "type": r.type.value, "source": r.source_id, "target": r.target_id}
                    for r in self.relations.values()
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            return {"status": "success", "format": "json", "filepath": filepath}
        
        return {"status": "error", "message": f"Export format not supported: {format_type}"}
    
    def _action_import_design(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Import design into knowledge graph."""
        design_data = params.get("design_data", {})
        project_name = params.get("project_name", "unnamed_project")
        
        # Create project entity
        project_id = self._generate_id(project_name)
        project = Entity(
            id=project_id,
            type=EntityType.PROJECT,
            name=project_name,
            properties={"imported_at": datetime.now().isoformat()}
        )
        self.entities[project_id] = project
        self.entity_index[EntityType.PROJECT].add(project_id)
        
        # Import components
        components = design_data.get("components", [])
        imported_count = 0
        
        for comp in components:
            comp_id = self._generate_id(comp.get("name", "component"))
            entity = Entity(
                id=comp_id,
                type=EntityType.COMPONENT,
                name=comp.get("name", f"Component {imported_count}"),
                properties=comp.get("properties", {})
            )
            self.entities[comp_id] = entity
            self.entity_index[EntityType.COMPONENT].add(comp_id)
            
            # Create relation to project
            self._create_relation(project_id, comp_id, RelationType.CONTAINS)
            imported_count += 1
        
        self._save_graph()
        
        return {
            "status": "success",
            "project_id": project_id,
            "components_imported": imported_count,
            "total_entities": len(self.entities)
        }
    
    def _action_get_entity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get single entity details."""
        entity_id = params.get("entity_id")
        
        if not entity_id or entity_id not in self.entities:
            return {"status": "error", "message": f"Entity not found: {entity_id}"}
        
        entity = self.entities[entity_id]
        
        # Get related entities
        outgoing = []
        for rel_id in self.relation_index.get(entity_id, []):
            rel = self.relations.get(rel_id)
            if rel:
                outgoing.append({
                    "relation": rel.type.value,
                    "target": {"id": rel.target_id, "name": self.entities.get(rel.target_id, Entity("", EntityType.COMPONENT, "")).name}
                })
        
        incoming = []
        for rel_id in self.reverse_relation_index.get(entity_id, []):
            rel = self.relations.get(rel_id)
            if rel:
                incoming.append({
                    "relation": rel.type.value,
                    "source": {"id": rel.source_id, "name": self.entities.get(rel.source_id, Entity("", EntityType.COMPONENT, "")).name}
                })
        
        return {
            "status": "success",
            "entity": {
                "id": entity.id,
                "type": entity.type.value,
                "name": entity.name,
                "properties": entity.properties,
                "created_at": entity.created_at.isoformat()
            },
            "relations": {
                "outgoing": outgoing,
                "incoming": incoming
            }
        }
    
    def _action_get_neighbors(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get neighbors of an entity."""
        entity_id = params.get("entity_id")
        relation_type = params.get("relation_type")
        
        if not entity_id or entity_id not in self.entities:
            return {"status": "error", "message": f"Entity not found: {entity_id}"}
        
        neighbors = []
        
        # Outgoing neighbors
        for rel_id in self.relation_index.get(entity_id, []):
            rel = self.relations.get(rel_id)
            if rel and (not relation_type or rel.type.value == relation_type):
                target = self.entities.get(rel.target_id)
                if target:
                    neighbors.append({
                        "direction": "outgoing",
                        "relation": rel.type.value,
                        "entity": {"id": target.id, "type": target.type.value, "name": target.name}
                    })
        
        # Incoming neighbors
        for rel_id in self.reverse_relation_index.get(entity_id, []):
            rel = self.relations.get(rel_id)
            if rel and (not relation_type or rel.type.value == relation_type):
                source = self.entities.get(rel.source_id)
                if source:
                    neighbors.append({
                        "direction": "incoming",
                        "relation": rel.type.value,
                        "entity": {"id": source.id, "type": source.type.value, "name": source.name}
                    })
        
        return {
            "status": "success",
            "entity_id": entity_id,
            "neighbor_count": len(neighbors),
            "neighbors": neighbors
        }
    
    def _action_find_path(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Find path between two entities (BFS)."""
        start_id = params.get("start_id")
        end_id = params.get("end_id")
        max_depth = params.get("max_depth", 5)
        
        if not start_id or not end_id:
            return {"status": "error", "message": "start_id and end_id required"}
        
        if start_id not in self.entities or end_id not in self.entities:
            return {"status": "error", "message": "Start or end entity not found"}
        
        # BFS
        queue = [(start_id, [start_id])]
        visited = {start_id}
        
        while queue:
            current, path = queue.pop(0)
            
            if current == end_id:
                return {
                    "status": "success",
                    "path_found": True,
                    "path_length": len(path) - 1,
                    "path": path,
                    "entities": [
                        {"id": eid, "name": self.entities[eid].name, "type": self.entities[eid].type.value}
                        for eid in path
                    ]
                }
            
            if len(path) >= max_depth:
                continue
            
            # Get neighbors
            for rel_id in self.relation_index.get(current, []):
                rel = self.relations.get(rel_id)
                if rel and rel.target_id not in visited:
                    visited.add(rel.target_id)
                    queue.append((rel.target_id, path + [rel.target_id]))
        
        return {
            "status": "success",
            "path_found": False,
            "message": f"No path found within {max_depth} hops"
        }
    
    def _action_delete_entity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete entity and its relations."""
        entity_id = params.get("entity_id")
        
        if not entity_id or entity_id not in self.entities:
            return {"status": "error", "message": f"Entity not found: {entity_id}"}
        
        entity = self.entities[entity_id]
        
        # Remove relations
        relation_ids = list(self.relation_index.get(entity_id, []))
        relation_ids.extend(self.reverse_relation_index.get(entity_id, []))
        
        for rel_id in relation_ids:
            if rel_id in self.relations:
                rel = self.relations[rel_id]
                del self.relations[rel_id]
                self.relation_index[rel.source_id].discard(rel_id)
                self.reverse_relation_index[rel.target_id].discard(rel_id)
        
        # Remove entity
        del self.entities[entity_id]
        self.entity_index[entity.type].discard(entity_id)
        
        self._save_graph()
        
        return {
            "status": "success",
            "entity_id": entity_id,
            "relations_removed": len(relation_ids)
        }
    
    def _action_get_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get graph statistics."""
        entity_counts = {t.value: len(ids) for t, ids in self.entity_index.items()}
        
        relation_counts = defaultdict(int)
        for rel in self.relations.values():
            relation_counts[rel.type.value] += 1
        
        # Calculate density
        n = len(self.entities)
        e = len(self.relations)
        density = e / (n * (n - 1)) if n > 1 else 0
        
        return {
            "status": "success",
            "entity_count": n,
            "relation_count": e,
            "entity_breakdown": entity_counts,
            "relation_breakdown": dict(relation_counts),
            "graph_density": round(density, 4),
            "storage_path": str(self.storage_path)
        }
    
    def _create_relation(self, source_id: str, target_id: str, rel_type: RelationType):
        """Helper to create relation."""
        relation_id = f"{source_id}_{rel_type.value}_{target_id}_{len(self.relations)}"
        relation = Relation(
            id=relation_id,
            source_id=source_id,
            target_id=target_id,
            type=rel_type
        )
        self.relations[relation_id] = relation
        self.relation_index[source_id].add(relation_id)
        self.reverse_relation_index[target_id].add(relation_id)
    
    def _generate_id(self, name: str) -> str:
        """Generate unique ID."""
        timestamp = datetime.now().isoformat()
        hash_input = f"{name}_{timestamp}_{len(self.entities)}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]


# API Integration
class NexusAPI:
    """FastAPI endpoints for knowledge graph."""
    
    @staticmethod
    def get_routes(agent: NexusAgent):
        """Get FastAPI routes."""
        from fastapi import APIRouter, HTTPException
        from pydantic import BaseModel, Field
        from typing import Dict, List, Optional
        
        router = APIRouter(prefix="/nexus", tags=["nexus"])
        
        class AddEntityRequest(BaseModel):
            entity_type: str
            name: str
            properties: Dict = Field(default_factory=dict)
            id: Optional[str] = None
        
        class AddRelationRequest(BaseModel):
            source_id: str
            target_id: str
            relation_type: str
            properties: Dict = Field(default_factory=dict)
        
        class QueryRequest(BaseModel):
            entity_types: List[str] = Field(default_factory=list)
            relation_types: List[str] = Field(default_factory=list)
            properties: Dict = Field(default_factory=dict)
        
        @router.post("/entity")
        async def add_entity(request: AddEntityRequest):
            """Add entity to graph."""
            result = agent.run({"action": "add_entity", **request.dict()})
            if result.get("status") == "error":
                raise HTTPException(status_code=400, detail=result.get("message"))
            return result
        
        @router.post("/relation")
        async def add_relation(request: AddRelationRequest):
            """Add relation between entities."""
            result = agent.run({"action": "add_relation", **request.dict()})
            if result.get("status") == "error":
                raise HTTPException(status_code=400, detail=result.get("message"))
            return result
        
        @router.post("/query")
        async def query_graph(request: QueryRequest):
            """Query knowledge graph."""
            return agent.run({"action": "query", **request.dict()})
        
        @router.get("/entity/{entity_id}")
        async def get_entity(entity_id: str):
            """Get entity details."""
            result = agent.run({"action": "get_entity", "entity_id": entity_id})
            if result.get("status") == "error":
                raise HTTPException(status_code=404, detail=result.get("message"))
            return result
        
        @router.get("/traverse/{start_id}")
        async def traverse(start_id: str, depth: int = 2):
            """Traverse graph from entity."""
            return agent.run({"action": "traverse", "start_id": start_id, "depth": depth})
        
        @router.get("/search")
        async def search(query: str, entity_types: Optional[List[str]] = None):
            """Search entities."""
            return agent.run({"action": "search", "query": query, "entity_types": entity_types or []})
        
        @router.get("/stats")
        async def get_stats():
            """Get graph statistics."""
            return agent.run({"action": "get_stats"})
        
        @router.get("/types")
        async def list_types():
            """List entity and relation types."""
            return {
                "entity_types": [t.value for t in EntityType],
                "relation_types": [t.value for t in RelationType]
            }
        
        return router
