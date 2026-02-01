import logging
import numpy as np
import copy
from typing import List, Dict, Any, Optional
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from agents.evolution import GeometryGenome
# Reuse extraction logic or import it. Ideally we should have a shared FeatureExtractor.
# For now, we reimplement a lightweight version or import from PINN if possible.
# Let's verify if we can import MultiPhysicsPINN easily. Yes.
from agents.surrogate.pinn_model import MultiPhysicsPINN

logger = logging.getLogger(__name__)

class LatentSpaceAgent:
    """
    The Cartographer.
    Maps the 'Design Space' to a low-dimensional 'Latent Space' (Z-Space).
    Allows interpolation, clustering, and manifold traversal.
    """
    
    def __init__(self):
        self.name = "LatentSpaceAgent"
        self.pinn = MultiPhysicsPINN() # Used only for feature extraction util
        self.pca = PCA(n_components=3) # 3D Latent Space for easy viz
        self.knn = NearestNeighbors(n_neighbors=1)
        
        self.memory_genomes: List[Dict[str, Any]] = [] # Store full genome dicts
        self.memory_vectors: List[List[float]] = [] # Store raw feature vectors
        self.memory_z: List[List[float]] = [] # Store projected Z vectors
        self.is_fitted = False
        
    def learn_manifold(self, population: List[GeometryGenome]):
        """
        Fits the generic manifold to the provided population history.
        """
        # 1. Extract Features
        X = []
        self.memory_genomes = []
        
        for g in population:
            # Convert to dict for PINN extractor
            nodes_data = [attr['data'].dict() for attr in g.graph.nodes.values()]
            features = self.pinn._extract_features(nodes_data)
            X.append(features)
            self.memory_genomes.append(g.to_json())
            
        self.memory_vectors = np.array(X)
        
        # 2. Fit PCA
        if len(X) > 3:
            self.pca.fit(self.memory_vectors)
            self.memory_z = self.pca.transform(self.memory_vectors)
            self.knn.fit(self.memory_vectors) # KNN on high-dim or low-dim? High-dim is more precise for retrieval.
            self.is_fitted = True
            logger.info(f"[LATENT] Manifold Learned. Explained Variance: {self.pca.explained_variance_ratio_}")
        else:
            logger.warning("[LATENT] Not enough data to fit manifold.")
            
    def encode(self, genome: GeometryGenome) -> np.ndarray:
        """Projects a genome into Z-Space."""
        if not self.is_fitted: return np.zeros(3)
        nodes_data = [attr['data'].dict() for attr in genome.graph.nodes.values()]
        x = np.array([self.pinn._extract_features(nodes_data)])
        return self.pca.transform(x)[0]
        
    def decode(self, z_vector: List[float]) -> Optional[Dict[str, Any]]:
        """
        Inverse Mapping (Z -> Genome).
        Since PCA is not generative for Graphs, we use 'Generalized Retrieval'.
        We project Z back to X (reconstruction), then find Nearest Neighbor in memory.
        """
        if not self.is_fitted: return None
        
        # 1. Project back to Feature Space (Reconstruction)
        x_recon = self.pca.inverse_transform([z_vector])
        
        # 2. Find closest existing design
        return self.find_closest_design(x_recon[0])

    def find_closest_design(self, feature_vector: List[float]) -> Dict[str, Any]:
        """Returns the genome from memory that best matches the features."""
        # Find index
        distances, indices = self.knn.kneighbors([feature_vector])
        idx = indices[0][0]
        return self.memory_genomes[idx]

    def interpolate(self, genome_a: GeometryGenome, genome_b: GeometryGenome, steps: int = 5) -> List[Dict[str, Any]]:
        """
        Generates a sequence of designs morphing from A to B.
        Strategy:
        1. Parametric Morph (if topologies are compatible).
        2. Latent Walk (if topologies differ).
        """
        # Check Isomorphism (Mock check: same node count)
        # Real isomorphism is harder, but node count is a good heuristic for simple morphs
        if len(genome_a.graph.nodes) == len(genome_b.graph.nodes):
            return self._parametric_interpolate(genome_a, genome_b, steps)
        else:
            return self._latent_walk(genome_a, genome_b, steps)
            
    def _parametric_interpolate(self, g_a: GeometryGenome, g_b: GeometryGenome, steps: int) -> List[Dict[str, Any]]:
        """Linearly interpolates numerical parameters between two matching topologies."""
        results = []
        
        # Assume nodes have stable IDs or are ordered. 
        # For this MVP, we zip by order (risky but functional for cloned variations)
        nodes_a = list(g_a.graph.nodes.values())
        nodes_b = list(g_b.graph.nodes.values())
        
        for i in range(steps + 1):
            alpha = i / float(steps)
            
            # Create a shallow clone structure
            # In a real impl, we'd deep copy the genome object
            # Here we construct the JSON rep directly
            new_nodes = []
            
            for idx, node_attr_a in enumerate(nodes_a):
                node_a = node_attr_a['data']
                # Try to find corresponding node in B (by ID or Index)
                node_b = nodes_b[idx]['data'] if idx < len(nodes_b) else node_a
                
                # Interpolate Params
                new_params = {}
                for p_key, p_val_a in node_a.params.items():
                    val_a = p_val_a.value
                    val_b = node_b.params.get(p_key, p_val_a).value
                    new_val = val_a * (1 - alpha) + val_b * alpha
                    
                    # Create param dict structure
                    new_params[p_key] = {
                        "name": p_val_a.name, 
                        "value": new_val,
                        "min_val": p_val_a.min_val,
                        "max_val": p_val_a.max_val
                    }
                    
                # Interpolate Transform
                t_a = np.array(node_a.transform)
                t_b = np.array(node_b.transform)
                t_new = (t_a * (1 - alpha) + t_b * alpha).tolist()
                
                new_nodes.append({
                    "id": node_a.id,
                    "type": node_a.type.value,
                    "params": new_params,
                    "transform": t_new
                })
                
            results.append({"nodes": new_nodes, "d DNA_mix": alpha})
            
        return results

    def _latent_walk(self, g_a: GeometryGenome, g_b: GeometryGenome, steps: int) -> List[Dict[str, Any]]:
        """Walks the Z-space manifold to find intermediate existing designs."""
        if not self.is_fitted: return [g_a.to_json(), g_b.to_json()]
        
        z_a = self.encode(g_a)
        z_b = self.encode(g_b)
        
        path = []
        for i in range(steps + 1):
            alpha = i / float(steps)
            z_interp = z_a * (1 - alpha) + z_b * alpha
            
            # Find closest existing design state (Projected)
            design = self.decode(z_interp)
            if design:
                path.append(design)
                
        return path
