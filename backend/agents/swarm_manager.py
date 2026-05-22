"""
SwarmManager — Von Neumann Distributed Simulation Engine

Runs a population of self-replicating agents (VonNeumannAgent) in a shared
resource environment. Agents harvest energy, process compute tasks, replicate
with genetic mutation, and leave pheromone trails (stigmergy).

Primary use: parallel design space exploration — multiple probes explore a
geometry/parameter landscape concurrently, competing for compute budget,
replicating the most fit configurations.
"""

from typing import Dict, Any, List, Optional
import logging
import uuid
import math
import random
import numpy as np

logger = logging.getLogger(__name__)


class SwarmManager:
    """
    Orchestrates the Von Neumann Swarm Simulation Loop.
    Manages agent registry, environment state, resource field, and time-stepping.
    """

    def __init__(self):
        self.agents: List[Any] = []
        self.resources: List[Dict] = []
        self.pheromones: Dict[str, float] = {}
        self.task_queue: List[Dict] = []
        self.construction_targets: List[Dict] = []
        self.tick_count = 0
        self.gravity = 9.81

    # ------------------------------------------------------------------
    # Public API — pipeline entry point
    # ------------------------------------------------------------------

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standard pipeline entry point.

        Accepts:
            ticks           int   — simulation steps (default 50)
            initial_pop     int   — seed agent count (default 4)
            task_count      int   — compute tasks in queue (default 10)
            resource_count  int   — resource nodes in environment (default 20)
            arena_size      float — square arena side length (default 200.0)
            agent_types     list  — ["VonNeumannAgent"] (only type available)
            geometry_tree   list  — optional construction targets
            environment     dict  — optional {gravity: float}

        Returns full simulation metrics dict.
        """
        config = {
            "initial_pop": params.get("initial_pop", 4),
            "task_count": params.get("task_count", 10),
            "resource_count": params.get("resource_count", 20),
            "agent_types": params.get("agent_types", ["VonNeumannAgent"]),
        }
        ticks = params.get("ticks", 50)
        arena_size = params.get("arena_size", 200.0)
        environment = params.get("environment", {})
        geometry_tree = params.get("geometry_tree", [])

        self.init_simulation(config, environment, geometry_tree, arena_size)
        return self.run_simulation(ticks)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_simulation(
        self,
        config: Dict[str, Any],
        environment: Optional[Dict[str, Any]] = None,
        geometry_tree: Optional[List[Dict[str, Any]]] = None,
        arena_size: float = 200.0,
    ):
        """Initialise swarm with seed agents, resources, and tasks."""
        from backend.physics.kernel import get_physics_kernel
        kernel = get_physics_kernel()
        self.gravity = (environment or {}).get("gravity", kernel.get_constant("g"))
        self.construction_targets = geometry_tree or []
        self.tick_count = 0

        # Task queue
        self.task_queue = [
            {"id": f"task_{i}", "type": "COMPUTE_HASH", "effort": 10.0, "status": "pending"}
            for i in range(config.get("task_count", 10))
        ]

        # Resource field — randomly scattered nodes in the arena
        count = config.get("resource_count", 20)
        self.resources = self._init_resources(arena_size, count)
        self.pheromones = {}

        # Seed population
        agent_names = config.get("agent_types", ["VonNeumannAgent"])
        pop = config.get("initial_pop", 4)

        from backend.agents.von_neumann_agent import VonNeumannAgent
        self.agents = []
        for i in range(pop):
            ag_type = agent_names[i % len(agent_names)]
            if ag_type == "VonNeumannAgent":
                agent = VonNeumannAgent(initial_energy=200.0)
                # Spread starting positions
                agent.pos = [
                    random.uniform(-arena_size / 4, arena_size / 4),
                    random.uniform(-arena_size / 4, arena_size / 4),
                ]
                self.agents.append(agent)
            else:
                logger.warning(f"Unknown agent type '{ag_type}' — skipping")

        logger.info(
            f"Swarm initialised: {len(self.agents)} agents, "
            f"{len(self.task_queue)} tasks, {len(self.resources)} resources"
        )

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def run_simulation(self, ticks: int = 50) -> Dict[str, Any]:
        """Run simulation for N ticks and return final metrics."""
        tick_logs: List[str] = []
        for _ in range(ticks):
            log_entry = self.run_tick()
            if log_entry:
                tick_logs.append(log_entry)
            if not self.agents:
                logger.info("Swarm extinct — stopping early")
                break

        metrics = self._capture_metrics()
        metrics["tick_log_sample"] = tick_logs[-10:]  # last 10 notable events
        return metrics

    def run_tick(self) -> Optional[str]:
        """Execute one simulation step. Returns a notable log line or None."""
        self.tick_count += 1

        env_state = {
            "resources": self.resources,
            "pheromones": self.pheromones,
            "tasks": self.task_queue,
            "targets": self.construction_targets,
        }

        new_agents: List[Any] = []
        dead_ids: set = set()
        notable: List[str] = []

        for agent in self.agents:
            res = agent.run(env_state)

            # Task completion
            if completed_id := res.get("completed_task_id"):
                before = len(self.task_queue)
                self.task_queue = [t for t in self.task_queue if t["id"] != completed_id]
                if len(self.task_queue) < before:
                    agent.energy += 20.0

            # Resource harvest
            if req := res.get("harvest_request"):
                harvested = self._consume_resource(res["pos"], req["amount"])
                if harvested > 0 and hasattr(agent, "energy"):
                    agent.energy += harvested
                    self.pheromones[req["target_id"]] = (
                        self.pheromones.get(req["target_id"], 0.0) + 1.0
                    )

            # Replication
            if child_cfg := res.get("child"):
                from backend.agents.von_neumann_agent import VonNeumannAgent
                if child_cfg.get("type", "VonNeumannAgent") == "VonNeumannAgent":
                    baby = VonNeumannAgent(
                        genetics=child_cfg["genetics"],
                        initial_energy=child_cfg["energy_grant"],
                    )
                    baby.pos = list(res.get("pos", [0.0, 0.0]))
                    new_agents.append(baby)
                    notable.append(
                        f"tick={self.tick_count} {agent.id} replicated → gen {child_cfg['genetics'].get('generation', '?')}"
                    )

            # Death
            if res.get("status") == "dead":
                dead_ids.add(agent.id)

        # Commit state
        self.agents = [a for a in self.agents if a.id not in dead_ids] + new_agents

        # Pheromone decay (evaporation rate 5% per tick)
        self.pheromones = {k: v * 0.95 for k, v in self.pheromones.items() if v * 0.95 > 0.01}

        # Collision detection (numpy — no external kernel needed)
        collisions = self._detect_collisions()
        if collisions:
            logger.debug(f"tick={self.tick_count}: {len(collisions)} collisions")

        return notable[-1] if notable else None

    # ------------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------------

    def _init_resources(self, arena: float, count: int) -> List[Dict]:
        """Generate randomly placed resource nodes."""
        return [
            {
                "id": f"res_{uuid.uuid4().hex[:6]}",
                "x": random.uniform(-arena / 2, arena / 2),
                "y": random.uniform(-arena / 2, arena / 2),
                "amount": random.uniform(50.0, 200.0),
            }
            for _ in range(count)
        ]

    def _consume_resource(self, agent_pos: List[float], amount: float, radius: float = 10.0) -> float:
        """
        Consume `amount` from the nearest resource within `radius`.
        Returns amount actually harvested.
        """
        best = None
        best_dist = radius

        for res in self.resources:
            if res["amount"] <= 0:
                continue
            d = math.hypot(res["x"] - agent_pos[0], res["y"] - agent_pos[1])
            if d < best_dist:
                best_dist = d
                best = res

        if best is None:
            return 0.0

        harvested = min(amount, best["amount"])
        best["amount"] -= harvested
        return harvested

    def _detect_collisions(self, margin: float = 0.1) -> List[str]:
        """
        O(N²) pairwise collision detection using numpy.
        Returns list of overlapping agent-pair strings.
        """
        n = len(self.agents)
        if n < 2:
            return []

        positions = np.array([a.pos + [0.0] for a in self.agents])  # Nx3
        radii = np.array([
            getattr(getattr(a, "genetics", None), "harvest_efficiency", 1.0) * 5.0
            for a in self.agents
        ])

        collisions = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(positions[i] - positions[j]))
                if dist - (radii[i] + radii[j] + margin) < 0:
                    collisions.append(f"{self.agents[i].id} <-> {self.agents[j].id}")

        return collisions

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _capture_metrics(self) -> Dict[str, Any]:
        """Aggregate swarm statistics."""
        total_energy = sum(getattr(a, "energy", 0.0) for a in self.agents)
        generations = [
            a.genetics.generation
            for a in self.agents
            if hasattr(a, "genetics") and a.genetics
        ]
        structures = sum(getattr(a, "structures_built", 0) for a in self.agents)
        tasks_done = sum(
            1 for t in self.task_queue if t.get("status") == "done"
        )

        return {
            "status": "success",
            "ticks": self.tick_count,
            "population": len(self.agents),
            "biomass_energy": round(total_energy, 2),
            "structures_built": structures,
            "max_generation": max(generations) if generations else 0,
            "resources_remaining": sum(1 for r in self.resources if r["amount"] > 1.0),
            "tasks_completed": tasks_done,
            "tasks_remaining": len(self.task_queue),
        }
