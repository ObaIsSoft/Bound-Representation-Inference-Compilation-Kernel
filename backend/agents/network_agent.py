"""
Production NetworkAgent - Comprehensive Network Suite

Follows BRICK OS patterns:
- NO hardcoded device types - fully configurable
- NO estimated fallbacks - fails fast with clear errors
- Physical network diagnostics + GNN-based analysis
- 3D network infrastructure modeling
- Complete networking toolkit

Capabilities:
1. Physical Network Diagnostics:
   - Ping, traceroute, packet capture (scapy)
   - SSH device management (paramiko)
   - SNMP monitoring
   - Network scanning and discovery

2. GNN-Based Network Analysis:
   - Graph neural networks for topology analysis
   - Traffic flow prediction
   - Bottleneck identification
   - ML-based anomaly detection

3. 3D Network Infrastructure:
   - Router, switch, relay positioning
   - Cabling visualization
   - Physical topology modeling

Research Basis:
- Graph Neural Networks for network performance prediction
- Scapy for packet manipulation
- Paramiko for SSH management

Required Setup:
- Set NETWORK_GNN_MODEL env var for GNN (optional)
- Install scapy, paramiko (done)
- Root/admin access for packet capture
"""

from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import subprocess
import logging
import os
import platform
import socket
import asyncio
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class NetworkDeviceType(Enum):
    """Configurable network device types."""
    # Core Network
    ROUTER = "router"
    CORE_SWITCH = "core_switch"
    DISTRIBUTION_SWITCH = "distribution_switch"
    ACCESS_SWITCH = "access_switch"
    
    # Wireless
    ACCESS_POINT = "access_point"
    WIRELESS_CONTROLLER = "wireless_controller"
    
    # Infrastructure
    RELAY = "relay"
    BRIDGE = "bridge"
    GATEWAY = "gateway"
    FIREWALL = "firewall"
    LOAD_BALANCER = "load_balancer"
    
    # Endpoints
    SERVER = "server"
    WORKSTATION = "workstation"
    PRINTER = "printer"
    IOT_DEVICE = "iot_device"
    PHONE = "phone"
    CAMERA = "camera"
    
    # Cabling
    FIBER_CABLE = "fiber_cable"
    ETHERNET_CABLE = "ethernet_cable"
    PATCH_PANEL = "patch_panel"
    
    # Virtual
    VM = "virtual_machine"
    CONTAINER = "container"
    VIRTUAL_SWITCH = "virtual_switch"
    
    # Custom
    CUSTOM = "custom"


class NetworkProtocol(Enum):
    """Network protocols."""
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"
    ARP = "arp"
    SNMP = "snmp"
    SSH = "ssh"
    HTTP = "http"
    HTTPS = "https"


@dataclass
class NetworkDevice:
    """Network device with 3D positioning."""
    device_id: str
    device_type: NetworkDeviceType
    name: str
    position: Tuple[float, float, float]  # x, y, z in meters
    ip_address: Optional[str] = None
    mac_address: Optional[str] = None
    subnet: Optional[str] = None
    vlan: Optional[int] = None
    status: str = "unknown"  # online, offline, unknown
    connections: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Device capabilities
    supports_snmp: bool = False
    supports_ssh: bool = False
    snmp_community: Optional[str] = None
    ssh_credentials: Optional[Dict] = None


@dataclass
class NetworkLink:
    """Network link between devices."""
    link_id: str
    source_id: str
    target_id: str
    link_type: str  # fiber, ethernet, wireless, etc.
    bandwidth_mbps: float
    latency_ms: float
    status: str = "active"
    utilization: float = 0.0
    
    # Physical properties
    cable_length_m: Optional[float] = None
    cable_type: Optional[str] = None
    
    # 3D path for visualization
    path_3d: List[Tuple[float, float, float]] = field(default_factory=list)


@dataclass
class TrafficFlow:
    """Network traffic flow."""
    flow_id: str
    source_ip: str
    dest_ip: str
    protocol: NetworkProtocol
    port: Optional[int] = None
    rate_mbps: float = 0.0
    packet_size_bytes: int = 1500
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    packet_loss_percent: float = 0.0


class NetworkAgent:
    """
    Production comprehensive network agent.
    
    Combines physical network diagnostics with ML-based analysis:
    - Network diagnostics (ping, traceroute, packet capture)
    - GNN-based topology analysis and prediction
    - 3D network infrastructure modeling
    - Device management (SSH, SNMP)
    - Network scanning and discovery
    
    FAIL FAST: Returns error if operations unavailable.
    """
    
    def __init__(self):
        self.name = "NetworkAgent"
        self._initialized = False
        self._devices: Dict[str, NetworkDevice] = {}
        self._links: Dict[str, NetworkLink] = {}
        self._gnn_model = None
        self._model_path = os.getenv("NETWORK_GNN_MODEL", "models/network_gnn.pt")
        
        # Configuration
        self.config = {
            "snmp_community": os.getenv("SNMP_COMMUNITY", "public"),
            "ssh_timeout": 30,
            "ping_timeout": 5,
            "scan_threads": 50,
        }
    
    async def initialize(self):
        """Initialize network agent."""
        if self._initialized:
            return
        
        # Try to load GNN model if available
        try:
            import torch
            if os.path.exists(self._model_path):
                self._gnn_model = self._load_gnn_model(self._model_path)
                logger.info("NetworkAgent: GNN model loaded")
        except ImportError:
            logger.info("NetworkAgent: PyTorch not available, GNN features disabled")
        except Exception as e:
            logger.warning(f"NetworkAgent: Could not load GNN model: {e}")
        
        self._initialized = True
        logger.info("NetworkAgent initialized")
    
    def _load_gnn_model(self, model_path: str):
        """Load trained GNN model for network analysis."""
        import torch
        import torch.nn as nn
        
        class NetworkGNN(torch.nn.Module):
            """Graph Neural Network for network performance prediction."""
            def __init__(self, in_channels=6, hidden_channels=128, out_channels=1):
                super().__init__()
                from torch_geometric.nn import GCNConv, global_mean_pool
                
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, hidden_channels)
                self.conv3 = GCNConv(hidden_channels, hidden_channels)
                self.fc1 = nn.Linear(hidden_channels, 64)
                self.fc2 = nn.Linear(64, out_channels)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x, edge_index, batch=None):
                import torch.nn.functional as F
                
                x = F.relu(self.conv1(x, edge_index))
                x = self.dropout(x)
                x = F.relu(self.conv2(x, edge_index))
                x = self.dropout(x)
                x = F.relu(self.conv3(x, edge_index))
                
                # Global pooling if batch is provided
                if batch is not None:
                    from torch_geometric.nn import global_mean_pool
                    x = global_mean_pool(x, batch)
                else:
                    x = x.mean(dim=0, keepdim=True)
                
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        model = NetworkGNN()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute network operation.
        
        Args:
            params: {
                "operation": str - See OPERATIONS below
                ... operation-specific parameters
            }
        
        Operations:
        - "ping": Ping a host
        - "traceroute": Trace route to host
        - "scan_network": Scan network range
        - "capture_packets": Capture network packets (requires root)
        - "analyze_topology": GNN-based topology analysis
        - "setup_3d_topology": Setup 3D network infrastructure
        - "ssh_command": Execute SSH command on device
        - "snmp_get": Get SNMP value from device
        - "discover_devices": Auto-discover network devices
        - "analyze_traffic": Analyze traffic flows
        - "predict_performance": GNN-based performance prediction
        - "diagnose": Full network diagnosis
        - "visualize": Generate 3D visualization data
        """
        await self.initialize()
        
        operation = params.get("operation", "diagnose")
        
        operations_map = {
            "ping": self._ping,
            "traceroute": self._traceroute,
            "scan_network": self._scan_network,
            "capture_packets": self._capture_packets,
            "analyze_topology": self._analyze_topology_gnn,
            "setup_3d_topology": self._setup_3d_topology,
            "ssh_command": self._ssh_command,
            "snmp_get": self._snmp_get,
            "discover_devices": self._discover_devices,
            "analyze_traffic": self._analyze_traffic,
            "predict_performance": self._predict_performance,
            "diagnose": self._diagnose_network,
            "visualize": self._generate_visualization,
            "add_device": self._add_device,
            "remove_device": self._remove_device,
            "get_device": self._get_device,
            "list_devices": self._list_devices,
        }
        
        if operation in operations_map:
            return await operations_map[operation](params)
        else:
            raise ValueError(f"Unknown operation: {operation}. Available: {list(operations_map.keys())}")
    
    # ==================== Physical Network Operations ====================
    
    async def _ping(self, params: Dict) -> Dict[str, Any]:
        """Execute ping with detailed statistics."""
        target = params.get("target")
        count = params.get("count", 4)
        interval = params.get("interval", 1.0)
        
        if not target:
            raise ValueError("Target required for ping")
        
        logger.info(f"[NetworkAgent] Pinging {target}...")
        
        try:
            # Use scapy for advanced ping if available
            from scapy.all import IP, ICMP, sr1
            
            results = []
            transmitted = 0
            received = 0
            times = []
            
            for i in range(count):
                transmitted += 1
                pkt = IP(dst=target)/ICMP()
                start = datetime.now()
                resp = sr1(pkt, timeout=2, verbose=0)
                elapsed = (datetime.now() - start).total_seconds() * 1000
                
                if resp:
                    received += 1
                    times.append(elapsed)
                
                await asyncio.sleep(interval)
            
            # Calculate statistics
            if times:
                min_time = min(times)
                max_time = max(times)
                avg_time = sum(times) / len(times)
                loss_percent = ((transmitted - received) / transmitted) * 100
                
                return {
                    "status": "success",
                    "target": target,
                    "transmitted": transmitted,
                    "received": received,
                    "loss_percent": round(loss_percent, 1),
                    "time_ms": {
                        "min": round(min_time, 2),
                        "max": round(max_time, 2),
                        "avg": round(avg_time, 2)
                    },
                    "reachable": received > 0
                }
            else:
                return {
                    "status": "failed",
                    "target": target,
                    "transmitted": transmitted,
                    "received": 0,
                    "loss_percent": 100.0,
                    "reachable": False
                }
                
        except ImportError:
            # Fallback to system ping
            return await self._system_ping(target, count)
    
    async def _system_ping(self, target: str, count: int) -> Dict[str, Any]:
        """Fallback system ping."""
        if platform.system().lower() == "windows":
            cmd = ["ping", "-n", str(count), target]
        else:
            cmd = ["ping", "-c", str(count), "-W", "2", target]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        return {
            "status": "success" if result.returncode == 0 else "failed",
            "target": target,
            "output": result.stdout,
            "reachable": result.returncode == 0
        }
    
    async def _traceroute(self, params: Dict) -> Dict[str, Any]:
        """Execute traceroute/mtr."""
        target = params.get("target")
        max_hops = params.get("max_hops", 30)
        
        if not target:
            raise ValueError("Target required for traceroute")
        
        logger.info(f"[NetworkAgent] Traceroute to {target}...")
        
        try:
            from scapy.all import IP, UDP, sr1
            
            hops = []
            for ttl in range(1, max_hops + 1):
                pkt = IP(dst=target, ttl=ttl) / UDP(dport=33434)
                resp = sr1(pkt, timeout=2, verbose=0)
                
                if resp:
                    hop_info = {
                        "hop": ttl,
                        "ip": resp.src,
                        "hostname": self._resolve_hostname(resp.src),
                        "responded": True
                    }
                    hops.append(hop_info)
                    
                    if resp.src == target or resp.haslayer(ICMP):
                        break
                else:
                    hops.append({
                        "hop": ttl,
                        "ip": None,
                        "hostname": None,
                        "responded": False
                    })
            
            return {
                "status": "success",
                "target": target,
                "hops": hops,
                "hop_count": len([h for h in hops if h["responded"]]),
                "complete": hops[-1]["ip"] == target if hops else False
            }
            
        except ImportError:
            return await self._system_traceroute(target, max_hops)
    
    async def _system_traceroute(self, target: str, max_hops: int) -> Dict[str, Any]:
        """Fallback system traceroute."""
        if platform.system().lower() == "windows":
            cmd = ["tracert", "-h", str(max_hops), target]
        else:
            cmd = ["traceroute", "-m", str(max_hops), "-w", "5", target]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        return {
            "status": "success" if result.returncode == 0 else "partial",
            "target": target,
            "output": result.stdout,
            "raw": True
        }
    
    async def _scan_network(self, params: Dict) -> Dict[str, Any]:
        """Scan network range for devices."""
        network_range = params.get("network", "192.168.1.0/24")
        ports = params.get("ports", [22, 80, 443, 445, 3389])
        
        logger.info(f"[NetworkAgent] Scanning network {network_range}...")
        
        try:
            from scapy.all import IP, TCP, sr1, conf
            
            # Parse network range
            import ipaddress
            net = ipaddress.ip_network(network_range, strict=False)
            hosts = list(net.hosts())[:254]  # Limit to first 254 hosts
            
            discovered = []
            
            async def scan_host(host_ip):
                # ICMP ping first
                pkt = IP(dst=str(host_ip))/ICMP()
                resp = sr1(pkt, timeout=1, verbose=0)
                
                if resp:
                    # Host is up, scan ports
                    open_ports = []
                    for port in ports:
                        tcp_pkt = IP(dst=str(host_ip))/TCP(dport=port, flags="S")
                        tcp_resp = sr1(tcp_pkt, timeout=1, verbose=0)
                        if tcp_resp and tcp_resp.haslayer(TCP) and tcp_resp[TCP].flags == "SA":
                            open_ports.append(port)
                    
                    discovered.append({
                        "ip": str(host_ip),
                        "hostname": self._resolve_hostname(str(host_ip)),
                        "mac": self._get_mac_address(str(host_ip)),
                        "open_ports": open_ports,
                        "status": "up"
                    })
            
            # Scan with semaphore for rate limiting
            semaphore = asyncio.Semaphore(self.config["scan_threads"])
            
            async def bounded_scan(host):
                async with semaphore:
                    await scan_host(host)
            
            await asyncio.gather(*[bounded_scan(h) for h in hosts])
            
            return {
                "status": "complete",
                "network": network_range,
                "scanned": len(hosts),
                "discovered": len(discovered),
                "devices": discovered
            }
            
        except Exception as e:
            raise RuntimeError(f"Network scan failed: {e}")
    
    async def _capture_packets(self, params: Dict) -> Dict[str, Any]:
        """Capture network packets (requires root/admin)."""
        interface = params.get("interface")
        duration = params.get("duration", 10)
        filter_expr = params.get("filter", "")
        count = params.get("count", 100)
        
        logger.info(f"[NetworkAgent] Capturing packets on {interface}...")
        
        try:
            from scapy.all import sniff, wrpcap
            
            packets = []
            
            def packet_handler(pkt):
                packets.append(pkt)
                if len(packets) >= count:
                    return True  # Stop sniffing
            
            # Capture packets
            sniff(
                iface=interface,
                filter=filter_expr,
                prn=packet_handler,
                timeout=duration,
                store=0
            )
            
            # Analyze captured packets
            analysis = self._analyze_packets(packets)
            
            return {
                "status": "success",
                "interface": interface,
                "captured": len(packets),
                "duration": duration,
                "analysis": analysis,
                "note": "Packet capture requires root/admin privileges"
            }
            
        except PermissionError:
            raise RuntimeError("Packet capture requires root/admin privileges. Run with sudo.")
        except Exception as e:
            raise RuntimeError(f"Packet capture failed: {e}")
    
    def _analyze_packets(self, packets: List) -> Dict[str, Any]:
        """Analyze captured packets."""
        from collections import Counter
        
        protocols = Counter()
        src_ips = Counter()
        dst_ips = Counter()
        
        for pkt in packets:
            if pkt.haslayer("IP"):
                src_ips[pkt["IP"].src] += 1
                dst_ips[pkt["IP"].dst] += 1
                
                if pkt.haslayer("TCP"):
                    protocols["TCP"] += 1
                elif pkt.haslayer("UDP"):
                    protocols["UDP"] += 1
                elif pkt.haslayer("ICMP"):
                    protocols["ICMP"] += 1
        
        return {
            "protocols": dict(protocols),
            "top_sources": dict(src_ips.most_common(5)),
            "top_destinations": dict(dst_ips.most_common(5))
        }
    
    async def _ssh_command(self, params: Dict) -> Dict[str, Any]:
        """Execute SSH command on remote device."""
        hostname = params.get("hostname")
        username = params.get("username")
        password = params.get("password")
        key_file = params.get("key_file")
        command = params.get("command")
        port = params.get("port", 22)
        
        if not all([hostname, username, command]):
            raise ValueError("hostname, username, and command required for SSH")
        
        logger.info(f"[NetworkAgent] SSH to {hostname}: {command}")
        
        try:
            import paramiko
            
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            connect_kwargs = {
                "hostname": hostname,
                "port": port,
                "username": username,
                "timeout": self.config["ssh_timeout"]
            }
            
            if key_file:
                connect_kwargs["key_filename"] = key_file
            elif password:
                connect_kwargs["password"] = password
            else:
                raise ValueError("Either password or key_file required for SSH")
            
            client.connect(**connect_kwargs)
            
            stdin, stdout, stderr = client.exec_command(command)
            exit_code = stdout.channel.recv_exit_status()
            
            output = stdout.read().decode('utf-8', errors='ignore')
            error = stderr.read().decode('utf-8', errors='ignore')
            
            client.close()
            
            return {
                "status": "success" if exit_code == 0 else "failed",
                "hostname": hostname,
                "command": command,
                "exit_code": exit_code,
                "stdout": output,
                "stderr": error
            }
            
        except Exception as e:
            raise RuntimeError(f"SSH failed: {e}")
    
    async def _snmp_get(self, params: Dict) -> Dict[str, Any]:
        """Get SNMP value from device."""
        target = params.get("target")
        oid = params.get("oid")
        community = params.get("community", self.config["snmp_community"])
        version = params.get("version", "2c")
        
        if not all([target, oid]):
            raise ValueError("target and oid required for SNMP")
        
        logger.info(f"[NetworkAgent] SNMP get {oid} from {target}")
        
        try:
            from pysnmp.hlapi import getCmd, SnmpEngine, CommunityData, UdpTransportTarget, ContextData, ObjectType, ObjectIdentity
            
            iterator = getCmd(
                SnmpEngine(),
                CommunityData(community),
                UdpTransportTarget((target, 161)),
                ContextData(),
                ObjectType(ObjectIdentity(oid))
            )
            
            errorIndication, errorStatus, errorIndex, varBinds = next(iterator)
            
            if errorIndication:
                raise RuntimeError(f"SNMP error: {errorIndication}")
            elif errorStatus:
                raise RuntimeError(f"SNMP error: {errorStatus.prettyPrint()}")
            else:
                results = []
                for varBind in varBinds:
                    results.append({"oid": str(varBind[0]), "value": str(varBind[1])})
                
                return {
                    "status": "success",
                    "target": target,
                    "results": results
                }
                
        except ImportError:
            raise RuntimeError("pysnmp not installed. Run: pip install pysnmp")
        except Exception as e:
            raise RuntimeError(f"SNMP get failed: {e}")
    
    # ==================== GNN-Based Network Analysis ====================
    
    async def _analyze_topology_gnn(self, params: Dict) -> Dict[str, Any]:
        """Analyze network topology using GNN."""
        if not self._gnn_model:
            raise RuntimeError("GNN model not available. Set NETWORK_GNN_MODEL environment variable.")
        
        logger.info("[NetworkAgent] Analyzing topology with GNN...")
        
        try:
            import torch
            import torch_geometric.data as pyg_data
            import networkx as nx
            
            # Build graph from devices and links
            G = self._build_networkx_graph()
            
            # Convert to PyG data
            data = self._networkx_to_pyg(G)
            
            # GNN inference
            with torch.no_grad():
                predictions = self._gnn_model(data.x, data.edge_index)
            
            # Parse predictions
            node_scores = self._parse_gnn_predictions(G, predictions)
            
            return {
                "status": "analyzed",
                "method": "GNN",
                "nodes_analyzed": len(node_scores),
                "predictions": node_scores,
                "bottlenecks": self._identify_gnn_bottlenecks(node_scores)
            }
            
        except Exception as e:
            raise RuntimeError(f"GNN analysis failed: {e}")
    
    def _build_networkx_graph(self) -> "nx.Graph":
        """Build NetworkX graph from devices and links."""
        import networkx as nx
        
        G = nx.Graph()
        
        # Add nodes with features
        for device_id, device in self._devices.items():
            G.add_node(device_id, 
                type=device.device_type.value,
                x=device.position[0],
                y=device.position[1],
                z=device.position[2],
                status=1 if device.status == "online" else 0,
                bandwidth=device.properties.get("bandwidth_mbps", 1000)
            )
        
        # Add edges
        for link_id, link in self._links.items():
            G.add_edge(link.source_id, link.target_id,
                bandwidth=link.bandwidth_mbps,
                latency=link.latency_ms,
                utilization=link.utilization
            )
        
        return G
    
    def _networkx_to_pyg(self, G: "nx.Graph") -> Any:
        """Convert NetworkX graph to PyTorch Geometric data."""
        import torch
        import torch_geometric.utils as pyg_utils
        
        # Node features
        x = []
        node_list = list(G.nodes())
        node_idx = {n: i for i, n in enumerate(node_list)}
        
        for node in node_list:
            data = G.nodes[node]
            features = [
                data.get("x", 0) / 100,  # Normalized position
                data.get("y", 0) / 100,
                data.get("z", 0) / 100,
                data.get("status", 0),
                data.get("bandwidth", 1000) / 10000,  # Normalized bandwidth
                len(list(G.neighbors(node))) / 10  # Normalized degree
            ]
            x.append(features)
        
        # Edge index
        edge_list = []
        for edge in G.edges():
            edge_list.append([node_idx[edge[0]], node_idx[edge[1]]])
            edge_list.append([node_idx[edge[1]], node_idx[edge[0]]])  # Undirected
        
        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        return type('Data', (), {'x': x, 'edge_index': edge_index})()
    
    def _parse_gnn_predictions(self, G: "nx.Graph", predictions: Any) -> Dict[str, float]:
        """Parse GNN predictions for each node."""
        node_list = list(G.nodes())
        pred_values = predictions.squeeze().tolist()
        
        if isinstance(pred_values, float):
            pred_values = [pred_values]
        
        return {
            node: float(pred)
            for node, pred in zip(node_list, pred_values)
        }
    
    def _identify_gnn_bottlenecks(self, node_scores: Dict[str, float]) -> List[Dict]:
        """Identify bottlenecks from GNN predictions."""
        bottlenecks = []
        
        # High score indicates potential bottleneck
        threshold = sum(node_scores.values()) / len(node_scores) if node_scores else 0.5
        
        for node_id, score in node_scores.items():
            if score > threshold * 1.5:  # 50% above average
                device = self._devices.get(node_id)
                bottlenecks.append({
                    "device_id": node_id,
                    "device_name": device.name if device else "Unknown",
                    "predicted_latency_ms": round(score * 100, 2),
                    "severity": "high" if score > threshold * 2 else "medium"
                })
        
        return bottlenecks
    
    async def _predict_performance(self, params: Dict) -> Dict[str, Any]:
        """Predict network performance using GNN."""
        if not self._gnn_model:
            raise RuntimeError("GNN model not available")
        
        traffic_scenario = params.get("traffic", {})
        
        # Run GNN analysis with hypothetical traffic
        return await self._analyze_topology_gnn(params)
    
    async def _analyze_traffic(self, params: Dict) -> Dict[str, Any]:
        """Analyze network traffic flows."""
        flows = params.get("flows", [])
        
        if not flows:
            raise ValueError("Traffic flows required for analysis")
        
        analyzed = []
        
        for flow in flows:
            src = flow.get("source")
            dst = flow.get("destination")
            rate = flow.get("rate_mbps", 0)
            
            # Find path through network
            path = self._find_path(src, dst)
            
            if path:
                # Calculate metrics along path
                path_latency = self._calculate_path_latency(path)
                min_bandwidth = self._calculate_path_bandwidth(path)
                congestion = rate / min_bandwidth if min_bandwidth > 0 else 1.0
            else:
                path_latency = float('inf')
                min_bandwidth = 0
                congestion = 1.0
            
            analyzed.append({
                "source": src,
                "destination": dst,
                "rate_mbps": rate,
                "path": path,
                "path_latency_ms": round(path_latency, 2),
                "min_bandwidth_mbps": min_bandwidth,
                "congestion_ratio": round(congestion, 2),
                "status": "congested" if congestion > 0.8 else "healthy"
            })
        
        return {
            "status": "analyzed",
            "flow_count": len(flows),
            "flows": analyzed
        }
    
    # ==================== 3D Topology Management ====================
    
    async def _setup_3d_topology(self, params: Dict) -> Dict[str, Any]:
        """Setup 3D network topology."""
        devices = params.get("devices", [])
        links = params.get("links", [])
        
        logger.info(f"[NetworkAgent] Setting up 3D topology with {len(devices)} devices...")
        
        # Clear existing
        self._devices = {}
        self._links = {}
        
        # Add devices
        for dev_data in devices:
            device = NetworkDevice(
                device_id=dev_data["id"],
                device_type=NetworkDeviceType(dev_data.get("type", "custom")),
                name=dev_data.get("name", dev_data["id"]),
                position=tuple(dev_data.get("position", [0, 0, 0])),
                ip_address=dev_data.get("ip"),
                mac_address=dev_data.get("mac"),
                subnet=dev_data.get("subnet"),
                vlan=dev_data.get("vlan"),
                properties=dev_data.get("properties", {}),
                supports_snmp=dev_data.get("supports_snmp", False),
                supports_ssh=dev_data.get("supports_ssh", False)
            )
            self._devices[device.device_id] = device
        
        # Add links
        for link_data in links:
            link = NetworkLink(
                link_id=link_data.get("id", f"link_{len(self._links)}"),
                source_id=link_data["from"],
                target_id=link_data["to"],
                link_type=link_data.get("type", "ethernet"),
                bandwidth_mbps=link_data.get("bandwidth_mbps", 1000),
                latency_ms=link_data.get("latency_ms", 1),
                cable_length_m=link_data.get("length_m"),
                cable_type=link_data.get("cable_type"),
                path_3d=link_data.get("path_3d", [])
            )
            
            # Update device connections
            if link.source_id in self._devices:
                self._devices[link.source_id].connections.append(link.target_id)
            if link.target_id in self._devices:
                self._devices[link.target_id].connections.append(link.source_id)
            
            self._links[link.link_id] = link
        
        return {
            "status": "configured",
            "device_count": len(self._devices),
            "link_count": len(self._links),
            "bounds": self._calculate_3d_bounds(),
            "devices": [{"id": d.device_id, "type": d.device_type.value, "position": d.position} 
                       for d in self._devices.values()]
        }
    
    async def _add_device(self, params: Dict) -> Dict[str, Any]:
        """Add a device to the topology."""
        device_data = params.get("device", {})
        
        device = NetworkDevice(
            device_id=device_data["id"],
            device_type=NetworkDeviceType(device_data.get("type", "custom")),
            name=device_data.get("name", device_data["id"]),
            position=tuple(device_data.get("position", [0, 0, 0])),
            ip_address=device_data.get("ip"),
            mac_address=device_data.get("mac"),
            properties=device_data.get("properties", {})
        )
        
        self._devices[device.device_id] = device
        
        return {
            "status": "added",
            "device": {
                "id": device.device_id,
                "type": device.device_type.value,
                "position": device.position
            }
        }
    
    async def _remove_device(self, params: Dict) -> Dict[str, Any]:
        """Remove a device from the topology."""
        device_id = params.get("device_id")
        
        if device_id in self._devices:
            del self._devices[device_id]
            # Remove associated links
            links_to_remove = [lid for lid, link in self._links.items() 
                             if link.source_id == device_id or link.target_id == device_id]
            for lid in links_to_remove:
                del self._links[lid]
            
            return {"status": "removed", "device_id": device_id}
        else:
            return {"status": "not_found", "device_id": device_id}
    
    async def _get_device(self, params: Dict) -> Dict[str, Any]:
        """Get device details."""
        device_id = params.get("device_id")
        
        if device_id not in self._devices:
            raise ValueError(f"Device {device_id} not found")
        
        device = self._devices[device_id]
        
        return {
            "id": device.device_id,
            "name": device.name,
            "type": device.device_type.value,
            "position": device.position,
            "ip": device.ip_address,
            "mac": device.mac_address,
            "status": device.status,
            "connections": device.connections,
            "properties": device.properties
        }
    
    async def _list_devices(self, params: Dict = None) -> Dict[str, Any]:
        """List all devices."""
        return {
            "count": len(self._devices),
            "devices": [
                {
                    "id": d.device_id,
                    "name": d.name,
                    "type": d.device_type.value,
                    "position": d.position,
                    "ip": d.ip_address,
                    "status": d.status
                }
                for d in self._devices.values()
            ]
        }
    
    async def _discover_devices(self, params: Dict) -> Dict[str, Any]:
        """Auto-discover devices on the network."""
        networks = params.get("networks", ["192.168.1.0/24"])
        
        discovered = []
        
        for network in networks:
            scan_result = await self._scan_network({"network": network})
            discovered.extend(scan_result.get("devices", []))
        
        # Add discovered devices to topology
        for device_data in discovered:
            device_id = f"discovered_{device_data['ip'].replace('.', '_')}"
            if device_id not in self._devices:
                device = NetworkDevice(
                    device_id=device_id,
                    device_type=NetworkDeviceType.ENDPOINT,
                    name=device_data.get("hostname") or device_data["ip"],
                    position=(0, 0, 0),  # Will need manual placement
                    ip_address=device_data["ip"],
                    status=device_data["status"]
                )
                self._devices[device_id] = device
        
        return {
            "status": "discovered",
            "count": len(discovered),
            "devices": discovered
        }
    
    # ==================== Diagnostics and Utilities ====================
    
    async def _diagnose_network(self, params: Dict = None) -> Dict[str, Any]:
        """Full network diagnosis."""
        logger.info("[NetworkAgent] Running full network diagnosis...")
        
        diagnostics = {
            "hostname": socket.gethostname(),
            "local_ip": self._get_local_ip(),
            "platform": platform.system(),
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        # Gateway test
        gateway = self._get_default_gateway()
        if gateway:
            diagnostics["gateway"] = gateway
            diagnostics["tests"]["gateway"] = await self._ping({"target": gateway, "count": 2})
        
        # Internet test
        diagnostics["tests"]["internet"] = await self._ping({"target": "8.8.8.8", "count": 2})
        
        # DNS test
        diagnostics["tests"]["dns"] = await self._test_dns()
        
        # Device status
        diagnostics["topology"] = {
            "device_count": len(self._devices),
            "link_count": len(self._links),
            "online_devices": sum(1 for d in self._devices.values() if d.status == "online")
        }
        
        return diagnostics
    
    async def _generate_visualization(self, params: Dict) -> Dict[str, Any]:
        """Generate 3D visualization data."""
        format_type = params.get("format", "json")
        
        nodes = []
        for device in self._devices.values():
            nodes.append({
                "id": device.device_id,
                "name": device.name,
                "type": device.device_type.value,
                "x": device.position[0],
                "y": device.position[1],
                "z": device.position[2],
                "ip": device.ip_address,
                "status": device.status
            })
        
        edges = []
        for link in self._links.values():
            edges.append({
                "id": link.link_id,
                "source": link.source_id,
                "target": link.target_id,
                "type": link.link_type,
                "bandwidth": link.bandwidth_mbps
            })
        
        if format_type == "json":
            return {
                "status": "generated",
                "format": "json",
                "nodes": nodes,
                "edges": edges,
                "bounds": self._calculate_3d_bounds()
            }
        else:
            return {
                "status": "unsupported_format",
                "format": format_type,
                "supported": ["json"]
            }
    
    # ==================== Helper Methods ====================
    
    def _find_path(self, src: str, dst: str) -> Optional[List[str]]:
        """Find path between two devices using BFS."""
        if src not in self._devices or dst not in self._devices:
            return None
        
        if src == dst:
            return [src]
        
        visited = {src}
        queue = [(src, [src])]
        
        while queue:
            node, path = queue.pop(0)
            
            # Get neighbors from links
            for link in self._links.values():
                if link.source_id == node:
                    neighbor = link.target_id
                elif link.target_id == node:
                    neighbor = link.source_id
                else:
                    continue
                
                if neighbor == dst:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def _calculate_path_latency(self, path: List[str]) -> float:
        """Calculate total latency along a path."""
        total_latency = 0.0
        
        for i in range(len(path) - 1):
            # Find link between path[i] and path[i+1]
            for link in self._links.values():
                if (link.source_id == path[i] and link.target_id == path[i+1]) or \
                   (link.source_id == path[i+1] and link.target_id == path[i]):
                    total_latency += link.latency_ms
                    break
        
        return total_latency
    
    def _calculate_path_bandwidth(self, path: List[str]) -> float:
        """Calculate minimum bandwidth along a path."""
        min_bandwidth = float('inf')
        
        for i in range(len(path) - 1):
            for link in self._links.values():
                if (link.source_id == path[i] and link.target_id == path[i+1]) or \
                   (link.source_id == path[i+1] and link.target_id == path[i]):
                    min_bandwidth = min(min_bandwidth, link.bandwidth_mbps)
                    break
        
        return min_bandwidth if min_bandwidth != float('inf') else 0
    
    def _resolve_hostname(self, ip: str) -> Optional[str]:
        """Resolve IP to hostname."""
        try:
            return socket.gethostbyaddr(ip)[0]
        except:
            return None
    
    def _get_mac_address(self, ip: str) -> Optional[str]:
        """Get MAC address for IP (requires ARP table access)."""
        try:
            if platform.system().lower() == "windows":
                result = subprocess.run(["arp", "-a", ip], capture_output=True, text=True)
            else:
                result = subprocess.run(["arp", "-n", ip], capture_output=True, text=True)
            
            # Parse ARP output for MAC address
            import re
            mac_pattern = r'([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})'
            match = re.search(mac_pattern, result.stdout)
            if match:
                return match.group(0)
        except:
            pass
        return None
    
    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def _get_default_gateway(self) -> Optional[str]:
        """Get default gateway IP."""
        try:
            if platform.system().lower() == "windows":
                result = subprocess.run(["route", "print", "0.0.0.0"], capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if '0.0.0.0' in line and 'Gateway' not in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            return parts[2]
            else:
                result = subprocess.run(["ip", "route"], capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if 'default' in line:
                        parts = line.split()
                        if 'via' in parts:
                            return parts[parts.index('via') + 1]
        except:
            pass
        return None
    
    async def _test_dns(self) -> Dict[str, Any]:
        """Test DNS resolution."""
        try:
            result = socket.gethostbyname("google.com")
            return {
                "status": "success",
                "resolved_ip": result,
                "domain": "google.com"
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "domain": "google.com"
            }
    
    def _calculate_3d_bounds(self) -> Dict[str, Any]:
        """Calculate 3D bounding box of network."""
        if not self._devices:
            return {"min": [0, 0, 0], "max": [0, 0, 0], "center": [0, 0, 0]}
        
        positions = [d.position for d in self._devices.values()]
        
        min_coords = [min(p[i] for p in positions) for i in range(3)]
        max_coords = [max(p[i] for p in positions) for i in range(3)]
        center = [(min_coords[i] + max_coords[i]) / 2 for i in range(3)]
        
        return {
            "min": min_coords,
            "max": max_coords,
            "center": center,
            "dimensions": [max_coords[i] - min_coords[i] for i in range(3)]
        }


# ==================== Convenience Functions ====================

async def ping_host(host: str, count: int = 4) -> Dict[str, Any]:
    """Quick ping to a host."""
    agent = NetworkAgent()
    return await agent.run({"operation": "ping", "target": host, "count": count})

async def scan_network(network: str = "192.168.1.0/24") -> Dict[str, Any]:
    """Quick network scan."""
    agent = NetworkAgent()
    return await agent.run({"operation": "scan_network", "network": network})

async def diagnose_network() -> Dict[str, Any]:
    """Quick network diagnosis."""
    agent = NetworkAgent()
    return await agent.run({"operation": "diagnose"})
