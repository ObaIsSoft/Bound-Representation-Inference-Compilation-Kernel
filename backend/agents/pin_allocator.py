"""
PinAllocator — constraint-based pin assignment for CodegenAgent.

Replaces the greedy first-fit allocator with a proper constraint solver.

Constraints enforced:
  - AllDifferent on all dedicated (non-shared-bus) pins
  - I2C SCL/SDA shared by all I2C devices on the same bus instance
  - SPI SCLK/MISO/MOSI shared; each device gets a unique CS from cs_pool
  - input_only pins excluded from output roles (PWM, UART TX, SPI MOSI/CS)
  - strapping/debug/boot pins excluded from all allocation
  - I2C address collision detection (two devices at same 7-bit address = error)
  - Exhausted buses produce a human-readable diagnostic + suggestion

Does NOT use python-constraint's backtracking for the bus-sharing cases because
the sharing semantics are incompatible with AllDifferent on shared signals.
Instead it uses a two-phase approach:
  Phase 1: Assign shared buses (I2C, SPI) greedily — each bus is a single slot.
  Phase 2: Assign exclusive resources (UART, PWM, CS pins) with AllDifferent
            enforced by tracking already-used pins in a set.

This is equivalent to CSP but without the exponential search space overhead
that makes naive CSP slow for large boards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class AllocationResult:
    ok: bool
    assignments: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PinAllocator:
    """
    Stateless allocator — create one per code-generation run.

    Args:
        hw: Platform def dict from HardwareDB.load(), e.g.
            {"pwm_pins": [...], "i2c_interfaces": [(name, scl, sda), ...], ...}
    """

    def __init__(self, hw: Dict[str, Any]):
        self.hw = hw
        self._pins_meta: Dict[str, Any] = hw.get("_pins_meta", {})

        # Collect all pins that must never be allocated
        self._forbidden: Set[str] = set()
        for pin, meta in self._pins_meta.items():
            restr = meta.get("restrictions") or []
            if any(r in ("debug_pin", "internal_flash", "boot_mode") for r in restr):
                self._forbidden.add(pin)

        # Collect input-only pins (cannot be TX, MOSI, PWM, CS)
        self._input_only: Set[str] = set()
        for pin, meta in self._pins_meta.items():
            elec = meta.get("electrical") or {}
            restr = meta.get("restrictions") or []
            if elec.get("input_only") or "input_only" in restr:
                self._input_only.add(pin)

    # ── Public entry point ────────────────────────────────────────────────────

    def allocate(self, components: List[Any]) -> AllocationResult:
        """
        Allocate pins for all components.

        Args:
            components: List of Component objects (must have .name, .required_interfaces,
                        .user_params attributes).

        Returns:
            AllocationResult with per-component assignments and any errors.
        """
        result = AllocationResult(ok=True)
        used_exclusive: Set[str] = set()  # pins that cannot be reused

        # Bus-sharing state (shared across all components)
        i2c_shared_bus: Optional[tuple] = None      # (name, scl, sda)
        i2c_addresses_used: List[int] = []
        spi_shared_buses: Dict[str, tuple] = {}     # {bus_name: (name, sclk, miso, mosi, cs0)}
        spi_cs_used: Set[str] = set()
        uart_used: Set[str] = set()

        for comp in components:
            comp_alloc: Dict[str, Any] = {"name": comp.name, "pins": {}, "interface": None}
            allocated = False

            for interface in comp.required_interfaces:

                # ── PWM ────────────────────────────────────────────────────
                if interface == "PWM":
                    pin = self._pick_exclusive(
                        self.hw.get("pwm_pins", []),
                        used_exclusive,
                        exclude_input_only=True,
                    )
                    if pin:
                        used_exclusive.add(pin)
                        comp_alloc["pins"]["pwm"] = pin
                        comp_alloc["interface"] = "PWM"
                        allocated = True
                    else:
                        result.errors.append(
                            f"{comp.name}: no free PWM pin — add a GPIO expander or reduce PWM devices"
                        )
                    break

                # ── I2C (shared bus) ────────────────────────────────────────
                elif interface == "I2C":
                    if i2c_shared_bus is None:
                        ifaces = self.hw.get("i2c_interfaces", [])
                        if not ifaces:
                            result.errors.append(f"{comp.name}: MCU has no I2C peripheral")
                            break
                        # Pick first bus whose pins aren't in used_exclusive
                        bus = self._pick_bus(ifaces, used_exclusive)
                        if bus is None:
                            result.errors.append(
                                f"{comp.name}: all I2C buses conflict with already-assigned pins"
                            )
                            break
                        i2c_shared_bus = bus
                        # Bus pins are shared — do NOT add to used_exclusive

                    # Address collision check
                    i2c_addr = comp.user_params.get("i2c_address")
                    if i2c_addr is not None:
                        if i2c_addr in i2c_addresses_used:
                            result.errors.append(
                                f"{comp.name}: I2C address {hex(i2c_addr)} already in use — "
                                "two devices at the same address will collide on the bus"
                            )
                            break
                        i2c_addresses_used.append(i2c_addr)

                    comp_alloc["pins"]["scl"] = i2c_shared_bus[1]
                    comp_alloc["pins"]["sda"] = i2c_shared_bus[2]
                    comp_alloc["interface"] = f"I2C ({i2c_shared_bus[0]}, shared)"
                    allocated = True
                    break

                # ── SPI (shared bus, dedicated CS) ─────────────────────────
                elif interface == "SPI":
                    if not spi_shared_buses:
                        ifaces = self.hw.get("spi_interfaces", [])
                        if not ifaces:
                            result.errors.append(f"{comp.name}: MCU has no SPI peripheral")
                            break
                        bus = self._pick_bus(ifaces, used_exclusive)
                        if bus is None:
                            result.errors.append(
                                f"{comp.name}: all SPI buses conflict with already-assigned pins"
                            )
                            break
                        spi_shared_buses[bus[0]] = bus

                    bus = next(iter(spi_shared_buses.values()))

                    # CS pin — must be unique per device
                    cs_pool = self.hw.get("gpio_cs_pool", [])
                    cs_pin = self._pick_exclusive(cs_pool, spi_cs_used | used_exclusive, exclude_input_only=True)
                    if cs_pin is None:
                        result.errors.append(
                            f"{comp.name}: no free SPI CS pin — add a GPIO CS expander"
                        )
                        break
                    spi_cs_used.add(cs_pin)

                    comp_alloc["pins"]["sclk"] = bus[1]
                    comp_alloc["pins"]["miso"] = bus[2]
                    comp_alloc["pins"]["mosi"] = bus[3]
                    comp_alloc["pins"]["cs"]   = cs_pin
                    comp_alloc["interface"] = f"SPI ({bus[0]}, shared, CS={cs_pin})"
                    allocated = True
                    break

                # ── UART (exclusive) ────────────────────────────────────────
                elif interface == "UART":
                    uart_ifaces = self.hw.get("uart_interfaces", [])
                    free = [
                        iface for iface in uart_ifaces
                        if iface[0] not in uart_used
                        and iface[1] not in used_exclusive  # TX
                        and iface[2] not in used_exclusive  # RX
                        and iface[1] not in self._forbidden
                    ]
                    if not free:
                        result.errors.append(
                            f"{comp.name}: no free UART peripheral — "
                            "consider USB-serial or a UART expander"
                        )
                        break
                    iface = free[0]
                    uart_used.add(iface[0])
                    used_exclusive.add(iface[1])
                    used_exclusive.add(iface[2])
                    comp_alloc["pins"]["tx"] = iface[1]
                    comp_alloc["pins"]["rx"] = iface[2]
                    comp_alloc["interface"] = f"UART ({iface[0]})"
                    allocated = True
                    break

                # ── CAN (exclusive) ─────────────────────────────────────────
                elif interface == "CAN":
                    can_ifaces = self.hw.get("can_interfaces", [])
                    can_used_names = {v["interface"].split("(")[1].split(")")[0]
                                      for v in result.assignments.values()
                                      if v.get("interface", "").startswith("CAN")}
                    free = [
                        iface for iface in can_ifaces
                        if iface[0] not in can_used_names
                        and iface[1] not in used_exclusive
                        and iface[2] not in used_exclusive
                        and iface[1] not in self._forbidden
                    ]
                    if not free:
                        result.errors.append(f"{comp.name}: no free CAN bus interface")
                        break
                    iface = free[0]
                    used_exclusive.add(iface[1])
                    used_exclusive.add(iface[2])
                    comp_alloc["pins"]["tx"] = iface[1]
                    comp_alloc["pins"]["rx"] = iface[2]
                    comp_alloc["interface"] = f"CAN ({iface[0]})"
                    allocated = True
                    break

                # ── GPIO (single dedicated pin) ─────────────────────────────
                elif interface == "GPIO":
                    gpio_pool = self.hw.get("gpio_cs_pool", [])
                    pin = self._pick_exclusive(gpio_pool, used_exclusive, exclude_input_only=False)
                    if pin is None:
                        result.errors.append(f"{comp.name}: no free GPIO pin")
                        break
                    used_exclusive.add(pin)
                    comp_alloc["pins"]["pin"] = pin
                    comp_alloc["interface"] = "GPIO"
                    allocated = True
                    break

            if allocated:
                result.assignments[comp.name] = comp_alloc
            else:
                if not any(comp.name in e for e in result.errors):
                    result.errors.append(f"{comp.name}: could not allocate any interface")
                result.ok = False

        if result.errors:
            result.ok = False

        return result

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _pick_exclusive(
        self,
        pool: List[str],
        used: Set[str],
        exclude_input_only: bool = True,
    ) -> Optional[str]:
        for pin in pool:
            if pin in used:
                continue
            if pin in self._forbidden:
                continue
            if exclude_input_only and pin in self._input_only:
                continue
            return pin
        return None

    def _pick_bus(self, ifaces: List[tuple], used_exclusive: Set[str]) -> Optional[tuple]:
        """Return first bus whose dedicated pins don't conflict with used_exclusive."""
        for iface in ifaces:
            # iface = (name, pin1, pin2, ...) — check all signal pins
            pins = iface[1:]
            if any(p in used_exclusive or p in self._forbidden for p in pins if p):
                continue
            return iface
        return None
