from typing import Dict, Any, List, Optional
import logging
import csv
import json
import os
import io
import zipfile
from pathlib import Path
from datetime import datetime

try:
    from backend.llm.provider import LLMProvider
    from backend.agent_registry import registry
except ImportError:
    from llm.provider import LLMProvider
    from agent_registry import registry

import asyncio
logger = logging.getLogger(__name__)

OUTPUT_BASE = Path("output")


class DocumentAgent:
    """
    Hardware design documentation and artifact generation agent.

    Responsibilities:
    - Phase 2: Synthesize design brief from agent data (planning phase)
    - Phase 6: Generate full deliverable package from pipeline outputs:
        BOM CSV, GCode file, DFM report, physics analysis report,
        firmware documentation, test protocol, risk register, main PDF,
        ZIP delivery package.

    All documents are written to output/{project_id}/ and served via API.
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.name = "DocumentAgent"
        self.llm_provider = llm_provider

    # =========================================================================
    # PHASE 6: FINAL DELIVERABLE PACKAGE
    # =========================================================================

    def generate_final_documentation(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate the complete hardware project deliverable package.

        Writes actual files to disk:
          - bom.csv            Bill of Materials (importable into Digi-Key / JLCPCB)
          - firmware.gcode     GCode from VMK kernel (if present)
          - dfm_report.md      DFM issues and manufacturability score
          - physics_report.md  Thermal + structural + fluid results, with plots if matplotlib available
          - firmware_docs.md   Pin map, build instructions, file manifest from CodegenAgent
          - test_protocol.md   Dynamic test cases based on environment and physics
          - risk_register.md   FMEA-style failure modes from physics safety factors + DFM issues
          - project_report.md  Master report linking all above
          - {project_id}_package.zip  Everything zipped for download

        Returns:
            {
                "status": "success",
                "document": {"content": str, "format": "markdown"},
                "files": {filename: filepath},
                "package_zip": str (path to zip),
            }
        """
        intent = project_data.get("intent", "Unknown Project")
        project_id = project_data.get("project_id", "project")
        env = project_data.get("environment", {})
        geometry = project_data.get("geometry", [])
        mass = project_data.get("mass_properties", {})
        physics = project_data.get("physics_results", {})
        manufacturing = project_data.get("manufacturing_plan", {})
        bom = project_data.get("bom", {})
        verification = project_data.get("verification", {})
        deployment = project_data.get("deployment_plan", {})
        sourced = project_data.get("sourced_components", [])
        gcode = project_data.get("gcode", "")
        generated_code = project_data.get("generated_code", {})
        dfm = project_data.get("dfm_analysis", {})
        tolerance = project_data.get("tolerance_analysis", {})

        out_dir = OUTPUT_BASE / project_id
        out_dir.mkdir(parents=True, exist_ok=True)

        generated_files: Dict[str, str] = {}
        generation_errors: List[str] = []

        # --- 1. BOM CSV ---
        try:
            bom_path = self._write_bom_csv(bom, sourced, out_dir)
            if bom_path:
                generated_files["bom.csv"] = str(bom_path)
        except Exception as e:
            generation_errors.append(f"BOM CSV: {e}")
            logger.warning(f"BOM CSV generation failed: {e}")

        # --- 2. GCode ---
        try:
            gcode_path = self._write_gcode(gcode, out_dir)
            if gcode_path:
                generated_files["firmware.gcode"] = str(gcode_path)
        except Exception as e:
            generation_errors.append(f"GCode: {e}")

        # --- 3. DFM Report ---
        try:
            dfm_path = self._write_dfm_report(dfm, manufacturing, out_dir)
            if dfm_path:
                generated_files["dfm_report.md"] = str(dfm_path)
        except Exception as e:
            generation_errors.append(f"DFM report: {e}")
            logger.warning(f"DFM report generation failed: {e}")

        # --- 4. Physics Report (with plots if matplotlib available) ---
        try:
            phys_path = self._write_physics_report(physics, mass, env, out_dir)
            if phys_path:
                generated_files["physics_report.md"] = str(phys_path)
        except Exception as e:
            generation_errors.append(f"Physics report: {e}")
            logger.warning(f"Physics report generation failed: {e}")

        # --- 5. Firmware Documentation ---
        try:
            fw_path = self._write_firmware_docs(generated_code, out_dir)
            if fw_path:
                generated_files["firmware_docs.md"] = str(fw_path)
        except Exception as e:
            generation_errors.append(f"Firmware docs: {e}")

        # --- 6. Test Protocol ---
        try:
            test_path = self._write_test_protocol(
                intent, env, physics, dfm, generated_code, manufacturing, out_dir
            )
            if test_path:
                generated_files["test_protocol.md"] = str(test_path)
        except Exception as e:
            generation_errors.append(f"Test protocol: {e}")
            logger.warning(f"Test protocol generation failed: {e}")

        # --- 7. Risk Register ---
        try:
            risk_path = self._write_risk_register(physics, dfm, tolerance, env, out_dir)
            if risk_path:
                generated_files["risk_register.md"] = str(risk_path)
        except Exception as e:
            generation_errors.append(f"Risk register: {e}")

        # --- 8. Master Report ---
        try:
            report_content = self._build_master_report(
                intent, project_id, env, geometry, mass, physics,
                manufacturing, bom, verification, deployment, sourced, generated_files
            )
            report_path = out_dir / "project_report.md"
            report_path.write_text(report_content, encoding="utf-8")
            generated_files["project_report.md"] = str(report_path)
        except Exception as e:
            generation_errors.append(f"Master report: {e}")
            report_content = f"# {intent}\n\nReport generation encountered errors: {e}"
            logger.error(f"Master report generation failed: {e}")

        # --- 9. Diagrams (2D views, schematic, assembly, workflow, architecture) ---
        try:
            diagram_paths = self.generate_diagrams(project_data, out_dir)
            generated_files.update(diagram_paths)
        except Exception as e:
            generation_errors.append(f"Diagrams: {e}")
            logger.warning(f"Diagram generation failed: {e}")

        # --- 10. ZIP Package ---
        try:
            zip_path = self._create_zip_package(out_dir, project_id, generated_files)
            generated_files[f"{project_id}_package.zip"] = str(zip_path)
        except Exception as e:
            generation_errors.append(f"ZIP package: {e}")
            logger.warning(f"ZIP packaging failed: {e}")

        logger.info(f"[DocumentAgent] Generated {len(generated_files)} files for project {project_id}")

        return {
            "status": "success" if not generation_errors else "partial",
            "document": {
                "content": report_content,
                "format": "markdown",
                "word_count": len(report_content.split()),
            },
            "files": generated_files,
            "package_zip": generated_files.get(f"{project_id}_package.zip"),
            "errors": generation_errors if generation_errors else None,
        }

    # -------------------------------------------------------------------------
    # File writers
    # -------------------------------------------------------------------------

    def _write_bom_csv(self, bom: Dict, sourced: List, out_dir: Path) -> Optional[Path]:
        """Write BOM as CSV importable into Digi-Key / JLCPCB / Octopart."""
        path = out_dir / "bom.csv"

        items = bom.get("items", bom.get("components", []))

        # Merge sourced components into BOM items if not already there
        sourced_names = {s.get("name", "") for s in sourced if isinstance(s, dict)}
        for s in sourced:
            if isinstance(s, dict) and s.get("name") not in sourced_names:
                items.append(s)

        if not items:
            return None

        with open(path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "Reference", "Quantity", "Description", "Manufacturer",
                "Part Number", "Supplier", "Supplier Part No",
                "Unit Cost (USD)", "Extended Cost (USD)",
                "Lead Time (days)", "Datasheet URL"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            for item in items:
                qty = int(item.get("quantity", item.get("qty", 1)))
                unit_cost = float(item.get("unit_cost_usd", item.get("cost", item.get("unit_price", 0))) or 0)
                writer.writerow({
                    "Reference": item.get("reference", item.get("ref", item.get("name", ""))),
                    "Quantity": qty,
                    "Description": item.get("description", item.get("name", "")),
                    "Manufacturer": item.get("manufacturer", ""),
                    "Part Number": item.get("part_number", item.get("mpn", "")),
                    "Supplier": item.get("supplier", ""),
                    "Supplier Part No": item.get("supplier_part_no", item.get("supplier_pn", "")),
                    "Unit Cost (USD)": f"{unit_cost:.2f}",
                    "Extended Cost (USD)": f"{unit_cost * qty:.2f}",
                    "Lead Time (days)": item.get("lead_time_days", ""),
                    "Datasheet URL": item.get("datasheet_url", item.get("datasheet", "")),
                })

        logger.info(f"[DocumentAgent] BOM CSV written: {len(items)} items → {path}")
        return path

    def _write_gcode(self, gcode: str, out_dir: Path) -> Optional[Path]:
        """Flush GCode from pipeline state to disk."""
        if not gcode or not gcode.strip():
            return None
        path = out_dir / "firmware.gcode"
        path.write_text(gcode, encoding="utf-8")
        logger.info(f"[DocumentAgent] GCode written: {len(gcode)} chars → {path}")
        return path

    def _write_dfm_report(self, dfm: Dict, manufacturing: Dict, out_dir: Path) -> Optional[Path]:
        """Format DFM agent output as structured Markdown report."""
        if not dfm and not manufacturing:
            return None

        path = out_dir / "dfm_report.md"
        lines = ["# Design for Manufacturability (DFM) Report", f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*", ""]

        score = dfm.get("dfm_score", dfm.get("overall_manufacturability_score"))
        if score is not None:
            grade = "A" if score >= 0.85 else "B" if score >= 0.70 else "C" if score >= 0.55 else "D"
            lines += [f"## Manufacturability Score: {score:.0%} (Grade {grade})", ""]

        process = dfm.get("process", manufacturing.get("process", ""))
        if process:
            lines += [f"**Manufacturing Process:** {process}", ""]

        # Critical issues
        critical = dfm.get("critical_issues", [])
        if critical:
            lines += ["## Critical Issues", ""]
            for issue in critical:
                if isinstance(issue, dict):
                    lines.append(f"- **{issue.get('severity', 'HIGH')}** — {issue.get('description', issue.get('message', issue))}")
                else:
                    lines.append(f"- {issue}")
            lines.append("")

        # Recommendations
        recommendations = dfm.get("recommendations", dfm.get("improvements", []))
        if recommendations:
            lines += ["## Recommendations", ""]
            for rec in recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Manufacturing details
        if manufacturing:
            lines += ["## Manufacturing Plan", ""]
            lead = manufacturing.get("lead_time_days", "N/A")
            cost = manufacturing.get("estimated_cost_usd", manufacturing.get("total_cost", "N/A"))
            lines += [
                f"- **Process:** {manufacturing.get('process', 'N/A')}",
                f"- **Lead Time:** {lead} days",
                f"- **Estimated Cost:** ${cost}",
            ]

        # Tolerance summary
        tol = dfm.get("tolerance_analysis", {})
        if tol:
            lines += ["", "## Tolerance Analysis", ""]
            lines.append(f"- Tightest Tolerance: {tol.get('tightest_mm', 'N/A')} mm")
            lines.append(f"- RSS Stack: {tol.get('rss_tolerance', 'N/A')} mm")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _write_physics_report(self, physics: Dict, mass: Dict, env: Dict, out_dir: Path) -> Optional[Path]:
        """Write physics analysis results. Generates matplotlib plots if available."""
        if not physics:
            return None

        path = out_dir / "physics_report.md"
        lines = [
            "# Physics Analysis Report",
            f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*",
            "",
            f"**Environment:** {env.get('type', 'GROUND')} | "
            f"**Mass:** {mass.get('mass_kg', 'N/A')} kg | "
            f"**Volume:** {mass.get('volume_m3', 'N/A')} m³",
            "",
        ]

        # Thermal
        thermal = physics.get("thermal", {})
        if thermal:
            status = thermal.get("status", "N/A")
            exceeded = thermal.get("exceeds_material_limit", False)
            flag = "❌ EXCEEDS LIMIT" if exceeded else "✅ Within limits"
            lines += [
                "## Thermal Analysis", "",
                f"| Parameter | Value |",
                f"|---|---|",
                f"| Max Temperature | {thermal.get('max_temperature_c', 'N/A')} °C |",
                f"| Ambient Temperature | {thermal.get('ambient_temp_c', 'N/A')} °C |",
                f"| ΔT | {thermal.get('delta_T_c', 'N/A')} °C |",
                f"| Thermal Resistance | {thermal.get('thermal_resistance_k_w', 'N/A')} K/W |",
                f"| Heat Flux | {thermal.get('heat_flux_w_m2', 'N/A')} W/m² |",
                f"| Nusselt Number | {thermal.get('nusselt_number', 'N/A')} |",
                f"| Flow Regime | {thermal.get('flow_regime', 'N/A')} |",
                f"| Solver | {thermal.get('solver_used', 'N/A')} |",
                f"| Material Limit | {thermal.get('material_max_temp_c', 'N/A')} °C |",
                f"| Safety Margin | {thermal.get('safety_margin_c', 'N/A')} °C |",
                f"| Status | {flag} |",
                "",
            ]

        # Structural
        structural = physics.get("structural", {})
        if structural:
            sf = structural.get("safety_factor_yield", structural.get("safety_factor"))
            sf_flag = "❌ UNSAFE (SF < 1.5)" if sf and float(sf) < 1.5 else "✅ Safe"
            lines += [
                "## Structural Analysis", "",
                f"| Parameter | Value |",
                f"|---|---|",
                f"| Max Stress | {structural.get('max_stress_mpa', 'N/A')} MPa |",
                f"| Yield Strength | {structural.get('yield_strength_mpa', 'N/A')} MPa |",
                f"| Safety Factor | {sf} — {sf_flag} |",
                f"| Max Displacement | {structural.get('max_displacement_mm', 'N/A')} mm |",
                f"| Analysis Type | {structural.get('analysis_type', 'static')} |",
                "",
            ]

        # Fluid
        fluid = physics.get("fluid", {})
        if fluid:
            lines += [
                "## Fluid / Aerodynamic Analysis", "",
                f"| Parameter | Value |",
                f"|---|---|",
                f"| Drag Coefficient | {fluid.get('drag_coefficient', 'N/A')} |",
                f"| Reynolds Number | {fluid.get('reynolds_number', 'N/A')} |",
                f"| Drag Force | {fluid.get('drag_force_n', 'N/A')} N |",
                f"| Flow Regime | {fluid.get('flow_regime', 'N/A')} |",
                "",
            ]

        # Try matplotlib plots
        try:
            plot_paths = self._generate_physics_plots(physics, out_dir)
            if plot_paths:
                lines += ["## Analysis Plots", ""]
                for label, ppath in plot_paths.items():
                    rel = os.path.basename(ppath)
                    lines.append(f"![{label}]({rel})")
                lines.append("")
        except Exception as e:
            logger.debug(f"Physics plots skipped: {e}")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _generate_physics_plots(self, physics: Dict, out_dir: Path) -> Dict[str, str]:
        """Generate matplotlib plots for physics results. Returns {label: filepath}."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        plots = {}

        thermal = physics.get("thermal", {})
        structural = physics.get("structural", {})

        # Thermal bar chart
        if thermal:
            t_max = thermal.get("max_temperature_c")
            t_mat = thermal.get("material_max_temp_c")
            t_amb = thermal.get("ambient_temp_c", thermal.get("ambient_temp", 25))
            if t_max is not None:
                fig, ax = plt.subplots(figsize=(6, 3))
                labels = ["Ambient", "Max Component Temp"]
                values = [float(t_amb or 25), float(t_max)]
                colors = ["#4CAF50", "#F44336" if t_mat and float(t_max) > float(t_mat) else "#FF9800"]
                bars = ax.bar(labels, values, color=colors, width=0.5)
                if t_mat:
                    ax.axhline(float(t_mat), color="red", linestyle="--", linewidth=1.5, label=f"Material limit ({t_mat}°C)")
                    ax.legend()
                ax.set_ylabel("Temperature (°C)")
                ax.set_title("Thermal Analysis Summary")
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}°C",
                            ha="center", va="bottom", fontsize=9)
                plt.tight_layout()
                plot_path = str(out_dir / "thermal_summary.png")
                plt.savefig(plot_path, dpi=100)
                plt.close()
                plots["Thermal Summary"] = plot_path

        # Structural safety factor gauge
        if structural:
            sf = structural.get("safety_factor_yield", structural.get("safety_factor"))
            stress = structural.get("max_stress_mpa")
            yield_str = structural.get("yield_strength_mpa")
            if sf and stress and yield_str:
                fig, ax = plt.subplots(figsize=(6, 3))
                categories = ["Applied Stress", "Yield Strength"]
                values = [float(stress), float(yield_str)]
                colors = ["#2196F3", "#4CAF50"]
                ax.barh(categories, values, color=colors, height=0.5)
                ax.set_xlabel("Stress (MPa)")
                ax.set_title(f"Structural Analysis — Safety Factor: {float(sf):.2f}")
                for i, val in enumerate(values):
                    ax.text(val + 1, i, f"{val:.1f} MPa", va="center", fontsize=9)
                plt.tight_layout()
                plot_path = str(out_dir / "structural_summary.png")
                plt.savefig(plot_path, dpi=100)
                plt.close()
                plots["Structural Summary"] = plot_path

        return plots

    def _write_firmware_docs(self, generated_code: Any, out_dir: Path) -> Optional[Path]:
        """Format codegen output as firmware documentation."""
        if not generated_code:
            return None

        path = out_dir / "firmware_docs.md"
        lines = ["# Firmware Documentation", f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*", ""]

        if isinstance(generated_code, dict):
            project = generated_code.get("project", generated_code)
            lang = project.get("language", generated_code.get("language", "Unknown"))
            platform = project.get("platform", generated_code.get("platform", "Unknown"))
            rtos = project.get("rtos", generated_code.get("rtos", ""))

            lines += [
                f"**Platform:** {platform}  |  **Language:** {lang}  |  **RTOS:** {rtos or 'None'}",
                "",
            ]

            # File manifest
            files = project.get("files", generated_code.get("files", {}))
            if files:
                lines += ["## Generated Files", "", "| Filename | Size |", "|---|---|"]
                for fname, content in files.items():
                    size = f"{len(content)} chars" if isinstance(content, str) else "binary"
                    lines.append(f"| `{fname}` | {size} |")
                lines.append("")

            # Pin map
            pinout = project.get("pinout", generated_code.get("pinout", {}))
            if pinout:
                lines += ["## Pin Allocation", "", "| Component | Signal | Pin |", "|---|---|---|"]
                for comp, allocs in pinout.items():
                    if isinstance(allocs, dict):
                        for signal, pin in allocs.items():
                            lines.append(f"| {comp} | {signal} | {pin} |")
                    elif isinstance(allocs, list):
                        for pin in allocs:
                            lines.append(f"| {comp} | — | {pin} |")
                lines.append("")

            # Libraries
            libs = project.get("libraries", generated_code.get("libraries", []))
            if libs:
                lines += ["## Required Libraries", ""]
                for lib in libs:
                    lines.append(f"- `{lib}`")
                lines.append("")

            # Build config
            build = project.get("build_config", generated_code.get("build_config", {}))
            if build:
                lines += ["## Build Configuration", "```yaml"]
                lines.append(json.dumps(build, indent=2))
                lines += ["```", ""]

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _write_test_protocol(
        self,
        intent: str,
        env: Dict,
        physics: Dict,
        dfm: Dict,
        generated_code: Any,
        manufacturing: Dict,
        out_dir: Path,
    ) -> Optional[Path]:
        """
        Generate dynamic test protocol based on project context.
        Tests are derived from actual physics results, environment type,
        manufacturing process, and component types — not generic strings.
        """
        path = out_dir / "test_protocol.md"
        env_type = env.get("type", "GROUND").upper()
        thermal = physics.get("thermal", {})
        structural = physics.get("structural", {})
        fluid = physics.get("fluid", {})
        process = manufacturing.get("process", "")
        has_firmware = bool(generated_code)

        lines = [
            "# Test Protocol",
            f"**Project:** {intent}",
            f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*",
            "",
            "---",
            "",
        ]

        # --- 1. Dimensional / First Article Inspection ---
        lines += ["## 1. Dimensional Verification (First Article Inspection)", ""]
        lines += ["| Check | Method | Acceptance Criteria | Pass/Fail |", "|---|---|---|---|"]
        lines += [
            "| Overall envelope | Caliper / CMM | Per drawing ±0.1 mm | [ ] |",
            "| Critical dimensions | CMM | Per tolerance spec | [ ] |",
            "| Surface finish | Profilometer | Ra ≤ specified | [ ] |",
            "| Visual inspection | Naked eye | No burrs, scratches, voids | [ ] |",
        ]
        if dfm:
            score = dfm.get("dfm_score", dfm.get("overall_manufacturability_score", 1.0))
            if score and float(score) < 0.7:
                lines.append("| DFM issues resolved | Visual + measure | All critical DFM flags cleared | [ ] |")
        lines.append("")

        # --- 2. Thermal Tests (only if thermal physics exists) ---
        if thermal:
            t_max = thermal.get("max_temperature_c")
            t_mat = thermal.get("material_max_temp_c")
            t_amb = thermal.get("ambient_temp_c", 25)
            lines += ["## 2. Thermal Testing", ""]
            lines += ["| Test | Conditions | Acceptance Criteria | Pass/Fail |", "|---|---|---|---|"]
            if t_max:
                lines.append(f"| Operating temperature | Full load, natural convection | T_junction ≤ {t_max:.0f} °C | [ ] |")
            if t_mat:
                lines.append(f"| Material thermal limit | Max rated load | Component temp < {float(t_mat)*0.9:.0f} °C (90% of limit) | [ ] |")

            # Environment-specific thermal tests
            if env_type in ("AEROSPACE", "SPACE"):
                lines.append("| Thermal vacuum cycling | -40°C to +85°C, 10 cycles | No failures, no outgassing | [ ] |")
                lines.append("| Thermal shock | -55°C ↔ +125°C, 5 min transfer | No cracks, no delamination | [ ] |")
            elif env_type == "UNDERWATER":
                lines.append("| Humidity soak | 95% RH, 40°C, 96 h | No corrosion, no condensation failure | [ ] |")
            elif env_type == "INDUSTRIAL":
                lines.append(f"| Elevated ambient | {float(t_amb or 25)+20:.0f} °C ambient, 2 h | Functional throughout | [ ] |")
            lines.append("")

        # --- 3. Structural Tests ---
        if structural:
            sf = structural.get("safety_factor_yield", 2.0)
            stress = structural.get("max_stress_mpa", 0)
            lines += ["## 3. Structural / Mechanical Testing", ""]
            lines += ["| Test | Load | Acceptance Criteria | Pass/Fail |", "|---|---|---|---|"]
            lines.append(f"| Static load | Rated load × 1.5 | No permanent deformation, SF ≥ {float(sf):.1f} | [ ] |")
            lines.append("| Proof load | Rated load × 2.0 | No fracture | [ ] |")
            if env_type in ("AEROSPACE", "SPACE", "INDUSTRIAL"):
                lines.append("| Vibration (sinusoidal) | Per MIL-STD-810 / ECSS-E-ST-10C | No loose fasteners, no failures | [ ] |")
                lines.append("| Random vibration | 20–2000 Hz, per spec PSD | No failures after 1 h | [ ] |")
            if env_type == "SPACE":
                lines.append("| Shock | 100g, 11 ms half-sine | Functional after test | [ ] |")
            lines.append("")

        # --- 4. Fluid / Aerodynamic Tests ---
        if fluid:
            cd = fluid.get("drag_coefficient")
            lines += ["## 4. Aerodynamic / Fluid Testing", ""]
            lines += ["| Test | Method | Acceptance Criteria | Pass/Fail |", "|---|---|---|---|"]
            if cd:
                lines.append(f"| Drag coefficient | Wind tunnel / CFD correlation | Cd ≤ {float(cd)*1.1:.3f} (10% margin) | [ ] |")
            lines.append("| Flow uniformity | Flow visualisation | No separation above rated speed | [ ] |")
            if env_type == "UNDERWATER":
                lines.append("| Pressure integrity | Hydrostatic, rated depth × 1.5 | No leakage after 30 min | [ ] |")
            lines.append("")

        # --- 5. Firmware / HIL Tests ---
        if has_firmware:
            lines += ["## 5. Firmware & Hardware-in-the-Loop (HIL) Testing", ""]
            lines += ["| Test | Method | Acceptance Criteria | Pass/Fail |", "|---|---|---|---|"]
            lines += [
                "| Power-on self-test (POST) | Apply power | All OK status within 5 s | [ ] |",
                "| Sensor read accuracy | Known reference signal | Within ±2% of reference | [ ] |",
                "| Actuator command response | Command + oscilloscope | Latency < 10 ms | [ ] |",
                "| Communication bus | Protocol analyser | No framing errors, 100 transactions | [ ] |",
                "| Watchdog recovery | Force hang via debug | Reset within watchdog period | [ ] |",
                "| Flash/erase endurance | 100 write cycles | Data integrity maintained | [ ] |",
            ]
            if env_type in ("AEROSPACE", "SPACE"):
                lines.append("| Radiation tolerance | TID test or SEU injection | No permanent latch-up | [ ] |")
            lines.append("")

        # --- 6. Manufacturing Process Tests ---
        if process:
            lines += [f"## 6. Process-Specific Testing ({process})", ""]
            lines += ["| Test | Method | Acceptance Criteria | Pass/Fail |", "|---|---|---|---|"]
            proc_lower = process.lower()
            if "cnc" in proc_lower or "machining" in proc_lower:
                lines += [
                    "| Tool marks / finish | Profilometer | Ra per drawing | [ ] |",
                    "| Feature positions | CMM | ±0.05 mm from nominal | [ ] |",
                    "| Thread gauging | Go/no-go gauges | All threads within tolerance | [ ] |",
                ]
            elif "fdm" in proc_lower or "3d print" in proc_lower or "additive" in proc_lower:
                lines += [
                    "| Layer adhesion | Tensile coupon | ≥ 80% of bulk material strength | [ ] |",
                    "| Dimensional accuracy | Caliper | ±0.3 mm on critical dims | [ ] |",
                    "| Porosity | Visual / dye penetrant | No open pores on sealing surfaces | [ ] |",
                ]
            elif "injection" in proc_lower:
                lines += [
                    "| Mould trial (T1) | Full shot | No flash, sink marks, shorts | [ ] |",
                    "| Warpage | Flatness gauge | ≤ 0.5 mm over 100 mm | [ ] |",
                    "| Gate vestige | Visual | ≤ 0.5 mm protrusion | [ ] |",
                ]
            elif "casting" in proc_lower:
                lines += [
                    "| X-ray inspection | Industrial CT / X-ray | No internal voids > 1 mm | [ ] |",
                    "| Porosity | Dye penetrant | No surface-breaking defects | [ ] |",
                ]
            lines.append("")

        # --- 7. Acceptance Sign-off ---
        lines += [
            "## 7. Acceptance Sign-off",
            "",
            "| Role | Name | Date | Signature |",
            "|---|---|---|---|",
            "| Design Engineer | | | |",
            "| Quality Engineer | | | |",
            "| Manufacturing Engineer | | | |",
            "| Customer / End User | | | |",
        ]

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _write_risk_register(
        self, physics: Dict, dfm: Dict, tolerance: Dict, env: Dict, out_dir: Path
    ) -> Optional[Path]:
        """Generate FMEA-style risk register from actual physics and DFM data."""
        path = out_dir / "risk_register.md"
        env_type = env.get("type", "GROUND").upper()

        lines = [
            "# Risk Register (FMEA)",
            f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*",
            "",
            "Severity (S): 1=negligible → 10=catastrophic  |  "
            "Occurrence (O): 1=unlikely → 10=certain  |  "
            "Detection (D): 1=easy to detect → 10=undetectable  |  "
            "RPN = S × O × D",
            "",
            "| # | Failure Mode | Effect | S | O | D | RPN | Mitigation | Status |",
            "|---|---|---|---|---|---|---|---|---|",
        ]

        risks = []
        row = 1

        # Thermal risks from physics
        thermal = physics.get("thermal", {})
        if thermal:
            exceeded = thermal.get("exceeds_material_limit", False)
            margin = thermal.get("safety_margin_c")
            s = 8 if exceeded else (6 if margin and float(margin) < 20 else 4)
            o = 6 if exceeded else 3
            d = 4
            rpn = s * o * d
            mitigation = "Add heatsink / increase airflow" if exceeded else "Monitor junction temperature in firmware"
            risks.append([row, "Thermal runaway / overtemperature",
                          "Component damage, fire risk", s, o, d, rpn, mitigation, "[ ] Open"])
            row += 1

        # Structural risks
        structural = physics.get("structural", {})
        if structural:
            sf = float(structural.get("safety_factor_yield", structural.get("safety_factor", 2.0)) or 2.0)
            s = 9 if sf < 1.0 else (7 if sf < 1.5 else 4)
            o = 4 if sf < 1.5 else 2
            d = 3
            rpn = s * o * d
            risks.append([row, "Structural overload / fracture",
                          f"Loss of structural integrity (SF={sf:.1f})", s, o, d, rpn,
                          "Increase cross-section or use higher-strength material", "[ ] Open"])
            row += 1

        # DFM risks
        if dfm:
            critical = dfm.get("critical_issues", [])
            for issue in critical[:3]:
                desc = issue.get("description", str(issue)) if isinstance(issue, dict) else str(issue)
                risks.append([row, f"Manufacturing defect: {desc[:50]}",
                              "Non-conformance, scrap", 5, 4, 3, 60,
                              "Review design per DFM recommendations", "[ ] Open"])
                row += 1

        # Tolerance risks
        if tolerance:
            wc = tolerance.get("worst_case", {})
            if wc:
                upper = wc.get("upper_limit")
                lower = wc.get("lower_limit")
                if upper and lower:
                    risks.append([row, "Tolerance stack-up out of spec",
                                  "Assembly interference or excessive clearance",
                                  5, 3, 4, 60,
                                  "Tighten critical tolerances or use selective assembly", "[ ] Open"])
                    row += 1

        # Environment-specific risks
        if env_type in ("AEROSPACE", "SPACE"):
            risks.append([row, "Outgassing in vacuum", "Contamination of optical surfaces", 6, 3, 5, 90,
                          "Use space-qualified materials (ASTM E595)", "[ ] Open"])
            row += 1
            risks.append([row, "Radiation-induced latch-up", "Processor hang / damage", 8, 3, 6, 144,
                          "Select rad-tolerant ICs, add latch-up protection", "[ ] Open"])
            row += 1
        elif env_type == "UNDERWATER":
            risks.append([row, "Seal failure / flooding", "Total loss of electronics", 9, 3, 4, 108,
                          "Double O-ring seals, pressure test at 1.5× rated depth", "[ ] Open"])
            row += 1
        elif env_type == "INDUSTRIAL":
            risks.append([row, "EMI / EMC interference", "Control system malfunction", 6, 4, 4, 96,
                          "Shielded enclosure, filtered power supply, CE/FCC testing", "[ ] Open"])
            row += 1

        for risk in sorted(risks, key=lambda r: -r[6]):  # sort by RPN desc
            lines.append(f"| {risk[0]} | {risk[1]} | {risk[2]} | {risk[3]} | {risk[4]} | {risk[5]} | **{risk[6]}** | {risk[7]} | {risk[8]} |")

        lines += [
            "",
            "---",
            f"*Total risks identified: {len(risks)}  |  High RPN (>100): {sum(1 for r in risks if r[6] > 100)}*",
        ]

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _build_master_report(
        self, intent, project_id, env, geometry, mass, physics,
        manufacturing, bom, verification, deployment, sourced, generated_files
    ) -> str:
        """Assemble the master project report linking all generated documents."""
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        env_type = env.get("type", "GROUND")

        sections = [
            f"# BRICK OS — Project Deliverable Package",
            f"**Project:** {intent}",
            f"**Project ID:** {project_id}",
            f"**Generated:** {now}",
            f"**Environment:** {env_type}",
            "",
            "---",
            "",
            "## Deliverable Files",
            "",
            "| Document | File | Description |",
            "|---|---|---|",
        ]

        file_descriptions = {
            "bom.csv": "Bill of Materials — importable into Digi-Key, JLCPCB, Octopart",
            "firmware.gcode": "GCode toolpath file for CNC / additive manufacturing",
            "dfm_report.md": "Design for Manufacturability analysis and recommendations",
            "physics_report.md": "Thermal, structural, and fluid analysis results with plots",
            "firmware_docs.md": "Pin map, build instructions, generated firmware file manifest",
            "test_protocol.md": "Acceptance test cases derived from physics results and environment",
            "risk_register.md": "FMEA-style risk register with RPN scores and mitigations",
            "2d_views.png": "Orthographic engineering views — top, front, side projections",
            "electronics_schematic.png": "Electronics system block diagram with signal and power buses",
            "assembly_sequence.png": "Step-by-step assembly sequence from BOM",
            "manufacturing_workflow.png": "Manufacturing process flow diagram",
            "system_architecture.png": "High-level system architecture showing all subsystems",
        }

        for fname, fpath in generated_files.items():
            if fname.endswith(".zip"):
                continue
            desc = file_descriptions.get(fname, "Generated document")
            sections.append(f"| {fname.replace('_', ' ').replace('.md','').replace('.csv','').title()} | `{fname}` | {desc} |")

        zip_file = next((f for f in generated_files if f.endswith(".zip")), None)
        if zip_file:
            sections += ["", f"📦 **Download Package:** `{zip_file}`", ""]

        # Summary stats
        sections += ["", "---", "", "## Summary", ""]

        if geometry:
            n = len(geometry) if isinstance(geometry, list) else "—"
            sections.append(f"- **Geometry nodes:** {n}")
        if mass:
            sections.append(f"- **Mass:** {mass.get('mass_kg', 'N/A')} kg")

        thermal = physics.get("thermal", {})
        structural = physics.get("structural", {})
        if thermal:
            flag = "⚠️ EXCEEDS LIMIT" if thermal.get("exceeds_material_limit") else "✅ Thermal OK"
            sections.append(f"- **Thermal:** {thermal.get('max_temperature_c', 'N/A')} °C max — {flag}")
        if structural:
            sf = structural.get("safety_factor_yield", structural.get("safety_factor"))
            flag = "⚠️ LOW SF" if sf and float(sf) < 1.5 else "✅ Structural OK"
            sections.append(f"- **Structural:** SF = {sf} — {flag}")

        if bom:
            total = bom.get("total_cost_usd", bom.get("total_cost", "N/A"))
            n_items = len(bom.get("items", bom.get("components", [])))
            sections.append(f"- **BOM:** {n_items} items, total ${total}")

        if verification:
            passed = verification.get("passed", verification.get("status") == "pass")
            sections.append(f"- **Verification:** {'✅ PASSED' if passed else '❌ FAILED'}")

        return "\n".join(sections)

    def _create_zip_package(
        self, out_dir: Path, project_id: str, generated_files: Dict[str, str]
    ) -> Path:
        """ZIP all generated files into a delivery package."""
        zip_path = out_dir / f"{project_id}_package.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, fpath in generated_files.items():
                if fname.endswith(".zip"):
                    continue
                p = Path(fpath)
                if p.exists():
                    zf.write(p, arcname=fname)
        logger.info(f"[DocumentAgent] Package ZIP: {zip_path} ({zip_path.stat().st_size // 1024} KB)")
        return zip_path

    # =========================================================================
    # DIAGRAMS
    # =========================================================================

    def generate_diagrams(self, project_data: Dict[str, Any], out_dir: Path) -> Dict[str, str]:
        """
        Generate all diagrams for the project and return {filename: filepath}.
        Dispatches to the appropriate generators based on available data.
        """
        import matplotlib
        matplotlib.use("Agg")

        diagram_files: Dict[str, str] = {}
        geometry = project_data.get("geometry", project_data.get("geometry_tree", []))
        components = project_data.get("components", project_data.get("electronics", {}).get("components", []))
        bom = project_data.get("bom", {})
        manufacturing = project_data.get("manufacturing_plan", {})
        generated_code = project_data.get("generated_code", {})

        # 1 — 2D engineering views (top / front / side)
        try:
            paths = self._draw_2d_views(geometry, out_dir)
            diagram_files.update(paths)
        except Exception as e:
            logger.warning(f"2D views failed: {e}")

        # 2 — Electronics schematic / block diagram
        if components or generated_code:
            try:
                path = self._draw_electronics_schematic(components, generated_code, bom, out_dir)
                if path:
                    diagram_files["electronics_schematic.png"] = path
            except Exception as e:
                logger.warning(f"Electronics schematic failed: {e}")

        # 3 — Assembly sequence diagram
        bom_items = bom.get("items", bom.get("components", []))
        if bom_items:
            try:
                path = self._draw_assembly_sequence(bom_items, manufacturing, out_dir)
                if path:
                    diagram_files["assembly_sequence.png"] = path
            except Exception as e:
                logger.warning(f"Assembly sequence failed: {e}")

        # 4 — Manufacturing / process workflow
        if manufacturing:
            try:
                path = self._draw_manufacturing_workflow(manufacturing, out_dir)
                if path:
                    diagram_files["manufacturing_workflow.png"] = path
            except Exception as e:
                logger.warning(f"Manufacturing workflow failed: {e}")

        # 5 — System architecture overview
        try:
            path = self._draw_system_architecture(project_data, out_dir)
            if path:
                diagram_files["system_architecture.png"] = path
        except Exception as e:
            logger.warning(f"System architecture failed: {e}")

        return diagram_files

    # -------------------------------------------------------------------------
    # 2D engineering views
    # -------------------------------------------------------------------------

    def _draw_2d_views(self, geometry: List, out_dir: Path) -> Dict[str, str]:
        """Draw orthographic top/front/side projections of the geometry tree."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyArrowPatch

        if not geometry:
            geometry = [{"type": "box", "dimensions": {"length": 1.0, "width": 0.5, "height": 0.3}}]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.patch.set_facecolor("#1a1a2e")
        view_titles = ["TOP VIEW  (X–Y)", "FRONT VIEW  (X–Z)", "SIDE VIEW  (Y–Z)"]
        colors = ["#00d4ff", "#00ff9f", "#ff6b6b", "#ffd93d", "#c77dff"]

        for ax, title in zip(axes, view_titles):
            ax.set_facecolor("#16213e")
            ax.set_title(title, color="white", fontsize=10, pad=8, fontfamily="monospace")
            ax.tick_params(colors="gray", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")
            ax.set_aspect("equal")
            ax.grid(True, color="#2a2a4a", linewidth=0.5, linestyle="--")

        x_min = y_min = z_min = 0.0
        x_max = y_max = z_max = 0.01  # avoid zero-size axes

        for i, node in enumerate(geometry):
            color = colors[i % len(colors)]
            dims = (
                node.get("dimensions") or
                node.get("params") or
                node.get("dim") or {}
            )
            L = float(dims.get("length", dims.get("l", dims.get("x", 0.1))))
            W = float(dims.get("width",  dims.get("w", dims.get("y", 0.05))))
            H = float(dims.get("height", dims.get("h", dims.get("z", 0.05))))
            ox = float(node.get("offset_x", node.get("x", 0.0)))
            oy = float(node.get("offset_y", node.get("y", 0.0)))
            oz = float(node.get("offset_z", node.get("z", 0.0)))
            label = node.get("name", node.get("type", f"part_{i+1}"))
            geo_type = node.get("type", "box").lower()

            x_max = max(x_max, ox + L)
            y_max = max(y_max, oy + W)
            z_max = max(z_max, oz + H)

            alpha = 0.55
            lw = 1.2

            if geo_type in ("cylinder", "circle", "rod"):
                from matplotlib.patches import Ellipse, Circle
                # Top view: circle (diameter = W)
                axes[0].add_patch(Circle((ox + L/2, oy + W/2), W/2, fill=True,
                    facecolor=color, alpha=alpha, edgecolor=color, linewidth=lw))
                # Front view: rectangle (L x H)
                axes[1].add_patch(mpatches.Rectangle((ox, oz), L, H, fill=True,
                    facecolor=color, alpha=alpha, edgecolor=color, linewidth=lw))
                # Side view: circle
                axes[2].add_patch(Circle((oy + W/2, oz + H/2), W/2, fill=True,
                    facecolor=color, alpha=alpha, edgecolor=color, linewidth=lw))
            else:
                # Default: box
                axes[0].add_patch(mpatches.Rectangle((ox, oy), L, W, fill=True,
                    facecolor=color, alpha=alpha, edgecolor=color, linewidth=lw))
                axes[1].add_patch(mpatches.Rectangle((ox, oz), L, H, fill=True,
                    facecolor=color, alpha=alpha, edgecolor=color, linewidth=lw))
                axes[2].add_patch(mpatches.Rectangle((oy, oz), W, H, fill=True,
                    facecolor=color, alpha=alpha, edgecolor=color, linewidth=lw))

            # Label on top view
            axes[0].text(ox + L/2, oy + W/2, label[:10], color="white",
                ha="center", va="center", fontsize=6, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#00000080", linewidth=0))

        pad = max(x_max, y_max, z_max) * 0.12 + 0.01
        axes[0].set_xlim(x_min - pad, x_max + pad); axes[0].set_ylim(y_min - pad, y_max + pad)
        axes[1].set_xlim(x_min - pad, x_max + pad); axes[1].set_ylim(z_min - pad, z_max + pad)
        axes[2].set_xlim(y_min - pad, y_max + pad); axes[2].set_ylim(z_min - pad, z_max + pad)
        axes[0].set_xlabel("X (m)", color="gray", fontsize=8); axes[0].set_ylabel("Y (m)", color="gray", fontsize=8)
        axes[1].set_xlabel("X (m)", color="gray", fontsize=8); axes[1].set_ylabel("Z (m)", color="gray", fontsize=8)
        axes[2].set_xlabel("Y (m)", color="gray", fontsize=8); axes[2].set_ylabel("Z (m)", color="gray", fontsize=8)

        # Dimension annotations on front view
        ax = axes[1]
        arrow_kw = dict(arrowstyle="<->", color="#aaaaaa", lw=0.8)
        if x_max > 0.01:
            ax.annotate("", xy=(x_max, z_min - pad*0.6), xytext=(x_min, z_min - pad*0.6),
                        arrowprops=dict(arrowstyle="<->", color="#aaaaaa", lw=0.8))
            ax.text((x_min + x_max)/2, z_min - pad*0.75, f"{x_max - x_min:.3f} m",
                    color="#aaaaaa", ha="center", va="top", fontsize=7, fontfamily="monospace")

        fig.suptitle("Orthographic Engineering Views", color="white", fontsize=12,
                     fontfamily="monospace", y=1.01)
        plt.tight_layout()
        path = str(out_dir / "2d_views.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"[DocumentAgent] 2D views written: {path}")
        return {"2d_views.png": path}

    # -------------------------------------------------------------------------
    # Electronics schematic / block diagram
    # -------------------------------------------------------------------------

    def _draw_electronics_schematic(
        self,
        components: List[Dict],
        generated_code: Any,
        bom: Dict,
        out_dir: Path,
    ) -> Optional[str]:
        """
        Draw a functional block diagram / schematic of the electronics system.
        Nodes = components. Edges = power/signal buses inferred from interface types.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx

        # Build component list from multiple sources
        comp_list = list(components) if components else []

        if not comp_list and isinstance(generated_code, dict):
            for c in generated_code.get("components", []):
                comp_list.append(c)
        if not comp_list:
            for item in bom.get("items", bom.get("components", [])):
                comp_list.append({"name": item.get("name", "?"), "type": "component",
                                  "interface": "—"})

        if not comp_list:
            return None

        # Assign roles and colors
        role_colors = {
            "microcontroller": "#00d4ff", "mcu": "#00d4ff", "processor": "#00d4ff",
            "sensor": "#00ff9f", "imu": "#00ff9f", "gps": "#00ff9f", "temperature": "#00ff9f",
            "motor": "#ff6b6b", "motor_driver": "#ff9f6b", "esc": "#ff9f6b", "actuator": "#ff6b6b",
            "power": "#ffd93d", "battery": "#ffd93d", "regulator": "#ffd93d", "pmic": "#ffd93d",
            "display": "#c77dff", "led": "#c77dff",
            "communication": "#74b9ff", "wifi": "#74b9ff", "bluetooth": "#74b9ff", "uart": "#74b9ff",
            "storage": "#a8e6cf",
        }

        def get_color(comp):
            t = (comp.get("type") or comp.get("category") or "").lower()
            for key, col in role_colors.items():
                if key in t:
                    return col
            return "#888899"

        G = nx.DiGraph()
        node_colors = []
        node_labels = {}

        # Always add a power bus and MCU if we can find them
        mcu_node = None
        power_node = "PWR BUS"
        G.add_node(power_node)
        node_colors_map = {power_node: "#ffd93d"}
        node_labels[power_node] = "PWR\nBUS"

        for i, comp in enumerate(comp_list):
            name = comp.get("name", comp.get("reference", f"U{i+1}"))
            short = name[:12]
            nid = f"{short}_{i}"
            G.add_node(nid)
            node_colors_map[nid] = get_color(comp)
            t = (comp.get("type") or "").lower()
            iface = comp.get("interface", "")
            node_labels[nid] = f"{short}\n{iface}" if iface and iface != "—" else short

            # Power edges: everything gets power from bus
            G.add_edge(power_node, nid, label="VCC")

            # Detect MCU
            if any(k in t for k in ("mcu", "microcontroller", "processor", "controller", "arduino", "esp", "stm")):
                mcu_node = nid

        # Signal edges: MCU → sensors/actuators
        if mcu_node:
            for nid in list(G.nodes):
                if nid == mcu_node or nid == power_node:
                    continue
                comp_idx = None
                for i, c in enumerate(comp_list):
                    if f"{c.get('name','')[:12]}_{i}" == nid:
                        comp_idx = i
                        break
                if comp_idx is None:
                    continue
                comp = comp_list[comp_idx]
                t = (comp.get("type") or "").lower()
                iface = (comp.get("interface") or "").upper()
                if any(k in t for k in ("sensor", "imu", "gps", "camera", "display")):
                    G.add_edge(mcu_node, nid, label=iface or "DATA")
                elif any(k in t for k in ("motor", "actuator", "esc", "servo", "led")):
                    G.add_edge(mcu_node, nid, label=iface or "PWM")
                elif any(k in t for k in ("comm", "wifi", "bluetooth", "radio", "uart")):
                    G.add_edge(mcu_node, nid, label=iface or "UART")

        # Layout
        nodes = list(G.nodes)
        nc = [node_colors_map.get(n, "#888899") for n in nodes]

        fig, ax = plt.subplots(figsize=(max(10, len(nodes) * 1.4), 7))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        ax.set_title("Electronics System Block Diagram / Schematic", color="white",
                     fontsize=12, fontfamily="monospace", pad=12)
        ax.axis("off")

        try:
            pos = nx.shell_layout(G, nlist=[
                [power_node],
                [mcu_node] if mcu_node else [],
                [n for n in nodes if n not in (power_node, mcu_node)],
            ])
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=2.5)

        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=nc,
                               node_size=2200, ax=ax, alpha=0.92)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7,
                                font_color="black", font_weight="bold", ax=ax)

        edge_labels = nx.get_edge_attributes(G, "label")
        pwr_edges = [(u, v) for u, v in G.edges if u == power_node]
        sig_edges = [(u, v) for u, v in G.edges if u != power_node]
        nx.draw_networkx_edges(G, pos, edgelist=pwr_edges, edge_color="#ffd93d",
                               arrows=True, arrowsize=15, width=1.5,
                               style="dashed", ax=ax, alpha=0.7)
        nx.draw_networkx_edges(G, pos, edgelist=sig_edges, edge_color="#00d4ff",
                               arrows=True, arrowsize=15, width=1.8, ax=ax, alpha=0.8)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_size=6, font_color="#cccccc", ax=ax)

        # Legend
        legend_items = [
            mpatches.Patch(color="#ffd93d", label="Power (dashed)"),
            mpatches.Patch(color="#00d4ff", label="Signal/Data"),
            mpatches.Patch(color="#00ff9f", label="Sensor"),
            mpatches.Patch(color="#ff6b6b", label="Actuator/Motor"),
            mpatches.Patch(color="#c77dff", label="Display/Output"),
        ]
        ax.legend(handles=legend_items, loc="lower right", fontsize=8,
                  facecolor="#161b22", edgecolor="#444", labelcolor="white")

        plt.tight_layout()
        path = str(out_dir / "electronics_schematic.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"[DocumentAgent] Electronics schematic written: {path}")
        return path

    # -------------------------------------------------------------------------
    # Assembly sequence
    # -------------------------------------------------------------------------

    def _draw_assembly_sequence(
        self, bom_items: List[Dict], manufacturing: Dict, out_dir: Path
    ) -> Optional[str]:
        """
        Draw a step-by-step assembly sequence from the BOM.
        Each step is a numbered box showing the component being installed.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if not bom_items:
            return None

        # Sort items: structural first, then electronics, then fasteners
        def sort_key(item):
            name = (item.get("description") or item.get("name") or "").lower()
            if any(k in name for k in ("frame", "housing", "body", "chassis", "bracket")):
                return 0
            if any(k in name for k in ("pcb", "board", "controller", "mcu")):
                return 1
            if any(k in name for k in ("motor", "actuator", "servo", "pump")):
                return 2
            if any(k in name for k in ("sensor", "camera", "imu", "gps")):
                return 3
            if any(k in name for k in ("wire", "cable", "connector", "harness")):
                return 4
            if any(k in name for k in ("screw", "bolt", "nut", "washer", "fastener")):
                return 5
            return 3

        items = sorted(bom_items, key=sort_key)
        steps = items[:12]  # cap at 12 steps for readability

        cols = 4
        rows = (len(steps) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.5))
        fig.patch.set_facecolor("#0d1117")
        fig.suptitle("Assembly Sequence", color="white", fontsize=13,
                     fontfamily="monospace", y=1.01)

        if rows == 1:
            axes = [axes] if cols == 1 else list(axes)
        else:
            axes = [ax for row in axes for ax in row]

        step_colors = ["#00d4ff", "#00ff9f", "#ff9f6b", "#c77dff",
                       "#ffd93d", "#74b9ff", "#ff6b6b", "#a8e6cf",
                       "#fdcb6e", "#6c5ce7", "#55efc4", "#fd79a8"]

        for i, (ax, item) in enumerate(zip(axes, steps)):
            ax.set_facecolor("#161b22")
            for spine in ax.spines.values():
                spine.set_edgecolor(step_colors[i % len(step_colors)])
                spine.set_linewidth(1.5)

            name = item.get("description", item.get("name", f"Component {i+1}"))
            qty  = item.get("quantity", item.get("qty", 1))
            pn   = item.get("part_number", item.get("mpn", ""))
            mfr  = item.get("manufacturer", "")

            # Step number badge
            ax.text(0.07, 0.88, f"{i+1:02d}", transform=ax.transAxes,
                    color=step_colors[i % len(step_colors)], fontsize=18,
                    fontfamily="monospace", fontweight="bold", va="top")

            # Component name
            ax.text(0.5, 0.62, name[:22], transform=ax.transAxes,
                    color="white", fontsize=8.5, fontfamily="monospace",
                    ha="center", va="center", fontweight="bold",
                    wrap=True)
            if len(name) > 22:
                ax.text(0.5, 0.46, name[22:44], transform=ax.transAxes,
                        color="white", fontsize=8, fontfamily="monospace",
                        ha="center", va="center")

            # Qty / PN
            detail = f"Qty: {qty}"
            if pn:
                detail += f"  |  {pn[:14]}"
            ax.text(0.5, 0.20, detail, transform=ax.transAxes,
                    color="#888899", fontsize=7, ha="center", va="center",
                    fontfamily="monospace")
            if mfr:
                ax.text(0.5, 0.08, mfr[:20], transform=ax.transAxes,
                        color="#555566", fontsize=6.5, ha="center",
                        fontfamily="monospace")

            ax.axis("off")

        # Hide unused subplot panels
        for ax in axes[len(steps):]:
            ax.set_visible(False)

        plt.tight_layout()
        path = str(out_dir / "assembly_sequence.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"[DocumentAgent] Assembly sequence written: {path}")
        return path

    # -------------------------------------------------------------------------
    # Manufacturing workflow
    # -------------------------------------------------------------------------

    def _draw_manufacturing_workflow(
        self, manufacturing: Dict, out_dir: Path
    ) -> Optional[str]:
        """
        Draw a process flow diagram for the manufacturing plan.
        Steps are inferred from process type and BOM components.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

        process = (manufacturing.get("process") or manufacturing.get("primary_process") or "").lower()

        # Build steps based on process type
        if "cnc" in process or "machining" in process or "milling" in process:
            steps = [
                ("Design", "CAD model\n& DXF export", "#00d4ff"),
                ("Toolpath", "CAM programming\nG-code generation", "#74b9ff"),
                ("Setup", "Fixture & clamp\nstock material", "#ffd93d"),
                ("Rough", "Roughing pass\n(~70% stock removal)", "#ff9f6b"),
                ("Finish", "Finishing pass\n(Ra target)", "#00ff9f"),
                ("Inspect", "CMM / caliper\ndimensional check", "#c77dff"),
                ("Deburr", "Deburr & clean\nsurface treatment", "#a8e6cf"),
                ("Deliver", "QC sign-off\n& packaging", "#00d4ff"),
            ]
        elif "fdm" in process or "3d" in process or "additive" in process or "print" in process:
            steps = [
                ("Slice", "Slicer software\nlayer & support gen", "#00d4ff"),
                ("Print", "FDM / SLA\nprint job", "#74b9ff"),
                ("Cool", "Controlled cool-down\n(avoid warpage)", "#ffd93d"),
                ("Remove", "Support removal\n& cleanup", "#ff9f6b"),
                ("Post", "Post-processing\n(sand, prime)", "#00ff9f"),
                ("Inspect", "Dimensional check\nwarp assessment", "#c77dff"),
                ("Deliver", "QC sign-off\n& packaging", "#00d4ff"),
            ]
        elif "injection" in process or "moulding" in process or "molding" in process:
            steps = [
                ("Tooling", "Mould design\n& machining", "#00d4ff"),
                ("Trial T1", "First article\ntrial shot", "#74b9ff"),
                ("Adjust", "Mould adjustment\n& optimise", "#ffd93d"),
                ("Qualify", "Process qualification\nCpk ≥ 1.33", "#ff9f6b"),
                ("Produce", "Volume production\nshots", "#00ff9f"),
                ("Inspect", "AQL sampling\ndimensional", "#c77dff"),
                ("Deliver", "Pack & ship", "#00d4ff"),
            ]
        elif "assembly" in process or "welding" in process:
            steps = [
                ("Procure", "Parts procurement\n& incoming QC", "#00d4ff"),
                ("Prep", "Part prep\n& surface clean", "#74b9ff"),
                ("Sub-assy", "Sub-assembly\n& alignment", "#ffd93d"),
                ("Join", "Welding / fastening\n& bonding", "#ff9f6b"),
                ("Inspect", "Weld inspection\nNDT if required", "#c77dff"),
                ("Test", "Functional test\n& leak check", "#00ff9f"),
                ("Deliver", "Final QC\n& packaging", "#00d4ff"),
            ]
        else:
            steps = [
                ("Plan", "Design review\n& BOM release", "#00d4ff"),
                ("Procure", "Material & parts\nprocurement", "#74b9ff"),
                ("Fabricate", "Primary\nfabrication", "#ffd93d"),
                ("Assemble", "Assembly\n& integration", "#ff9f6b"),
                ("Test", "Verification\n& validation", "#c77dff"),
                ("Deliver", "QC sign-off\n& delivery", "#00d4ff"),
            ]

        n = len(steps)
        fig, ax = plt.subplots(figsize=(max(14, n * 1.9), 4.5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        ax.set_xlim(-0.5, n * 2.2 - 0.5)
        ax.set_ylim(-1.2, 2.2)
        ax.axis("off")
        ax.set_title(f"Manufacturing Workflow  —  {process.upper() or 'GENERAL'}",
                     color="white", fontsize=12, fontfamily="monospace", pad=10)

        box_w, box_h = 1.7, 1.1
        for i, (label, detail, color) in enumerate(steps):
            x = i * 2.2
            # Box
            rect = FancyBboxPatch((x - box_w/2, -box_h/2), box_w, box_h,
                                  boxstyle="round,pad=0.05",
                                  facecolor=color + "22", edgecolor=color,
                                  linewidth=1.8)
            ax.add_patch(rect)

            # Step label
            ax.text(x, 0.25, label, color=color, ha="center", va="center",
                    fontsize=10, fontweight="bold", fontfamily="monospace")
            # Detail text
            ax.text(x, -0.22, detail, color="#cccccc", ha="center", va="center",
                    fontsize=7, fontfamily="monospace", multialignment="center")

            # Step number circle
            circle = plt.Circle((x, 0.82), 0.22, color=color, zorder=3)
            ax.add_patch(circle)
            ax.text(x, 0.82, str(i+1), color="#0d1117", ha="center", va="center",
                    fontsize=9, fontweight="bold", fontfamily="monospace", zorder=4)

            # Arrow to next
            if i < n - 1:
                ax.annotate("", xy=(x + box_w/2 + 0.27, 0),
                            xytext=(x + box_w/2 + 0.01, 0),
                            arrowprops=dict(arrowstyle="-|>", color="#555566",
                                          lw=1.5, mutation_scale=14))

        # Lead time bar if available
        lead = manufacturing.get("lead_time_days")
        if lead:
            ax.text(n * 2.2 / 2 - 0.5, -0.92, f"Estimated Lead Time: {lead} days",
                    color="#888899", ha="center", va="center", fontsize=8,
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22",
                              edgecolor="#444", linewidth=1))

        plt.tight_layout()
        path = str(out_dir / "manufacturing_workflow.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"[DocumentAgent] Manufacturing workflow written: {path}")
        return path

    # -------------------------------------------------------------------------
    # System architecture overview
    # -------------------------------------------------------------------------

    def _draw_system_architecture(
        self, project_data: Dict[str, Any], out_dir: Path
    ) -> Optional[str]:
        """
        Draw a high-level system architecture block diagram showing all major
        subsystems (Mechanical, Thermal, Electronics, Power, Software) and
        their interfaces — derived from what's actually in the project data.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch
        import networkx as nx

        subsystems = {}

        # Mechanical subsystem
        geometry = project_data.get("geometry", project_data.get("geometry_tree", []))
        if geometry:
            subsystems["Mechanical\nStructure"] = {
                "color": "#00d4ff",
                "detail": f"{len(geometry)} geometry nodes",
            }

        # Thermal subsystem
        thermal = (project_data.get("physics_results") or {}).get("thermal", {})
        if thermal:
            t_max = thermal.get("max_temperature_c", "?")
            subsystems["Thermal\nSystem"] = {
                "color": "#ff9f6b",
                "detail": f"T_max={t_max}°C",
            }

        # Electronics subsystem
        components = project_data.get("components", [])
        generated_code = project_data.get("generated_code", {})
        if components or generated_code:
            n = len(components) if components else "?"
            subsystems["Electronics\n& Firmware"] = {
                "color": "#00ff9f",
                "detail": f"{n} components",
            }

        # Power subsystem
        bom_items = project_data.get("bom", {}).get("items", [])
        pwr = next((i for i in bom_items
                    if any(k in (i.get("name","") or i.get("description","")).lower()
                           for k in ("battery","power","psu","lipo","charger","bec","regulator"))), None)
        if pwr:
            subsystems["Power\nSubsystem"] = {
                "color": "#ffd93d",
                "detail": (pwr.get("name") or pwr.get("description", ""))[:18],
            }

        # Manufacturing subsystem
        mfg = project_data.get("manufacturing_plan", {})
        if mfg:
            proc = mfg.get("process", mfg.get("primary_process", ""))
            subsystems["Manufacturing\n& Assembly"] = {
                "color": "#c77dff",
                "detail": proc or "TBD",
            }

        # Software/GNC
        if generated_code or project_data.get("gcode"):
            subsystems["Software\n& Control"] = {
                "color": "#74b9ff",
                "detail": (generated_code or {}).get("language", "Firmware") if isinstance(generated_code, dict) else "GCode",
            }

        if not subsystems:
            subsystems["System"] = {"color": "#888899", "detail": "No data"}

        keys = list(subsystems.keys())
        n = len(keys)

        fig, ax = plt.subplots(figsize=(max(12, n * 2.2), 6))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        ax.axis("off")
        ax.set_xlim(-1, n * 2.8 + 1)
        ax.set_ylim(-0.5, 4.5)
        ax.set_title(
            f"System Architecture — {project_data.get('intent', 'Hardware Project')}",
            color="white", fontsize=12, fontfamily="monospace", pad=12
        )

        # Central "System" hub
        hub_x = n * 2.8 / 2 - 1.0
        hub = FancyBboxPatch((hub_x - 1.1, 1.5), 2.2, 1.0,
                             boxstyle="round,pad=0.1",
                             facecolor="#1a1a3e", edgecolor="#ffffff44", linewidth=1.5)
        ax.add_patch(hub)
        ax.text(hub_x, 2.0, "BRICK OS\nOrchestrator", color="white",
                ha="center", va="center", fontsize=8, fontfamily="monospace",
                fontweight="bold")

        spacing = n * 2.8 / (n + 1)
        for i, (name, meta) in enumerate(subsystems.items()):
            bx = spacing * (i + 1) - 0.5
            by = 0.0 if i % 2 == 0 else 3.2
            color = meta["color"]
            detail = meta["detail"]

            box = FancyBboxPatch((bx - 1.0, by), 2.0, 1.0,
                                 boxstyle="round,pad=0.1",
                                 facecolor=color + "22", edgecolor=color,
                                 linewidth=1.8)
            ax.add_patch(box)
            ax.text(bx, by + 0.65, name, color=color, ha="center", va="center",
                    fontsize=8, fontfamily="monospace", fontweight="bold")
            ax.text(bx, by + 0.20, detail[:18], color="#aaaaaa", ha="center",
                    va="center", fontsize=6.5, fontfamily="monospace")

            # Arrow from subsystem to hub
            mid_y = 2.0
            ax.annotate("",
                        xy=(hub_x if abs(bx - hub_x) > 0.1 else bx, mid_y - 0.05 if by > 1.5 else mid_y + 0.55),
                        xytext=(bx, by + 1.0 if by < 1.5 else by - 0.05),
                        arrowprops=dict(arrowstyle="-", color=color + "99",
                                       lw=1.2, connectionstyle="arc3,rad=0.0"))

        # Bus labels
        ax.text(hub_x, 4.3, "◀ Power Bus  |  Data Bus  |  Control Bus ▶",
                color="#555566", ha="center", fontsize=7, fontfamily="monospace")

        plt.tight_layout()
        path = str(out_dir / "system_architecture.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"[DocumentAgent] System architecture written: {path}")
        return path

    # =========================================================================
    # PHASE 2: DESIGN BRIEF (planning phase)
    # =========================================================================

    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return await self.generate_design_plan(
            intent=params.get("project_name", params.get("user_intent", "Untitled")),
            env=params.get("environment", {}),
            params=params,
        )

    async def generate_design_plan(
        self,
        intent: str,
        env: Dict[str, Any],
        params: Dict[str, Any],
        design_scheme: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate Phase 2 design brief by querying available agents.
        Gracefully handles missing agents and missing params — never hard-fails.
        """
        logger.info(f"{self.name}: generating design brief for '{intent}'")
        agent_data, agent_errors = await self._gather_agent_data(intent, env, params)

        if self.llm_provider:
            return await self._synthesize_with_llm(intent, env, params, agent_data, agent_errors)
        return await self._generate_structured_plan(intent, env, params, agent_data, agent_errors)

    async def _gather_agent_data(
        self, intent: str, env: Dict[str, Any], params: Dict[str, Any]
    ) -> tuple:
        """
        Query specialized agents. Every call is fully optional — missing agents,
        missing params, and service failures are captured as errors, not exceptions.
        The plan is generated from whatever data is available.
        """
        data: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        # 1. Material
        try:
            material_agent = registry.get_agent("MaterialAgent")
            if not material_agent:
                raise ValueError("MaterialAgent not registered")
            # Infer material from params — don't require material_preference
            material_hint = (
                params.get("material_preference")
                or params.get("material")
                or params.get("material_name")
                or _infer_material_from_intent(intent, env)
            )
            call = material_agent.run(material_name=material_hint, temperature=env.get("temp_c", 20))
            if asyncio.iscoroutine(call):
                call = await call
            if isinstance(call, dict):
                data["materials"] = call
        except Exception as e:
            errors["materials"] = str(e)
            logger.debug(f"MaterialAgent skipped: {e}")

        # 2. Manufacturing
        try:
            mfg_agent = registry.get_agent("ManufacturingAgent")
            if not mfg_agent:
                raise ValueError("ManufacturingAgent not registered")
            primary = data.get("materials", {}).get("primary_material") or params.get("material", "aluminum")
            call = mfg_agent.run(geometry_tree=[], material=primary)
            if asyncio.iscoroutine(call):
                call = await call
            if isinstance(call, dict):
                data["manufacturing"] = call
        except Exception as e:
            errors["manufacturing"] = str(e)
            logger.debug(f"ManufacturingAgent skipped: {e}")

        # 3. Cost (quick estimate — doesn't need DB in local dev)
        try:
            cost_agent = registry.get_agent("CostAgent")
            if not cost_agent:
                raise ValueError("CostAgent not registered")
            material_key = (
                data.get("materials", {}).get("primary_material")
                or params.get("material", "aluminum_6061")
            )
            call = cost_agent.quick_estimate({
                "material_name": material_key,
                "process_type": data.get("manufacturing", {}).get("process", "cnc_milling"),
                "quantity": params.get("quantity", 1),
                "design_parameters": params,
            })
            if asyncio.iscoroutine(call):
                call = await call
            if isinstance(call, dict):
                data["cost"] = call
        except Exception as e:
            errors["cost"] = str(e)
            logger.debug(f"CostAgent skipped: {e}")

        # 4. Design quality (DesignerAgent in refine mode)
        try:
            design_agent = registry.get_agent("DesignerAgent")
            if not design_agent:
                raise ValueError("DesignerAgent not registered")
            call = design_agent.run({"mode": "refine", "design_type": intent,
                                     "requirements": params, "environment": env})
            if asyncio.iscoroutine(call):
                call = await call
            if isinstance(call, dict):
                data["quality"] = call
        except Exception as e:
            errors["quality"] = str(e)
            logger.debug(f"DesignerAgent quality skipped: {e}")

        # 5. Testing plan — always generated locally, no external agent
        data["testing"] = self._generate_testing_plan(intent, params, env)

        return data, errors

    def _generate_testing_plan(self, intent: str, params: Dict, env: Dict) -> Dict:
        """
        Dynamic test plan derived from intent keywords, environment type,
        and design parameters — not hardcoded generic strings.
        """
        env_type = env.get("type", "GROUND").upper()
        intent_lower = intent.lower()

        # Derive domain from intent
        is_electronics = any(w in intent_lower for w in ("pcb", "circuit", "sensor", "controller", "drone", "robot"))
        is_structural = any(w in intent_lower for w in ("bracket", "frame", "housing", "enclosure", "structural"))
        is_fluid = any(w in intent_lower for w in ("pump", "valve", "nozzle", "pipe", "flow", "aerodynamic", "propeller"))
        is_thermal = any(w in intent_lower for w in ("heat", "thermal", "cooling", "radiator", "heatsink"))

        unit = ["Dimensional verification per drawing", "Surface finish measurement"]
        integration = ["Assembly fit-check: all mating interfaces"]
        performance = []
        acceptance = "All critical dimensions within tolerance; no safety-factor violations"

        if is_electronics:
            unit += ["Power-on self-test (POST)", "Sensor read accuracy ±2%", "Communication bus integrity (100 transactions)"]
            integration += ["Sensor-to-actuator latency < 10 ms", "Firmware flash success verification"]
            performance += ["EMC pre-compliance scan (radiated emissions)", "ESD susceptibility (IEC 61000-4-2)"]

        if is_structural:
            performance += [f"Static load test at 1.5× rated load", "Proof load at 2.0× rated load — no fracture"]
            unit += ["Material hardness verification (Rockwell/Brinell)", "Weld / joint inspection (if applicable)"]

        if is_fluid:
            performance += ["Flow rate vs pressure drop characterisation", "Seal/leak test at 1.5× rated pressure"]
            unit += ["O-ring groove dimensional check", "Port thread gauging"]

        if is_thermal:
            t_max = params.get("max_operating_temp_c", 85)
            performance += [f"Thermal soak at {t_max}°C for 2 h — functional throughout",
                            "Temperature distribution mapping (IR camera or thermocouples)"]

        # Environment-specific additions
        if env_type in ("AEROSPACE", "SPACE"):
            performance += [
                "Vibration: sinusoidal sweep 5–2000 Hz per ECSS-E-ST-10",
                "Random vibration: 1 h per axis per qualification spec",
                "Thermal-vacuum cycling: −40°C to +85°C, 10 cycles",
            ]
            acceptance += "; meets ECSS / MIL-STD qualification"
        elif env_type == "UNDERWATER":
            performance += [
                "Hydrostatic pressure test: rated depth × 1.5 for 30 min",
                "Ingress protection: IP68 verification",
            ]
        elif env_type == "INDUSTRIAL":
            performance += [
                "IP54 dust and water ingress test",
                "Operating temperature range validation",
                "Vibration per IEC 60068-2-6 (industrial machinery profile)",
            ]

        return {
            "unit_tests": unit,
            "integration_tests": integration,
            "performance_tests": performance,
            "acceptance_criteria": acceptance,
        }

    # -------------------------------------------------------------------------
    # Plan synthesis (no change in logic, just cleaner)
    # -------------------------------------------------------------------------

    async def _synthesize_with_llm(
        self, intent, env, params, agent_data, agent_errors
    ) -> Dict[str, Any]:
        error_section = ""
        if agent_errors:
            error_section = "\n**Data gaps (agent errors):**\n" + "\n".join(
                f"- {k}: {v}" for k, v in agent_errors.items()
            )

        prompt = (
            f"Synthesize a professional hardware design brief.\n\n"
            f"**Project:** {intent}\n"
            f"**Environment:** {env.get('type','GROUND')}\n\n"
            f"**Agent data:**\n"
            f"- Materials: {agent_data.get('materials','N/A')}\n"
            f"- Manufacturing: {agent_data.get('manufacturing','N/A')}\n"
            f"- Cost: {agent_data.get('cost','N/A')}\n"
            f"- Quality/Risks: {agent_data.get('quality','N/A')}\n"
            f"- Testing: {agent_data.get('testing',{})}\n"
            f"{error_section}\n\n"
            "Sections: Overview, Architecture, Materials, Manufacturing, "
            "Cost Estimate, Technical Challenges, Test Plan, Roadmap, Next Steps.\n"
            "Use agent data as source of truth. State clearly when data is unavailable."
        )

        try:
            call = self.llm_provider.generate(
                prompt=prompt,
                system_prompt="You synthesize engineering data into clear hardware design documentation.",
            )
            if asyncio.iscoroutine(call):
                call = await call
            plan_content = call

            # PDF via reportlab (more reliable than weasyprint)
            pdf_path = self._try_write_pdf(plan_content, intent)

            return {
                "status": "success" if not agent_errors else "partial",
                "document": {
                    "title": f"Design Brief: {intent}",
                    "content": plan_content,
                    "type": "design_brief",
                    "pdf_path": pdf_path,
                },
                "agent_data": agent_data,
                "agent_errors": agent_errors,
            }
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return await self._generate_structured_plan(intent, env, params, agent_data, agent_errors)

    def _try_write_pdf(self, markdown_content: str, title: str) -> Optional[str]:
        """
        Attempt PDF export. Tries reportlab first (pure Python), then weasyprint.
        Returns path on success, None if neither available.
        """
        safe_title = title.replace(" ", "_").replace("/", "-")[:40]
        pdf_path = f"output/reports/{safe_title}.pdf"
        os.makedirs("output/reports", exist_ok=True)

        # Try reportlab (pure Python, no system deps)
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet

            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            for line in markdown_content.split("\n"):
                if line.startswith("# "):
                    story.append(Paragraph(line[2:], styles["Title"]))
                elif line.startswith("## "):
                    story.append(Paragraph(line[3:], styles["Heading2"]))
                elif line.startswith("### "):
                    story.append(Paragraph(line[4:], styles["Heading3"]))
                elif line.strip():
                    story.append(Paragraph(line, styles["Normal"]))
                else:
                    story.append(Spacer(1, 6))
            doc.build(story)
            logger.info(f"PDF written via reportlab: {pdf_path}")
            return pdf_path
        except ImportError:
            pass

        # Fallback: weasyprint
        try:
            import markdown as md_lib
            from weasyprint import HTML
            html = md_lib.markdown(markdown_content)
            HTML(string=html).write_pdf(pdf_path)
            logger.info(f"PDF written via weasyprint: {pdf_path}")
            return pdf_path
        except Exception:
            pass

        return None

    async def _generate_structured_plan(
        self, intent, env, params, agent_data, agent_errors
    ) -> Dict[str, Any]:
        materials = agent_data.get("materials", {})
        manufacturing = agent_data.get("manufacturing", {})
        cost = agent_data.get("cost", {})
        quality = agent_data.get("quality", {})
        testing = agent_data.get("testing", {})

        plan_md = f"""# Design Brief: {intent}

## 1. Project Overview
Design of **{intent}** for **{env.get('type', 'GROUND')}** environment.

## 2. Materials & Manufacturing

{self._format_materials_section(materials, agent_errors.get('materials'))}

{self._format_manufacturing_section(manufacturing, agent_errors.get('manufacturing'))}

## 3. Cost Estimate

{self._format_cost_section(cost, agent_errors.get('cost'))}

## 4. Technical Challenges & Risks

{self._format_quality_section(quality, agent_errors.get('quality'))}

## 5. Test Plan

### Unit Tests
{self._format_list(testing.get('unit_tests', []))}

### Integration Tests
{self._format_list(testing.get('integration_tests', []))}

### Performance Tests
{self._format_list(testing.get('performance_tests', []))}

**Acceptance Criteria:** {testing.get('acceptance_criteria', 'TBD')}

## 6. Roadmap

**Week 1–2:** CAD modelling + simulation
**Week 2–3:** Procurement + tooling order
**Week 3–5:** Fabrication + assembly
**Week 5–6:** Test + documentation
"""

        return {
            "status": "success" if not agent_errors else "partial",
            "document": {
                "title": f"Design Brief: {intent}",
                "content": plan_md,
                "type": "design_brief",
                "pdf_path": None,
            },
            "agent_data": agent_data,
            "agent_errors": agent_errors,
        }

    # -------------------------------------------------------------------------
    # Formatting helpers
    # -------------------------------------------------------------------------

    def _format_materials_section(self, materials: Dict, error: Optional[str]) -> str:
        if error:
            return f"### Materials\n⚠️ **Error:** {error}"
        primary = materials.get("primary_material", "Not specified")
        return f"### Materials\n- **Primary:** {primary}\n- **Justification:** {materials.get('justification','N/A')}"

    def _format_manufacturing_section(self, manufacturing: Dict, error: Optional[str]) -> str:
        if error:
            return f"### Manufacturing\n⚠️ **Error:** {error}"
        processes = manufacturing.get("processes", [])
        lead = manufacturing.get("lead_time_days", "N/A")
        return f"### Manufacturing\n{self._format_list(processes)}\n**Lead Time:** {lead} days"

    def _format_cost_section(self, cost: Dict, error: Optional[str]) -> str:
        if error:
            return f"### Cost\n⚠️ **Error:** {error}"
        total = cost.get("estimated_cost", cost.get("total_estimate"))
        bd = cost.get("breakdown", {})
        if total is None:
            return "### Cost\n*No estimate available.*"
        return (
            f"### Cost\n**Total:** ${float(total):,.2f}\n\n"
            + "\n".join(f"- **{k}:** ${v:,.2f}" if isinstance(v, (int, float)) else f"- **{k}:** {v}" for k, v in bd.items())
        )

    def _format_quality_section(self, quality: Dict, error: Optional[str]) -> str:
        if error:
            return f"### Quality / Risks\n⚠️ **Error:** {error}"
        risks = quality.get("risks", [])
        score = quality.get("score")
        return (
            f"### Quality / Risks\n"
            + (f"**Quality Score:** {score:.0%}\n\n" if score else "")
            + (self._format_list(risks) if risks else "*No risks identified.*")
        )

    def _format_list(self, items: List) -> str:
        return "\n".join(f"- {i}" for i in items) if items else "- None"


# -------------------------------------------------------------------------
# Helper: infer likely material from project intent and environment
# -------------------------------------------------------------------------

def _infer_material_from_intent(intent: str, env: Dict) -> str:
    intent_lower = intent.lower()
    env_type = env.get("type", "GROUND").upper()
    if env_type == "SPACE":
        return "aluminum_6061"
    if any(w in intent_lower for w in ("titanium", "aerospace", "aircraft")):
        return "titanium_grade5"
    if any(w in intent_lower for w in ("stainless", "food", "medical")):
        return "stainless_316"
    if any(w in intent_lower for w in ("plastic", "housing", "enclosure", "print")):
        return "abs_plastic"
    if any(w in intent_lower for w in ("carbon", "composite", "uav", "drone")):
        return "cfrp"
    return "aluminum_6061"


# =============================================================================
# FastAPI routes
# =============================================================================

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    router = None

if HAS_FASTAPI:
    router = APIRouter(prefix="/document", tags=["documentation"])

    class DesignPlanRequest(BaseModel):
        intent: str
        environment: Dict[str, Any] = Field(default_factory=dict)
        parameters: Dict[str, Any] = Field(default_factory=dict)

    class FinalDocRequest(BaseModel):
        project_data: Dict[str, Any]

    @router.post("/plan/generate")
    async def generate_design_plan(request: DesignPlanRequest):
        try:
            agent = DocumentAgent()
            return await agent.generate_design_plan(
                intent=request.intent, env=request.environment, params=request.parameters
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/final/generate")
    def generate_final_docs(request: FinalDocRequest):
        """Generate complete deliverable package and return file manifest."""
        try:
            agent = DocumentAgent()
            return agent.generate_final_documentation(request.project_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/download/{project_id}")
    def download_package(project_id: str):
        """Return ZIP package path for download."""
        from fastapi.responses import FileResponse
        zip_path = OUTPUT_BASE / project_id / f"{project_id}_package.zip"
        if not zip_path.exists():
            raise HTTPException(status_code=404, detail="Package not yet generated")
        return FileResponse(str(zip_path), media_type="application/zip",
                            filename=f"{project_id}_package.zip")

    @router.post("/run")
    async def run_document_agent(params: Dict[str, Any]):
        try:
            agent = DocumentAgent()
            return await agent.run(params)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
