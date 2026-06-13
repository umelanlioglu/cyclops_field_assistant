"""Label normalization utilities."""

from __future__ import annotations

from typing import Dict, Optional

# In the current test annotation, "filament" was used for the spool/roll.
# Remove this alias later if you rename it properly in Labellerr.
DEFAULT_LABEL_ALIASES = {
    "filament": "filament_spool",
    "gantry": "gantry_frame",
}

YOLO_CANONICAL_LABELS = {
    "bed_clamp",
    "bowden_tube",
    "button",
    "display_screen",
    "extruder_motor",
    "filament_detector",
    "filament_holder",
    "filament_spool",
    "gantry_frame",
    "network_interface",
    "nozzle_kit",
    "power_switch",
    "print_bed",
    "qr_code",
    "sd_card_port",
    "side_ports",
    "toolbox",
    "usb_port",
    "x_axis_gantry",
    "x_axis_motor",
    "y_axis_belt_adjuster",
    "y_axis_motor",
}

DEFAULT_LABEL_ALIASES = {
    "nozzle_assembly": "nozzle_kit",
    "nozzle": "nozzle_kit",
    "hotend": "nozzle_kit",
    "hot_end": "nozzle_kit",

    "extruder_area": "extruder_motor",
    "extruder": "extruder_motor",

    "sd_card_slot": "sd_card_port",
    "storage_card_slot": "sd_card_port",
    "tf_card_slot": "sd_card_port",

    "tool_box": "toolbox",

    "y_axis_belt_tension_knob": "y_axis_belt_adjuster",

    "power_port_switch": "power_switch",
    "switch_control": "power_switch",

    "lcd_screen": "display_screen",
    "screen": "display_screen",
    "display": "display_screen",

    "teflon_tube": "bowden_tube",

    "feeding_holder_components": "filament_holder",
    "material_rack": "filament_holder",
    "rack": "filament_holder",

    "spool": "filament_spool",

    "printing_platform": "print_bed",
    "platform": "print_bed",

    "lan_port": "network_interface",
    "usb": "usb_port",

    "gantry": "gantry_frame",
    "x_axis_bar": "x_axis_gantry",
}

def normalize_label(label: str, aliases: Optional[Dict[str, str]] = None) -> str:
    aliases = aliases or DEFAULT_LABEL_ALIASES
    clean = label.strip().lower().replace(" ", "_").replace("-", "_")
    normalized = aliases.get(clean, clean)
    return normalized
