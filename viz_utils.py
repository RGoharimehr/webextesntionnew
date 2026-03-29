import os
import json
from typing import Dict, Optional, Tuple, List, Set, Any

from .fnx_api import FNXApi
from .fnx_io_definition import OutputDefinition

import omni.usd
from pxr import Usd, UsdShade, Gf, Sdf

try:
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import io
    from matplotlib.ticker import MaxNLocator
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False


print(f"[DEBUG] Matplotlib available: {MATPLOTLIB_AVAILABLE}")
print(f"[DEBUG] Pillow available: {PILLOW_AVAILABLE}")


COLOR_MAP_OPTIONS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "Blues", "Greens", "Reds",
    "coolwarm", "bwr", "seismic",
    "jet", "rainbow",
    "gray",
]

FALLBACK_NO_RESULT_MATERIAL_PATH = "/World/Looks/Aging_Copper_66"

# Small caches to avoid redoing heavy legend work every fetch.
_COLORBAR_IMAGE_CACHE: Dict[Tuple[float, float, str, str, int, int], Optional[tuple]] = {}
_LAST_LEGEND_STATE: Optional[Tuple[float, float, str, str]] = None


def get_visualizable_properties():
    return [
        "Temperature",
        "Pressure",
        "Quality",
        "Velocity",
        "Volume Flow Rate",
        "Mass Flux",
    ]



def _norm_text(value: Optional[str]) -> str:
    text = (value or "").strip().lower()
    for token in ("_", "-", "/", "(", ")", ":", ","):
        text = text.replace(token, " ")
    return " ".join(text.split())



def _clean_property_text(text: str) -> str:
    text = (text or "").strip().lower()

    # Remove Flownex prefix like {Flow Element Results,Generic}
    if "}" in text:
        text = text.split("}", 1)[1]

    for ch in "_-/:,()":
        text = text.replace(ch, " ")

    return " ".join(text.split())



def _property_phrases(selected_prop_name: str) -> Set[str]:
    target = _norm_text(selected_prop_name)

    mapping = {
        "temperature": {"total temperature", "temperature"},
        "pressure": {"mean pressure", "pressure"},
        "volume flow rate": {"total volume flow", "volume flow"},
        "quality": {"quality"},
        "velocity": {"velocity"},
        "mass flux": {"mass flux", "mass flow"},
    }

    return mapping.get(target, {target})



def _matches_selected_property(selected_prop_name: str, output_def: OutputDefinition) -> bool:
    phrases = _property_phrases(selected_prop_name)

    property_identifier = _clean_property_text(output_def.PropertyIdentifier)
    description = _clean_property_text(output_def.Description)

    searchable_text = f"{property_identifier} {description}".strip()

    for phrase in phrases:
        norm_phrase = _clean_property_text(phrase)
        if not norm_phrase:
            continue

        if searchable_text == norm_phrase:
            return True

        if norm_phrase in searchable_text:
            return True

    return False



def _summary_log(
    selected_prop_name: str,
    selected_cmap: str,
    unit: str,
    vmin: Optional[float],
    vmax: Optional[float],
    processed: int,
    fallback_prims: int,
    unmapped_count: int,
    matched_outputs_count: int,
    mapped_outputs_count: int,
    fallback_components_without_property: int,
    manual_bounds_used: bool,
) -> str:
    header = f"Visualizing '{selected_prop_name}' with '{selected_cmap}' colormap.\n"
    header += "-------------------------------------\n"

    range_label = f"[{vmin:.3g}, {vmax:.3g}]" if vmin is not None and vmax is not None else "N/A"
    unit_suffix = f" {unit}" if unit else ""
    bounds_mode = "manual" if manual_bounds_used else "auto"

    text = header
    text += f"Range ({bounds_mode}): {range_label}{unit_suffix}\n"
    text += f"Matched outputs: {matched_outputs_count}\n"
    text += f"Matched outputs with mapping: {mapped_outputs_count}\n"
    text += f"Visualization material bound to {processed} prims.\n"
    text += f"Fallback material bound to {fallback_prims} prims.\n"
    text += f"Mapped components with no output for selected property: {fallback_components_without_property}\n"
    text += f"Unmapped matched outputs: {unmapped_count}\n"
    return text



def color_map(norm: float, cmap: str = "viridis") -> Tuple[float, float, float]:
    norm = max(0.0, min(1.0, float(norm)))

    if not MATPLOTLIB_AVAILABLE:
        print("[viz] Warning: Matplotlib not found. Falling back to grayscale.")
        return (norm, norm, norm)

    try:
        colormap_func = cm.get_cmap(cmap)
        rgba = colormap_func(norm)
        return float(rgba[0]), float(rgba[1]), float(rgba[2])
    except ValueError:
        print(f"[viz] Warning: Colormap '{cmap}' not found. Falling back to 'viridis'.")
        rgba = cm.get_cmap("viridis")(norm)
        return float(rgba[0]), float(rgba[1]), float(rgba[2])



def _load_component_to_prim_map(mapping_json_path: str) -> Dict[str, List[str]]:
    with open(mapping_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Mapping JSON must be an object of {ComponentIdentifier: primPath or [primPaths]}")

    normalized: Dict[str, List[str]] = {}
    for component_id, prim_value in data.items():
        if isinstance(prim_value, str):
            normalized[component_id] = [prim_value]
        elif isinstance(prim_value, list):
            normalized[component_id] = [p for p in prim_value if isinstance(p, str) and p.strip()]
        else:
            normalized[component_id] = []
    return normalized



def _make_safe_token(value: str) -> str:
    safe = value or "unknown"
    for ch in [' ', '/', '\\', ':', '.', ',', ';', '(', ')', '[', ']', '{', '}', '-', '+', '*', '?', '#', '@', '!']:
        safe = safe.replace(ch, "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "unknown"



def _get_or_create_visual_material(stage: Usd.Stage, component_id: str) -> Tuple[UsdShade.Material, UsdShade.Shader]:
    safe_component = _make_safe_token(component_id)
    looks_scope_path = "/World/Looks"
    material_path = f"{looks_scope_path}/FNX_{safe_component}"
    shader_path = f"{material_path}/PreviewSurface"

    stage.DefinePrim(looks_scope_path, "Scope")

    material = UsdShade.Material.Get(stage, material_path)
    if not material:
        material = UsdShade.Material.Define(stage, material_path)

    shader = UsdShade.Shader.Get(stage, shader_path)
    if not shader:
        shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")

    diffuse_input = shader.GetInput("diffuseColor")
    if not diffuse_input:
        diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    diffuse_input.Set(Gf.Vec3f(0.5, 0.5, 0.5))

    roughness_input = shader.GetInput("roughness")
    if not roughness_input:
        roughness_input = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
    roughness_input.Set(0.1)

    metallic_input = shader.GetInput("metallic")
    if not metallic_input:
        metallic_input = shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)
    metallic_input.Set(1.0)

    emissive_input = shader.GetInput("emissiveColor")
    if not emissive_input:
        emissive_input = shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
    emissive_input.Set(Gf.Vec3f(0.0, 0.0, 0.0))

    clearcoat_input = shader.GetInput("clearcoat")
    if not clearcoat_input:
        clearcoat_input = shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float)
    clearcoat_input.Set(0.35)

    opacity_input = shader.GetInput("opacity")
    if not opacity_input:
        opacity_input = shader.CreateInput("opacity", Sdf.ValueTypeNames.Float)
    opacity_input.Set(1.0)

    surface_output = material.GetSurfaceOutput()
    if not surface_output:
        surface_output = material.CreateSurfaceOutput()

    if not surface_output.HasConnectedSource():
        surface_output.ConnectToSource(shader.ConnectableAPI(), "surface")

    return material, shader



def _set_material_color(shader: UsdShade.Shader, rgb: Tuple[float, float, float]) -> None:
    color_input = shader.GetInput("diffuseColor")
    if not color_input:
        color_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    color_input.Set(Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2])))

    emissive_input = shader.GetInput("emissiveColor")
    if not emissive_input:
        emissive_input = shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
    emissive_input.Set(Gf.Vec3f(0.0, 0.0, 0.0))



def _bind_material_to_prim(prim: Usd.Prim, material: UsdShade.Material) -> bool:
    if not prim or not prim.IsValid():
        return False
    try:
        UsdShade.MaterialBindingAPI(prim).Bind(material)
        return True
    except Exception as e:
        print(f"[viz] Failed to bind material to {prim.GetPath()}: {e}")
        return False



def _bind_fallback_material(stage: Usd.Stage, prim_paths: List[str]) -> int:
    fallback_mat = UsdShade.Material.Get(stage, FALLBACK_NO_RESULT_MATERIAL_PATH)
    if not fallback_mat:
        print(f"[Viz] Fallback material not found: {FALLBACK_NO_RESULT_MATERIAL_PATH}")
        return 0

    bound_count = 0
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            UsdShade.MaterialBindingAPI(prim).Bind(fallback_mat)
            bound_count += 1

    return bound_count



def _unbind_material_from_prim(prim: Usd.Prim) -> None:
    if not prim or not prim.IsValid():
        return
    try:
        UsdShade.MaterialBindingAPI(prim).UnbindAllBindings()
    except Exception as e:
        print(f"[viz] Failed to unbind material from {prim.GetPath()}: {e}")



def _reset_prim_colors(stage: Usd.Stage, prim_paths: Set[str]) -> None:
    if not stage or not prim_paths:
        return
    with Sdf.ChangeBlock():
        for path_str in prim_paths:
            prim = stage.GetPrimAtPath(path_str)
            if prim and prim.IsValid():
                _unbind_material_from_prim(prim)



def _visualize_single_component(
    output_def: OutputDefinition,
    value: float,
    comp_to_prim_map: Dict[str, List[str]],
    property_ranges: Dict[str, Dict[str, float]],
    cmap: str,
    usd_context=None,
) -> Dict[str, Any]:
    key = output_def.Key
    component_id = output_def.ComponentIdentifier
    info: Dict[str, Any] = {
        "key": key,
        "status": "error",
        "message": "",
        "colored_prims": 0,
        "colored_paths": [],
    }

    prim_paths = comp_to_prim_map.get(component_id)
    if not prim_paths:
        info["message"] = "Not found in mapping file"
        return info

    pr = property_ranges.get(key)
    if not pr or pr.get("min") is None or pr.get("max") is None:
        info["message"] = "No valid range specified"
        return info

    vmin, vmax = pr["min"], pr["max"]
    norm = (value - vmin) / (vmax - vmin) if (vmax - vmin) > 1e-9 else 0.5
    rgb = color_map(norm, cmap=cmap)

    ctx = usd_context or omni.usd.get_context()
    stage = ctx.get_stage()
    if not stage:
        info["message"] = "USD stage not available"
        return info

    try:
        material, shader = _get_or_create_visual_material(stage, component_id)
        _set_material_color(shader, rgb)
    except Exception as e:
        info["message"] = f"Failed to create/update material: {e}"
        return info

    total_bound = 0
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            continue
        if _bind_material_to_prim(prim, material):
            total_bound += 1
            info["colored_paths"].append(prim_path)

    info["colored_prims"] = total_bound
    if total_bound > 0:
        info["status"] = "ok"
        info["message"] = f"Bound visualization material to {total_bound} mapped prims."
    else:
        info["message"] = "No valid mapped prims were colored."
    return info



def _manual_bounds_are_placeholder(manual_min_bound, manual_max_bound) -> bool:
    try:
        if manual_min_bound is None or manual_max_bound is None:
            return False
        a = float(manual_min_bound)
        b = float(manual_max_bound)
        return abs(a - 0.0) < 1e-12 and abs(b - 1000.0) < 1e-12
    except Exception:
        return False



def visualize_property_layer(
    log_field,
    property_combo,
    colormap_combo,
    property_names_for_viz,
    user_config,
    fnx_outputs: List[OutputDefinition],
    output_fields: Dict[str, str],
    fnx_api: FNXApi,
    prims_to_reset: Set[str],
    manual_min_bound=None,
    manual_max_bound=None,
):
    if not log_field or not user_config:
        return None, None, None, None, set()

    stage = omni.usd.get_context().get_stage()
    if stage and prims_to_reset:
        _reset_prim_colors(stage, prims_to_reset)

    newly_colored_prims: Set[str] = set()

    selected_prop_index = property_combo.model.get_item_value_model().as_int
    selected_prop_name = property_names_for_viz[selected_prop_index]

    selected_cmap_index = colormap_combo.model.get_item_value_model().as_int
    selected_cmap = COLOR_MAP_OPTIONS[selected_cmap_index]

    io_dir = user_config.Setup.IOFileDirectory
    mapping_json_path = os.path.join(io_dir, "FlownexMapping.json")

    if not os.path.exists(mapping_json_path):
        log_field.model.set_value(f"Mapping file not found at {mapping_json_path}")
        return None, None, None, None, newly_colored_prims

    try:
        comp_to_prim_map = _load_component_to_prim_map(mapping_json_path)
    except Exception as e:
        log_field.model.set_value(f"Error loading mapping file: {e}")
        return None, None, None, None, newly_colored_prims

    if not fnx_outputs:
        log_field.model.set_value("No Flownex outputs loaded.")
        return None, None, None, None, newly_colored_prims

    outputs_to_visualize = [
        out for out in fnx_outputs
        if _matches_selected_property(selected_prop_name, out)
    ]

    if not outputs_to_visualize:
        fallback_count = 0
        for _, prim_paths in comp_to_prim_map.items():
            bound_count = _bind_fallback_material(stage, prim_paths)
            fallback_count += bound_count
            newly_colored_prims.update(prim_paths)

        log_field.model.set_value(
            f"Visualizing '{selected_prop_name}' with '{selected_cmap}' colormap.\n"
            f"-------------------------------------\n"
            f"No outputs found matching '{selected_prop_name}'.\n"
            f"Fallback material bound to {fallback_count} prims."
        )
        return None, None, selected_cmap, selected_prop_name, newly_colored_prims

    unit = outputs_to_visualize[0].Unit if outputs_to_visualize else ""
    full_label = f"{selected_prop_name} ({unit})" if unit else selected_prop_name

    use_manual_bounds = (
        manual_min_bound is not None and
        manual_max_bound is not None and
        not _manual_bounds_are_placeholder(manual_min_bound, manual_max_bound)
    )

    if use_manual_bounds:
        vmin, vmax = float(manual_min_bound), float(manual_max_bound)
    else:
        numeric_values: List[float] = []
        for out_def in outputs_to_visualize:
            value_str = output_fields.get(out_def.Key)
            if value_str is None or value_str == "":
                continue
            try:
                numeric_values.append(float(value_str))
            except (ValueError, TypeError):
                continue

        if numeric_values:
            vmin, vmax = min(numeric_values), max(numeric_values)
        else:
            vmin, vmax = 0.0, 1.0

    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0

    property_ranges = {
        out_def.Key: {"min": vmin, "max": vmax}
        for out_def in outputs_to_visualize
    }

    outputs_with_mapping = [
        out for out in outputs_to_visualize
        if out.ComponentIdentifier in comp_to_prim_map
    ]

    matched_component_ids = {out.ComponentIdentifier for out in outputs_with_mapping}
    all_mapped_component_ids = set(comp_to_prim_map.keys())
    components_with_no_output_for_selected_property = all_mapped_component_ids - matched_component_ids

    processed = 0
    errors = 0
    fallback_prims = 0

    for out_def in outputs_with_mapping:
        value_str = output_fields.get(out_def.Key)
        prim_paths = comp_to_prim_map.get(out_def.ComponentIdentifier, [])

        if value_str is None or value_str == "":
            bound_count = _bind_fallback_material(stage, prim_paths)
            fallback_prims += bound_count
            newly_colored_prims.update(prim_paths)
            continue

        try:
            value = float(value_str)
        except (ValueError, TypeError):
            bound_count = _bind_fallback_material(stage, prim_paths)
            fallback_prims += bound_count
            newly_colored_prims.update(prim_paths)
            continue

        result_info = _visualize_single_component(
            output_def=out_def,
            value=value,
            comp_to_prim_map=comp_to_prim_map,
            property_ranges=property_ranges,
            cmap=selected_cmap,
        )

        if result_info["status"] == "ok":
            processed += result_info.get("colored_prims", 0)
            newly_colored_prims.update(result_info["colored_paths"])
        else:
            bound_count = _bind_fallback_material(stage, prim_paths)
            fallback_prims += bound_count
            newly_colored_prims.update(prim_paths)
            errors += 1

    fallback_components_without_property = 0
    for component_id in sorted(components_with_no_output_for_selected_property):
        prim_paths = comp_to_prim_map.get(component_id, [])
        bound_count = _bind_fallback_material(stage, prim_paths)
        fallback_prims += bound_count
        newly_colored_prims.update(prim_paths)
        fallback_components_without_property += 1

    unmapped_outputs = [
        out for out in outputs_to_visualize
        if out.ComponentIdentifier not in comp_to_prim_map
    ]

    log_field.model.set_value(
        _summary_log(
            selected_prop_name=selected_prop_name,
            selected_cmap=selected_cmap,
            unit=unit,
            vmin=vmin,
            vmax=vmax,
            processed=processed,
            fallback_prims=fallback_prims,
            unmapped_count=len(unmapped_outputs),
            matched_outputs_count=len(outputs_to_visualize),
            mapped_outputs_count=len(outputs_with_mapping),
            fallback_components_without_property=fallback_components_without_property,
            manual_bounds_used=use_manual_bounds,
        )
    )
    return vmin, vmax, selected_cmap, full_label, newly_colored_prims



def legend_state_changed(current_state, vmin, vmax, cmap, label) -> bool:
    if current_state is None:
        return True

    return not (
        current_state.get("vmin") == vmin and
        current_state.get("vmax") == vmax and
        current_state.get("cmap") == cmap and
        current_state.get("label") == label
    )



def get_legend_segments(
    vmin: float,
    vmax: float,
    cmap_name: str,
    label: str,
    segments: int = 64,
) -> Dict[str, Any]:
    """
    Omniverse-native legend data source.
    Use the returned RGB segment list to draw ui.Rectangle widgets instead of
    regenerating a PNG on every fetch.
    """
    if segments < 2:
        segments = 2

    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0

    colors: List[Tuple[float, float, float]] = []
    for idx in range(segments):
        norm = idx / float(segments - 1)
        colors.append(color_map(norm, cmap=cmap_name))

    return {
        "label": label,
        "vmin": vmin,
        "vmax": vmax,
        "cmap": cmap_name,
        "colors": colors,
    }



def generate_colorbar_image(
    vmin: float,
    vmax: float,
    cmap_name: str,
    label: str,
    width: int = 800,
    height: int = 50,
) -> Optional[tuple]:
    """
    Backward-compatible PNG legend generator.
    Now cached so repeated fetches with the same legend state do not regenerate it.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[viz] Matplotlib not found, cannot generate colorbar.")
        return None

    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0

    cache_key = (
        round(float(vmin), 8),
        round(float(vmax), 8),
        cmap_name,
        label,
        int(width),
        int(height),
    )
    if cache_key in _COLORBAR_IMAGE_CACHE:
        return _COLORBAR_IMAGE_CACHE[cache_key]

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=300)
    fig.patch.set_alpha(0.0)
    ax.set_axis_off()

    cax = fig.add_axes([0.05, 0.4, 0.9, 0.4])

    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        print(f"[viz] Warning: Colormap '{cmap_name}' not found. Falling back to 'viridis'.")
        cmap = plt.get_cmap("viridis")

    norm = Normalize(vmin=vmin, vmax=vmax)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = plt.colorbar(mappable, cax=cax, orientation="horizontal")

    cb.set_label(label, color="white", fontsize=14, weight="bold")
    cb.ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    cb.ax.tick_params(colors="white", labelsize=12)
    cb.outline.set_edgecolor("white")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    if not PILLOW_AVAILABLE:
        print("[viz] Pillow not found, cannot guarantee correct image format.")
        _COLORBAR_IMAGE_CACHE[cache_key] = None
        return None

    try:
        buf.seek(0)
        with Image.open(buf) as pil_image:
            rgba_image = pil_image.convert("RGBA")
            result = (bytearray(rgba_image.tobytes()), [rgba_image.width, rgba_image.height])
            _COLORBAR_IMAGE_CACHE[cache_key] = result
            return result
    except Exception as e:
        print(f"[viz] [ERROR] Failed to process image with Pillow: {e}")
        _COLORBAR_IMAGE_CACHE[cache_key] = None
        return None
