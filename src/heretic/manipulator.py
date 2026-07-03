# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from contextlib import suppress
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from rich.table import Table
from transformers import PreTrainedModel

from .config import (
    ModelManipulationOperation,
    ModelManipulationSpecification,
    Settings,
)
from .model import Model, get_model_class
from .system import empty_cache
from .utils import print


@dataclass(frozen=True)
class ModuleTarget:
    component: str
    module: Module
    expert_index: int | None = None
    path_hint: str | None = None


def _get_layers(model: PreTrainedModel) -> ModuleList:
    # Most multimodal models.
    with suppress(Exception):
        return model.model.language_model.layers

    # Text-only models.
    return model.model.layers


def _parse_index_selector(selector: str | None, count: int) -> set[int] | None:
    if selector is None:
        return None

    selector = selector.strip().lower()
    if selector == "all":
        return set(range(count))

    selected: set[int] = set()
    for item in selector.split(","):
        item = item.strip()
        if not item:
            continue

        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending selector range: {item!r}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(item))

    invalid = sorted(index for index in selected if index < 0 or index >= count)
    if invalid:
        raise ValueError(
            f"Selector indexes out of range for count {count}: {invalid}"
        )

    return selected


def _append_if_module(
    targets: list[ModuleTarget],
    component: str,
    module: Any,
    expert_index: int | None = None,
    path_hint: str | None = None,
):
    if isinstance(module, Module):
        targets.append(
            ModuleTarget(
                component=component,
                module=module,
                expert_index=expert_index,
                path_hint=path_hint,
            )
        )
    else:
        assert not isinstance(module, Tensor), (
            f"Unexpected Tensor in {component} - expected nn.Module"
        )


def _get_layer_targets(layer: Module) -> list[ModuleTarget]:
    targets: list[ModuleTarget] = []
    layer = cast(Any, layer)

    # Standard self-attention out-projection (most models).
    with suppress(Exception):
        _append_if_module(
            targets,
            "attn.o_proj",
            layer.self_attn.o_proj,
            path_hint="self_attn.o_proj",
        )

    # Qwen3.5 MoE hybrid linear attention.
    with suppress(Exception):
        _append_if_module(
            targets,
            "attn.o_proj",
            layer.linear_attn.out_proj,
            path_hint="linear_attn.out_proj",
        )

    # Most dense models.
    with suppress(Exception):
        _append_if_module(
            targets,
            "mlp.down_proj",
            layer.mlp.down_proj,
            path_hint="mlp.down_proj",
        )

    # Qwen-style MoE.
    with suppress(Exception):
        for index, expert in enumerate(layer.mlp.experts):
            _append_if_module(
                targets,
                "mlp.down_proj",
                expert.down_proj,
                expert_index=index,
                path_hint=f"mlp.experts.{index}.down_proj",
            )

    # Phi-3.5-MoE and similar.
    with suppress(Exception):
        for index, expert in enumerate(layer.block_sparse_moe.experts):
            _append_if_module(
                targets,
                "mlp.down_proj",
                expert.w2,
                expert_index=index,
                path_hint=f"block_sparse_moe.experts.{index}.w2",
            )

    # LFM dense operator blocks.
    with suppress(Exception):
        _append_if_module(
            targets,
            "attn.o_proj",
            layer.conv.out_proj,
            path_hint="conv.out_proj",
        )

    with suppress(Exception):
        _append_if_module(
            targets,
            "mlp.down_proj",
            layer.feed_forward.w2,
            path_hint="feed_forward.w2",
        )

    # LFM transformer blocks.
    with suppress(Exception):
        _append_if_module(
            targets,
            "attn.o_proj",
            layer.self_attn.out_proj,
            path_hint="self_attn.out_proj",
        )

    with suppress(Exception):
        for index, expert in enumerate(layer.feed_forward.experts):
            _append_if_module(
                targets,
                "mlp.down_proj",
                expert.w2,
                expert_index=index,
                path_hint=f"feed_forward.experts.{index}.w2",
            )

    # Granite MoE Hybrid - attention layers with shared_mlp.
    with suppress(Exception):
        _append_if_module(
            targets,
            "mlp.down_proj",
            layer.shared_mlp.output_linear,
            path_hint="shared_mlp.output_linear",
        )

    # Granite MoE Hybrid - MoE layers with experts.
    with suppress(Exception):
        for index, expert in enumerate(layer.moe.experts):
            _append_if_module(
                targets,
                "mlp.down_proj",
                expert.output_linear,
                expert_index=index,
                path_hint=f"moe.experts.{index}.output_linear",
            )

    return targets


def _select_targets(
    model: PreTrainedModel,
    spec: ModelManipulationSpecification,
) -> dict[int, list[ModuleTarget]]:
    layers = _get_layers(model)
    selected_layers = _parse_index_selector(spec.layers, len(layers))
    assert selected_layers is not None

    components = set(spec.components)
    use_all_components = "*" in components
    selected_by_layer: dict[int, list[ModuleTarget]] = {}

    for layer_index in sorted(selected_layers):
        targets = _get_layer_targets(layers[layer_index])
        max_expert_count = 1 + max(
            (target.expert_index for target in targets if target.expert_index is not None),
            default=-1,
        )
        selected_experts = _parse_index_selector(spec.experts, max_expert_count)

        layer_targets = []
        for target in targets:
            if not use_all_components and target.component not in components:
                continue
            if selected_experts is not None:
                if target.expert_index is None or target.expert_index not in selected_experts:
                    continue
            layer_targets.append(target)

        selected_by_layer[layer_index] = layer_targets

    return selected_by_layer


def _direct_targets(
    model: Module,
    patterns: list[str],
) -> dict[str, Module]:
    if not patterns:
        return {}

    targets = {}
    for name, module in model.named_modules():
        if any(fnmatch(name, pattern) for pattern in patterns):
            targets[name] = module

    return targets


def _apply_tensor_operation(
    base_tensor: Tensor,
    donor_tensor: Tensor,
    operation: ModelManipulationOperation,
    weight: float,
    label: str,
):
    if base_tensor.shape != donor_tensor.shape:
        raise ValueError(
            f"Shape mismatch for {label}: base {tuple(base_tensor.shape)} vs donor {tuple(donor_tensor.shape)}"
        )

    donor_tensor = donor_tensor.to(device=base_tensor.device, dtype=base_tensor.dtype)

    if operation == ModelManipulationOperation.SWAP:
        base_tensor.copy_(donor_tensor)
    elif operation == ModelManipulationOperation.MERGE:
        base_tensor.mul_(1.0 - weight).add_(donor_tensor, alpha=weight)
    else:
        raise ValueError(f"Unsupported manipulation operation: {operation}")


def _apply_module_operation(
    base_module: Module,
    donor_module: Module,
    spec: ModelManipulationSpecification,
    label: str,
):
    base_parameters = dict(base_module.named_parameters(recurse=False))
    donor_parameters = dict(donor_module.named_parameters(recurse=False))

    base_buffers = dict(base_module.named_buffers(recurse=False))
    donor_buffers = dict(donor_module.named_buffers(recurse=False))

    tensor_count = 0

    for name, base_tensor in base_parameters.items():
        donor_tensor = donor_parameters.get(name)
        if donor_tensor is None:
            raise ValueError(f"Donor module is missing parameter {label}.{name}")
        _apply_tensor_operation(
            base_tensor.data,
            donor_tensor.data,
            spec.operation,
            spec.weight,
            f"{label}.{name}",
        )
        tensor_count += 1

    for name, base_tensor in base_buffers.items():
        donor_tensor = donor_buffers.get(name)
        if donor_tensor is None:
            raise ValueError(f"Donor module is missing buffer {label}.{name}")
        _apply_tensor_operation(
            base_tensor.data,
            donor_tensor.data,
            spec.operation,
            spec.weight,
            f"{label}.{name}",
        )
        tensor_count += 1

    if tensor_count == 0:
        raise ValueError(f"Selected module has no direct tensors: {label}")


def _target_key(target: ModuleTarget) -> tuple[str, int | None]:
    return (target.component, target.expert_index)


def _tensor_summaries(module: Module) -> list[dict[str, Any]]:
    tensors: list[dict[str, Any]] = []

    def append_tensors(prefix: str, inspected_module: Module):
        for name, tensor in inspected_module.named_parameters(recurse=False):
            tensors.append(
                {
                    "name": f"{prefix}{name}",
                    "kind": "parameter",
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "requires_grad": tensor.requires_grad,
                }
            )

        for name, tensor in inspected_module.named_buffers(recurse=False):
            tensors.append(
                {
                    "name": f"{prefix}{name}",
                    "kind": "buffer",
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "requires_grad": False,
                }
            )

    append_tensors("", module)

    base_layer = getattr(module, "base_layer", None)
    if isinstance(base_layer, Module):
        append_tensors("base_layer.", base_layer)

    return tensors


def _parameter_count(tensors: list[dict[str, Any]]) -> int:
    count = 0
    for tensor in tensors:
        if tensor["kind"] != "parameter":
            continue

        tensor_count = 1
        for dimension in tensor["shape"]:
            tensor_count *= dimension
        count += tensor_count

    return count


def build_model_inspection_report(
    model: Model,
    module_patterns: list[str] | None = None,
) -> dict[str, Any]:
    layers = model.get_layers()

    report: dict[str, Any] = {
        "model": model.settings.model,
        "layer_count": len(layers),
        "components": {},
        "layers": [],
        "direct_modules": [],
    }

    component_counts: dict[str, int] = {}
    expert_counts: dict[str, int] = {}

    for layer_index, layer in enumerate(layers):
        layer_targets = []

        for target in _get_layer_targets(layer):
            tensors = _tensor_summaries(target.module)
            record = {
                "component": target.component,
                "expert_index": target.expert_index,
                "path_hint": target.path_hint,
                "module_type": type(target.module).__name__,
                "parameter_count": _parameter_count(tensors),
                "tensors": tensors,
            }
            layer_targets.append(record)

            component_counts[target.component] = (
                component_counts.get(target.component, 0) + 1
            )
            if target.expert_index is not None:
                expert_counts[target.component] = max(
                    expert_counts.get(target.component, 0),
                    target.expert_index + 1,
                )

        report["layers"].append(
            {
                "index": layer_index,
                "module_type": type(layer).__name__,
                "targets": layer_targets,
            }
        )

    report["components"] = {
        component: {
            "module_count": count,
            "max_experts_per_layer": expert_counts.get(component, 0),
        }
        for component, count in sorted(component_counts.items())
    }

    if module_patterns:
        direct_modules = _direct_targets(model.model, module_patterns)
        report["direct_modules"] = [
            {
                "name": name,
                "module_type": type(module).__name__,
                "parameter_count": _parameter_count(_tensor_summaries(module)),
                "tensors": _tensor_summaries(module),
            }
            for name, module in sorted(direct_modules.items())
        ]

    return report


def print_model_inspection_report(report: dict[str, Any]):
    print()
    print(
        f"Model inspection for [bold]{report['model']}[/]: "
        f"[bold]{report['layer_count']}[/] layers"
    )

    if report["components"]:
        table = Table()
        table.add_column("Component")
        table.add_column("Modules", justify="right")
        table.add_column("Max experts/layer", justify="right")

        for component, summary in report["components"].items():
            table.add_row(
                component,
                str(summary["module_count"]),
                str(summary["max_experts_per_layer"]),
            )

        print(table)

    target_table = Table()
    target_table.add_column("Layer", justify="right")
    target_table.add_column("Component")
    target_table.add_column("Expert", justify="right")
    target_table.add_column("Path")
    target_table.add_column("Module")
    target_table.add_column("Tensors")

    for layer in report["layers"]:
        for target in layer["targets"]:
            shapes = []
            for tensor in target["tensors"]:
                if tensor["kind"] == "parameter":
                    shapes.append(
                        f"{tensor['name']}={tuple(tensor['shape'])}"
                    )
            target_table.add_row(
                str(layer["index"]),
                target["component"],
                "" if target["expert_index"] is None else str(target["expert_index"]),
                target["path_hint"] or "",
                target["module_type"],
                ", ".join(shapes[:3]),
            )

    print(target_table)

    if report["direct_modules"]:
        direct_table = Table()
        direct_table.add_column("Direct module")
        direct_table.add_column("Module")
        direct_table.add_column("Parameters", justify="right")

        for module in report["direct_modules"]:
            direct_table.add_row(
                module["name"],
                module["module_type"],
                str(module["parameter_count"]),
            )

        print(direct_table)


def _load_donor_model(
    settings: Settings,
    spec: ModelManipulationSpecification,
    dtype: torch.dtype,
) -> PreTrainedModel:
    revision_kwargs = {}
    if spec.source_model_commit is not None:
        revision_kwargs["revision"] = spec.source_model_commit

    max_memory = (
        {int(k) if k.isdigit() else k: v for k, v in settings.max_memory.items()}
        if settings.max_memory
        else None
    )

    return get_model_class(spec.source_model).from_pretrained(
        spec.source_model,
        dtype=dtype,
        device_map=settings.device_map,
        max_memory=max_memory,
        trust_remote_code=True,
        **revision_kwargs,
    )


def apply_manipulations(settings: Settings, model: Model) -> PreTrainedModel:
    if not settings.manipulations:
        raise ValueError("No model manipulation operations were configured.")

    print("* Preparing editable base model...")
    base_model = model.get_merged_model()

    for operation_index, spec in enumerate(settings.manipulations, start=1):
        print()
        print(
            f"Applying manipulation [bold]{operation_index}[/]/[bold]{len(settings.manipulations)}[/]: "
            f"[bold]{spec.operation.value}[/] from [bold]{spec.source_model}[/]"
        )

        donor_model = _load_donor_model(settings, spec, base_model.dtype)

        base_targets = _select_targets(base_model, spec)
        donor_targets = _select_targets(donor_model, spec)

        module_count = 0
        for layer_index, layer_targets in base_targets.items():
            donor_layer_targets = {
                _target_key(target): target for target in donor_targets[layer_index]
            }

            for base_target in layer_targets:
                donor_target = donor_layer_targets.get(_target_key(base_target))
                if donor_target is None:
                    expert = (
                        ""
                        if base_target.expert_index is None
                        else f" expert {base_target.expert_index}"
                    )
                    raise ValueError(
                        f"Donor model is missing {base_target.component}{expert} in layer {layer_index}"
                    )

                expert = (
                    ""
                    if base_target.expert_index is None
                    else f".expert.{base_target.expert_index}"
                )
                _apply_module_operation(
                    base_target.module,
                    donor_target.module,
                    spec,
                    f"layers.{layer_index}.{base_target.component}{expert}",
                )
                module_count += 1

        base_direct_targets = _direct_targets(base_model, spec.module_patterns)
        donor_direct_targets = _direct_targets(donor_model, spec.module_patterns)

        for name, base_module in base_direct_targets.items():
            donor_module = donor_direct_targets.get(name)
            if donor_module is None:
                raise ValueError(f"Donor model is missing direct module {name}")
            _apply_module_operation(base_module, donor_module, spec, name)
            module_count += 1

        if module_count == 0:
            raise ValueError("Manipulation did not match any modules.")

        print(f"* Modified [bold]{module_count}[/] modules")

        del donor_model
        empty_cache()

    return base_model
