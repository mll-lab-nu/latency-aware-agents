"""
metrics.py

Utility classes defining a Metrics container and multiple Trackers to enable model/stage-specific logging to various
endpoints (e.g., JSONL local logs, Weights & Biases).
"""

from typing import Tuple
import re
import json
import numpy as np
import torch
import torch.distributed as dist
from transformers import get_scheduler

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from starVLA.training.trainer_utils.config_tracker import reload_module_paths

logger = get_logger(__name__)


def build_accelerator(cfg, *, use_deepspeed: bool) -> Accelerator:
    accelerator_kwargs = {
        "gradient_accumulation_steps": cfg.trainer.gradient_accumulation_steps,
    }
    if use_deepspeed:
        accelerator_kwargs["deepspeed_plugin"] = DeepSpeedPlugin()

    accelerator = Accelerator(**accelerator_kwargs)
    accelerator.print(accelerator.state)
    return accelerator


_VOCAB_SIZED_WEIGHT_SUFFIXES = (
    "embed_tokens.weight",
    "lm_head.weight",
)


def _adapt_vocab_sized_weights(module, state_dict):
    """Copy overlapping vocab rows when tokenizer-sized Qwen weights changed."""
    target_state_dict = module.state_dict()
    adapted = dict(state_dict)
    adapted_keys = []
    for key, source_tensor in state_dict.items():
        target_tensor = target_state_dict.get(key)
        if target_tensor is None or not hasattr(source_tensor, "shape") or not hasattr(target_tensor, "shape"):
            continue
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            continue
        if not key.endswith(_VOCAB_SIZED_WEIGHT_SUFFIXES):
            continue
        if source_tensor.ndim != 2 or target_tensor.ndim != 2:
            continue
        if source_tensor.shape[1] != target_tensor.shape[1]:
            continue

        patched_tensor = target_tensor.detach().clone()
        rows = min(source_tensor.shape[0], target_tensor.shape[0])
        patched_tensor[:rows].copy_(source_tensor[:rows].to(dtype=patched_tensor.dtype, device=patched_tensor.device))
        adapted[key] = patched_tensor
        adapted_keys.append((key, tuple(source_tensor.shape), tuple(target_tensor.shape), rows))
    return adapted, adapted_keys


# === Define Tracker Interface ===
#

# utils/cli_parser.py


def normalize_dotlist_args(args):
    """
    Convert ['--x.y', 'val'] and ['--flag'] → ['x.y=val', 'flag=true']
    """
    normalized = []
    skip = False
    for i in range(len(args)):
        if skip:
            skip = False
            continue

        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                normalized.append(key)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                normalized.append(f"{key}={args[i + 1]}")
                skip = True
            else:
                normalized.append(f"{key}=true")
        else:
            pass  # skip orphaned values
    return normalized


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Build AdamW optimizer + LR scheduler from cfg.trainer.

    Supports:
    - Per-module learning rates via cfg.trainer.learning_rate
    - cosine_with_min_lr (or any transformers scheduler) via cfg.trainer.lr_scheduler_type
    """
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    fused_available = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=fused_available,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = cfg.trainer.scheduler_specific_kwargs
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


def _qwen_vl_interface(model):
    return model.qwen_vl_interface


def _llm_language_model(model):
    return _qwen_vl_interface(model).model.model.language_model


def _vit_module(model):
    # Qwen-VL keeps the visual tower and its vision-to-language connector/merger
    # under this module.
    return _qwen_vl_interface(model).model.model.visual


def _llm_decoder_layers(model):
    return _llm_language_model(model).layers


def _tied_embedding_modules(model):
    language_model = _llm_language_model(model)
    hf_model = _qwen_vl_interface(model).model
    return language_model.embed_tokens, hf_model.lm_head


def resolve_freeze_llm_layer_indices(model, cfg):
    """Translate ``trainer.freeze_llm_layers=[start,end]`` to an inclusive layer list."""
    freeze_llm_layers = cfg.trainer.freeze_llm_layers
    if not freeze_llm_layers:
        return [], len(_llm_decoder_layers(model))
    start, end = [int(value) for value in freeze_llm_layers]
    return list(range(start, end + 1)), len(_llm_decoder_layers(model))


def _module_param_ids(module):
    return {id(param) for param in module.parameters()}


def _has_vit_or_llm_freeze_request(cfg) -> bool:
    return bool(cfg.trainer.freeze_vit or cfg.trainer.freeze_tied_embedding or cfg.trainer.freeze_llm_layers)


def vit_llm_frozen_param_ids(model, cfg):
    """Parameter ids excluded from optimizer by trainer freeze controls."""
    if not _has_vit_or_llm_freeze_request(cfg):
        return set()

    frozen_params = set()
    freeze_vit = cfg.trainer.freeze_vit
    if freeze_vit:
        frozen_params.update(_module_param_ids(_vit_module(model)))

    if cfg.trainer.freeze_tied_embedding:
        for module in _tied_embedding_modules(model):
            frozen_params.update(_module_param_ids(module))

    frozen_layer_indices, _ = resolve_freeze_llm_layer_indices(model, cfg)
    layers = list(_llm_decoder_layers(model))
    for layer_index in frozen_layer_indices:
        frozen_params.update(_module_param_ids(layers[layer_index]))

    return frozen_params


def build_param_lr_groups(model, cfg):
    """
    build multiple param groups based on cfg.trainer.learning_rate.
    support specifying different learning rates for different modules, the rest use base.

    Args:
        vla: nn.Module model object
        cfg: config object, requires cfg.trainer.learning_rate dictionary

    Returns:
        List[Dict]: param_groups that can be used to build optimizer with torch.optim
    """

    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 1e-4)  # default base learning rate

    freeze_modules = cfg.trainer.get("freeze_modules", "")
    if not isinstance(freeze_modules, str):
        freeze_modules = ""
    freeze_patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]

    used_params = set()
    frozen_params = set()
    param_groups = []

    for freeze_path in freeze_patterns:
        module = model
        try:
            for attr in freeze_path.split("."):
                module = getattr(module, attr)
            frozen_params.update(id(p) for p in module.parameters())
        except AttributeError:
            print(f"⚠️ freeze module path does not exist: {freeze_path}")
            continue

    # Exclude trainer freeze controls so frozen parameters never enter an
    # optimizer param group.
    frozen_params.update(vit_llm_frozen_param_ids(model, cfg))

    for module_name, lr in lr_cfg.items():
        if module_name == "base":
            continue
        # try to find the module under vla by module_name (support nested paths)
        module = model
        try:
            for attr in module_name.split("."):
                module = getattr(module, attr)
            # filter out frozen parameters
            params = [p for p in module.parameters() if id(p) not in frozen_params]
            if params:  # only add param group if there are trainable parameters
                param_groups.append({"params": params, "lr": lr, "name": module_name})
                used_params.update(id(p) for p in params)
        except AttributeError:
            ReferenceError(f"⚠️ module path `{module_name}` not found in vla")

    # assign base learning rate to the remaining unused parameters (exclude frozen ones)
    other_params = [p for p in model.parameters() if id(p) not in used_params and id(p) not in frozen_params]
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})

    return param_groups


import torch.distributed as dist


def only_main_process(func):
    """
    decorator: only run in main process (rank=0)
    """

    def wrapper(*args, **kwargs):
        if dist.is_initialized() and dist.get_rank() != 0:
            return None  # non-main process does not execute
        return func(*args, **kwargs)

    return wrapper


from torchvision.ops import box_iou
from PIL import Image


def resize_images(images, target_size=(224, 224)):
    """
    recursively resize all images in the nested list.

    :param images: nested list of images or single image.
    :param target_size: target size (width, height) after resizing.
    :return: resized images list, keeping the original nested structure.
    """
    if isinstance(images, Image.Image):  # if it is a single PIL image
        return images.resize(target_size)
    elif isinstance(images, list):  # if it is a list, recursively process each element
        return [resize_images(img, target_size) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")


def stitch_frames(frames, grid=(2, 2), size=(224, 224)):
    """Tile temporal frames into one image, keeping the token count of a single image.

    Used by the ``stitch`` multi-image mode: instead of feeding N frames as N
    tokenized images, they are laid out on a fixed grid and the whole canvas is
    returned at ``size``. Both training (``_pack_sample``) and eval
    (``policy/starvla.py``) call this so the layouts match.

    Args:
        frames: list of ``PIL.Image`` or ``np.ndarray`` (H, W, C), oldest first.
        grid: (rows, cols). Cells beyond ``len(frames)`` repeat the last frame so
            the layout is fixed regardless of how many frames are available.
        size: (width, height) of the returned canvas.
    """
    rows, cols = grid
    canvas_w, canvas_h = size
    cell_w, cell_h = canvas_w // cols, canvas_h // rows
    pil = [f if isinstance(f, Image.Image) else Image.fromarray(f) for f in frames]
    capacity = rows * cols
    pil = (pil + [pil[-1]] * (capacity - len(pil))) if len(pil) < capacity else pil[:capacity]
    canvas = Image.new("RGB", (canvas_w, canvas_h))
    for idx, frame in enumerate(pil):
        r, c = divmod(idx, cols)
        canvas.paste(frame.convert("RGB").resize((cell_w, cell_h)), (c * cell_w, r * cell_h))
    return canvas


import torch.distributed as dist


class TrainerUtils:
    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        """
        directly freeze the specified submodules based on the relative module path list (patterns), no longer recursively find all submodule names:
          - patterns: read from config.trainer.freeze_modules, separated by commas to get the "relative path" list
            for example "qwen_vl_interface, action_model.net",
            it means to freeze model.qwen_vl_interface and model.action_model.net.

        Args:
            model: nn.Module model object
            freeze_modules: relative module path list (patterns)

        Returns:
            model: nn.Module model object
        return:
          - model:
        """
        frozen = []
        print("#"*30)
        print(freeze_modules)
        if freeze_modules and type(freeze_modules) == str:
            # split and remove whitespace
            patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()] if freeze_modules else []

            for path in patterns:
                # split the "relative path" by dots, for example "action_model.net" → ["action_model", "net"]
                attrs = path.split(".")
                module = model
                try:
                    for attr in attrs:
                        module = getattr(module, attr)
                    # if the module is successfully get, freeze it and its all submodule parameters
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen.append(path)
                except AttributeError:
                    # if the attribute does not exist, skip and print warning
                    print(f"⚠️ module path does not exist, cannot freeze: {path}")
                    continue

        # accelerator.wait_for_everyone()  # synchronize when distributed training
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"🔒 Frozen modules with re pattern: {frozen}")
        return model

    @staticmethod
    def freeze_vit_and_llm_layers(model, cfg):
        """Apply trainer.freeze_vit/freeze_tied_embedding/freeze_llm_layers.

        trainer.freeze_vit freezes Qwen-VL's visual module, including the
        vision-to-language connector/merger it contains.
        """
        if not _has_vit_or_llm_freeze_request(cfg):
            return model

        freeze_vit = cfg.trainer.freeze_vit
        if freeze_vit:
            for param in _vit_module(model).parameters():
                param.requires_grad = False

        freeze_tied_embedding = cfg.trainer.freeze_tied_embedding
        if freeze_tied_embedding:
            for module in _tied_embedding_modules(model):
                for param in module.parameters():
                    param.requires_grad = False

        frozen_layer_indices, total = resolve_freeze_llm_layer_indices(model, cfg)
        layers = list(_llm_decoder_layers(model))
        for layer_index in frozen_layer_indices:
            for param in layers[layer_index].parameters():
                param.requires_grad = False

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                "🔒 freeze_vit="
                f"{freeze_vit}, freeze_tied_embedding={freeze_tied_embedding}, "
                f"freeze_llm_layers={list(frozen_layer_indices)}"
                f" / total={total}"
            )
        return model

    @staticmethod
    def print_trainable_parameters(model):
        """
        print the total number of parameters and trainable parameters of the model
        :param model: PyTorch model instance
        """
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        print("📊 model parameter statistics:")
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
        )
        return num_params, num_trainable_params

    @staticmethod
    def load_pretrained_backbones(model, checkpoint_path=None, reload_modules=None):
        """
        load checkpoint:
        - if reload_modules is set, load by path part
        - otherwise → load the entire model parameters (overwrite model)

        return:
            replace, loaded_modules: list of module paths that successfully loaded parameters; if global load, then ["<full_model>"]
        """
        if not checkpoint_path:
            return []
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"📦 loading checkpoint: {checkpoint_path}")
        try:
            if _is_safetensors_path(checkpoint_path):
                from safetensors.torch import load_file

                checkpoint = load_file(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"❌ loading checkpoint failed: {e}")

        loaded_modules = []

        module_paths = reload_module_paths(reload_modules)
        if module_paths:  # partial load
            for path in module_paths:
                reload_path = path.split(".")
                module = model
                try:
                    for module_name in reload_path:  # find the module to modify level by level
                        module = getattr(module, module_name)
                    prefix = path + "."
                    sub_state_dict = {k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)}
                    if sub_state_dict:
                        sub_state_dict, adapted_keys = _adapt_vocab_sized_weights(module, sub_state_dict)
                        for key, source_shape, target_shape, rows in adapted_keys:
                            if not dist.is_initialized() or dist.get_rank() == 0:
                                print(
                                    f"⚠️ adapted vocab-sized weight '{path}.{key}' from checkpoint shape "
                                    f"{source_shape} to current shape {target_shape}; copied {rows} rows"
                                )
                        module.load_state_dict(sub_state_dict, strict=True)
                        if not dist.is_initialized() or dist.get_rank() == 0:
                            print(f"✅ parameters loaded to module '{path}'")
                        loaded_modules.append(path)
                    else:
                        print(f"⚠️ parameters not found in checkpoint '{path}'")
                except AttributeError:
                    print(f"❌ cannot find module path: {path}")
        else:  # full load
            try:
                checkpoint, adapted_keys = _adapt_vocab_sized_weights(model, checkpoint)
                for key, source_shape, target_shape, rows in adapted_keys:
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        print(
                            f"⚠️ adapted vocab-sized weight '{key}' from checkpoint shape "
                            f"{source_shape} to current shape {target_shape}; copied {rows} rows"
                        )
                model.load_state_dict(checkpoint, strict=False)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print("✅ loaded <full_model> model parameters")
                loaded_modules = ["<full_model>"]
            except Exception as e:
                raise RuntimeError(f"❌ loading full model failed: {e}")
        return model

    @staticmethod
    def print_freeze_status(model):
        """
        print the freezing status of each parameter in the model
        :param model: PyTorch model instance
        """
        for name, param in model.named_parameters():
            status = "Frozen" if not param.requires_grad else "Trainable"
            print(f"{name:60s}  |  {status}")

    @staticmethod
    def setup_distributed_training(accelerator, *components):
        """
        use Accelerator to prepare distributed training components
        :param accelerator: Accelerate instance
        :param components: any number of components (such as model, optimizer, dataloader, etc.)
        :return: prepared distributed components (in the same order as input)
        """

        # use accelerator.prepare method to wrap components
        prepared_components = accelerator.prepare(*components)
        return prepared_components

    @staticmethod
    def euclidean_distance(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.linalg.norm(predicted - ground_truth)

    @staticmethod
    def _reset_dataloader(dataloader, epoch_counter):
        """safe reset dataloader iterator"""
        # 1. update epoch counter
        epoch_counter += 1

        dataset = getattr(dataloader, "dataset", None)
        if dataset is not None and callable(getattr(dataset, "set_epoch", None)):
            dataset.set_epoch(epoch_counter)

        # 2. set new epoch (distributed core)
        if hasattr(dataloader, "sampler") and callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch_counter)

        # 3. create new iterator
        return iter(dataloader), epoch_counter

    @staticmethod
    def compute_grad_angle_with_stats(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> Tuple[float, float]:
        """
        compute the cosine angle between two groups of gradient vectors (degrees), and calculate the average angle and variance.
        grads_a, grads_v: gradient Tensor list corresponding to the same parameter list interface_params
        return:
            mean_angle_deg: average angle (degrees)
            angle_variance: angle variance
        """
        angle_degs = []

        # compute the cosine angle between each gradient block grads_a[0].shape = 1280, 3, 14, 14
        # grads_1 = grads_a[0][0]  # [3, 14, 14]
        # grads_2 = grads_v[0][0]
        # grads_a = grads_1.view(-1, 3)  # reshape to [196, 3]
        # grads_v = grads_2.view(-1, 3)

        # lang linear
        # reshape to 14*14, 3
        # layer
        grads_action = grads_a[0]  # [2048, 11008]
        grads_action = grads_action[
            :32, :7
        ]  # only take the first 7 elements, avoid cosim failure in high-dimensional space
        grads_vl = grads_v[0]  # [2048, 11008]
        grads_vl = grads_vl[
            :32, :7
        ]  # only take the first 32 elements, 7 dimensions, avoid cosim failure in high-dimensional space
        for g_a, g_v in zip(grads_action, grads_vl):
            dot = torch.sum(g_a * g_v)
            norm_a_sq = torch.sum(g_a * g_a)
            norm_v_sq = torch.sum(g_v * g_v)

            # avoid division by zero
            norm_a = torch.sqrt(norm_a_sq + 1e-16)
            norm_v = torch.sqrt(norm_v_sq + 1e-16)

            cos_sim = (dot / (norm_a * norm_v)).clamp(-1.0, 1.0)
            angle_rad = torch.acos(cos_sim)
            angle_deg = angle_rad * (180.0 / torch.pi)

            angle_degs.append(angle_deg.item())

        # compute the average angle and variance
        angle_degs_tensor = torch.tensor(angle_degs)
        mean_angle_deg = torch.mean(angle_degs_tensor).item()
        angle_variance = torch.sqrt(torch.var(angle_degs_tensor)).item()
        # accelerator.wait_for_everyone()
        return mean_angle_deg, angle_variance

    @staticmethod
    def pcgrad_project(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        apply PCGrad projection to the second group of gradients grads_v, suppress negative transfer between grads_a and grads_v
        if the dot product of two groups of gradients < 0, then:
            grads_v <- grads_v - (dot / ||grads_a||^2) * grads_a
        return the new grads_v list
        """
        # first compute dot and ||grads_a||^2
        dot, norm_a_sq = 0.0, 0.0
        for g_a, g_v in zip(grads_a, grads_v):
            dot += torch.sum(g_a * g_v)
            norm_a_sq += torch.sum(g_a * g_a)

        if dot < 0:
            coeff = dot / (norm_a_sq + 1e-6)
            # projection
            grads_v = [g_v - coeff * g_a for g_a, g_v in zip(grads_a, grads_v)]

        return grads_v

    @staticmethod
    def eval_qwenpi(qwenpi, dataloader, num_batches=20):
        """
        evaluate QwenQFormerDiT model, compute IoU and action distance.

        Args:
            qwenpi: QwenQFormerDiT model instance.
            dataloader: data loader.
            num_batches: number of batches to evaluate.

        Returns:
            dict: contains IoU and action distance evaluation results.
        """
        iou_scores = []
        action_distances = []
        count = 0

        dataset_iter = iter(dataloader)
        while count < num_batches:
            try:
                batch_samples = next(dataset_iter)
                count += 1
            except StopIteration:
                break

            # extract data
            images = [example["image"] for example in batch_samples]
            instructions = [example["lang"] for example in batch_samples]
            actions = [example["action"] for example in batch_samples]
            solutions = [example["solution"] for example in batch_samples]

            # model prediction
            predicted_solutions, normalized_actions = qwenpi.predict_action_withCoT(
                images=images, instructions=instructions, use_ddim=False, num_ddim_steps=20
            )

            # extract and convert predicted results
            parsed_solutions = []
            for solution in predicted_solutions:
                parsed_solution = TrainerUtils.extract_json_from_string(solution)
                parsed_solutions.append(parsed_solution)

            # compute IoU
            for pred_dict, gt_dict in zip(parsed_solutions, solutions):
                pred_pick_bbox = torch.tensor(pred_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_pick_bbox = torch.tensor(gt_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                pred_place_bbox = torch.tensor(pred_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_place_bbox = torch.tensor(gt_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)

                pick_iou = box_iou(pred_pick_bbox, gt_pick_bbox).item()
                place_iou = box_iou(pred_place_bbox, gt_place_bbox).item()

                iou_scores.append({"pick_iou": pick_iou, "place_iou": place_iou})

            # compute action distance
            actions = np.array(actions)  # convert to numpy array
            num_pots = np.prod(actions.shape)  # B*len*dim
            action_distance = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_action_distance = action_distance / num_pots
            action_distances.append(average_action_distance)

        # summarize results
        avg_action_distance = np.mean(action_distances)
        return {"iou_scores": iou_scores, "average_action_distance": avg_action_distance}

    @staticmethod
    def extract_json_from_string(input_string):
        """
        extract valid JSON part from string and convert to dictionary.

        Args:
            input_string (str): string containing extra characters.

        Returns:
            dict: dictionary extracted and parsed.
        """
        json_match = re.search(r"{.*}", input_string, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON decode failed: {e}")
                return None
        else:
            print("No valid JSON part found")
            return None

    def _get_latest_checkpoint(self, checkpoint_dir):
        """Find the latest checkpoint in the directory based on step number."""
        if not os.path.exists(checkpoint_dir):
            self.accelerator.print(f"No checkpoint directory found at {checkpoint_dir}")
            return None, 0

        checkpoints = []
        for name in os.listdir(checkpoint_dir):
            path = os.path.join(checkpoint_dir, name)
            state_match = re.match(r"steps_(\d+)_state$", name)
            if state_match and os.path.isdir(path):
                checkpoints.append((name, int(state_match.group(1)), 1))
                continue
            file_match = re.match(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$", name)
            if file_match and os.path.isfile(path):
                checkpoints.append((name, int(file_match.group(1)), 0))

        if not checkpoints:
            self.accelerator.print(f"No checkpoints found in {checkpoint_dir}")
            return None, 0

        # Sort by step number, preferring full training-state directories over
        # model-only files when both exist for the same step.
        checkpoints.sort(key=lambda x: (x[1], x[2]))
        latest_checkpoint, completed_steps, _ = checkpoints[-1]

        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        self.accelerator.print(f"Latest checkpoint found: {latest_checkpoint_path}")
        return latest_checkpoint_path, completed_steps

import os


def is_main_process():
    rank = int(os.environ.get("RANK", 0))  # if RANK is not set, default to 0
    return rank == 0


def _is_safetensors_path(path):
    """Check if a path refers to a safetensors file."""
    return str(path).endswith(".safetensors")
