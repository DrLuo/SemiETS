import pickle, os
from fvcore.common.file_io import PathManager
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts
from fvcore.common.checkpoint import _strip_prefix_if_present, _IncompatibleKeys


class AdetCheckpointer(DetectionCheckpointer):
    """
    Same as :class:`DetectronCheckpointer`, but is able to convert models
    in AdelaiDet, such as LPF backbone.
    """
    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                if "weight_order" in data:
                    del data["weight_order"]
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}

        basename = os.path.basename(filename).lower()
        if "lpf" in basename or "dla" in basename:
            loaded["matching_heuristics"] = True
        return loaded



class AdetTSCheckpointer(AdetCheckpointer):
    """
    Same as :class:`AdetCheckpointer`,
    but is able to load the whole model or only teacher and student modal
    """
    def _load_model(self, checkpoint):
        if checkpoint.get("student", None) == True:
            # pretrained model weight: only update student model
            if checkpoint.get("matching_heuristics", False):
                self._convert_ndarray_to_tensor(checkpoint["model"])
                # convert weights by name-matching heuristics
                checkpoint["model"] = align_and_update_state_dicts(
                    self.model.student.state_dict(),
                    checkpoint["model"],
                    c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
                )

            # for non-caffe2 models, use standard ways to load it
            incompatible = self._load_student_model(checkpoint)

            model_buffers = dict(self.model.student.named_buffers(recurse=False))
            for k in ["pixel_mean", "pixel_std"]:
                # Ignore missing key message about pixel_mean/std.
                # Though they may be missing in old checkpoints, they will be correctly
                # initialized from config anyway.
                if k in model_buffers:
                    try:
                        incompatible.missing_keys.remove(k)
                    except ValueError:
                        pass
            for k in incompatible.unexpected_keys[:]:
                # Ignore unexpected keys about cell anchors. They exist in old checkpoints
                # but now they are non-persistent buffers and will not be in semi_5s checkpoints.
                if "anchor_generator.cell_anchors" in k:
                    incompatible.unexpected_keys.remove(k)
            return incompatible



        else:  # whole model
            if checkpoint.get("matching_heuristics", False):
                self._convert_ndarray_to_tensor(checkpoint["model"])
                # convert weights by name-matching heuristics
                checkpoint["model"] = align_and_update_state_dicts(
                    self.model.state_dict(),
                    checkpoint["model"],
                    c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
                )
            # for non-caffe2 models, use standard ways to load it
            incompatible = super()._load_model(checkpoint)

            model_buffers = dict(self.model.named_buffers(recurse=False))
            for k in ["pixel_mean", "pixel_std"]:
                # Ignore missing key message about pixel_mean/std.
                # Though they may be missing in old checkpoints, they will be correctly
                # initialized from config anyway.
                if k in model_buffers:
                    try:
                        incompatible.missing_keys.remove(k)
                    except ValueError:
                        pass
            for k in incompatible.unexpected_keys[:]:
                # Ignore unexpected keys about cell anchors. They exist in old checkpoints
                # but now they are non-persistent buffers and will not be in semi_5s checkpoints.
                if "anchor_generator.cell_anchors" in k:
                    incompatible.unexpected_keys.remove(k)
            return incompatible

    def _load_student_model(self, checkpoint) -> _IncompatibleKeys:  # pyre-ignore
        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)

        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching.
        _strip_prefix_if_present(checkpoint_state_dict, "module.")

        # work around https://github.com/pytorch/pytorch/issues/24139
        model_state_dict = self.model.student.state_dict()
        incorrect_shapes = []
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    checkpoint_state_dict.pop(k)
        # pyre-ignore
        incompatible = self.model.modelStudent.load_state_dict(
            checkpoint_state_dict, strict=False
        )
        return _IncompatibleKeys(
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            incorrect_shapes=incorrect_shapes,
        )