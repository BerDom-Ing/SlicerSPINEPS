"""Inference parameters for a SPINEPS run."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Semantic models shipped by SPINEPS, keyed by the acquisition they were trained on.
# See https://github.com/Hendrik-code/spineps -- the semantic model must match the
# modality of the input image, the instance and labeling models are modality agnostic.
SEMANTIC_MODEL_T2W = "t2w"
SEMANTIC_MODEL_T1W = "t1w"
SEMANTIC_MODEL_VIBE = "vibe"
SEMANTIC_MODEL_CT = "ct"

SEMANTIC_MODELS = [
    SEMANTIC_MODEL_T2W,
    SEMANTIC_MODEL_T1W,
    SEMANTIC_MODEL_VIBE,
    SEMANTIC_MODEL_CT,
]

# BIDS suffix expected by SPINEPS for each semantic model. The input file is named
# using this suffix so that SPINEPS' BIDS parser recognises the modality on its own
# and no compatibility override flag is needed.
SEMANTIC_MODEL_TO_BIDS_SUFFIX = {
    SEMANTIC_MODEL_T2W: "T2w",
    SEMANTIC_MODEL_T1W: "T1w",
    SEMANTIC_MODEL_VIBE: "vibe",
    SEMANTIC_MODEL_CT: "ct",
}

# The instance and labeling models are NOT modality agnostic: SPINEPS ships CT-specific
# variants (spineps/utils/auto_download.py -> instances["ct_instance"],
# labeling["ct_labeling"]) trained separately from the sagittal-MRI ones. The `sample`
# subcommand has no "auto" resolution -- only `dataset` does -- so it silently accepts
# the MRI instance model on a CT and produces a poor segmentation rather than an error.
# Pair them here instead.
SEMANTIC_MODEL_TO_COMPANION_MODELS = {
    SEMANTIC_MODEL_CT: ("ct_instance", "ct_labeling"),
}
DEFAULT_COMPANION_MODELS = ("instance", "t2w_labeling")

# BIDS "acq" entity written into the exported file name. SPINEPS checks it against the
# acquisitions the model accepts (spineps/seg_utils.py, check_input_model_compatibility)
# and refuses to run on a mismatch. "iso" is accepted by every model, which is the right
# claim for a CT; the MRI models are trained on sagittal acquisitions.
SEMANTIC_MODEL_TO_BIDS_ACQUISITION = {
    SEMANTIC_MODEL_CT: "iso",
}
DEFAULT_BIDS_ACQUISITION = "sag"

DERIVATIVES_FOLDER_NAME = "derivatives_seg"


def defaultModelsFolder() -> Path:
    """Return the folder SPINEPS should store its model weights in.

    Honours SPINEPS_SEGMENTOR_MODELS when the deployment already sets it, so that an
    offline machine can be pre-seeded with the weights without touching the code.
    """
    override = os.environ.get("SPINEPS_SEGMENTOR_MODELS")
    if override:
        return Path(override)

    import slicer  # noqa: PLC0415 - keeps this module importable outside Slicer

    return Path(slicer.app.slicerUserSettingsFilePath).parent.joinpath("SPINEPS", "models")


@dataclass
class Parameter:
    """Parameters for one ``spineps sample`` invocation.

    :param semanticModel: one of :data:`SEMANTIC_MODELS`; must match the input modality.
    :param instanceModel: vertebra instance segmentation model. Leave None to derive it
        from the semantic model (see :data:`SEMANTIC_MODEL_TO_COMPANION_MODELS`).
    :param labelingModel: vertebra labeling classifier. Leave None to derive it likewise.
    :param forceTwelveThoracic: pass ``-no_tltv_labeling``, forbidding the labeling model
        from claiming a thoracolumbar transitional anomaly. See
        :attr:`forceTwelveThoracic` below for when to use it.
    :param useCpu: force CPU inference. When None the device is auto-detected.
    :param modelsFolder: value for the SPINEPS_SEGMENTOR_MODELS environment variable.
        Weights are downloaded here on first use; pre-seed it for offline machines.
    :param verbose: pass -verbose to SPINEPS.
    :param extraArgs: escape hatch for SPINEPS flags not modelled here.
    """

    semanticModel: str = SEMANTIC_MODEL_T2W
    instanceModel: Optional[str] = None
    labelingModel: Optional[str] = None

    #: Forbid the thoracolumbar transitional-vertebra anomaly during labeling.
    #:
    #: SPINEPS labels vertebrae by searching for the most probable label sequence, and by
    #: default that search is allowed to skip the class after T11 -- i.e. to conclude the
    #: subject has no T12 (``allow_skip_at_class = [T11_CLASS_IDX]`` in
    #: ``spineps/phase_labeling.py``). On a full-spine scan that correctly models real
    #: transitional anatomy. On a lumbar-FOV MR, where only one or two thoracic vertebrae
    #: are visible, the classifier has little context and the skip shows up as a hole in
    #: the label sequence with no matching gap in the anatomy, shifting every thoracic
    #: label by one.
    #:
    #: Left False by default: forcing 12 thoracic vertebrae onto a patient who genuinely
    #: has a variant is the same error in the other direction. Turn it on only when the
    #: vertebra spacing shows the levels are continuous across the skipped label.
    forceTwelveThoracic: bool = False

    useCpu: Optional[bool] = None
    modelsFolder: Optional[Path] = None
    verbose: bool = False
    extraArgs: List[str] = field(default_factory=list)

    def isValid(self) -> bool:
        return self.semanticModel in SEMANTIC_MODELS

    def _companionModels(self):
        return SEMANTIC_MODEL_TO_COMPANION_MODELS.get(
            self.semanticModel, DEFAULT_COMPANION_MODELS
        )

    def resolvedInstanceModel(self) -> str:
        return self.instanceModel or self._companionModels()[0]

    def resolvedLabelingModel(self) -> str:
        return self.labelingModel or self._companionModels()[1]

    def bidsSuffix(self) -> str:
        return SEMANTIC_MODEL_TO_BIDS_SUFFIX.get(self.semanticModel, "T2w")

    def bidsAcquisition(self) -> str:
        return SEMANTIC_MODEL_TO_BIDS_ACQUISITION.get(
            self.semanticModel, DEFAULT_BIDS_ACQUISITION
        )

    def inputFileName(self) -> str:
        """Return a BIDS compliant file name for the exported input volume.

        SPINEPS derives the modality and acquisition from the file name. Naming the
        file correctly avoids having to pass compatibility override flags.
        """
        return f"sub-spineps_acq-{self.bidsAcquisition()}_{self.bidsSuffix()}.nii.gz"

    def asArgList(self, inputFile: Path) -> List[str]:
        """Build the ``spineps sample`` argument list for the given input file."""
        if not self.isValid():
            raise RuntimeError(
                f"Unknown SPINEPS semantic model '{self.semanticModel}'. "
                f"Expected one of {SEMANTIC_MODELS}."
            )

        args = [
            "sample",
            "-i", Path(inputFile).as_posix(),
            "-model_semantic", self.semanticModel,
            "-model_instance", self.resolvedInstanceModel(),
            "-model_labeling", self.resolvedLabelingModel(),
            "-der_name", DERIVATIVES_FOLDER_NAME,
        ]

        if self.forceTwelveThoracic:
            args.append("-no_tltv_labeling")
        if self.useCpu:
            args.append("-cpu")
        if self.verbose:
            args.append("-verbose")

        args.extend(self.extraArgs)
        return args

    def debugString(self) -> str:
        return (
            f"semanticModel={self.semanticModel}, "
            f"instanceModel={self.resolvedInstanceModel()}, "
            f"labelingModel={self.resolvedLabelingModel()}, "
            f"forceTwelveThoracic={self.forceTwelveThoracic}, useCpu={self.useCpu}, "
            f"modelsFolder={self.modelsFolder}"
        )
