"""SPINEPS whole-spine segmentation module for 3D Slicer.

The widget is a thin front end over SlicerSPINEPSLib: it owns the dependency install
and offers a plain run-and-load UI. The library underneath is the real interface, and
is meant to be driven directly by other modules that need a spine segmentation -- see
"Using it from another module" in the README.

SPINEPS: https://github.com/Hendrik-code/spineps (Apache-2.0)
"""

from pathlib import Path

import qt
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)

from SlicerSPINEPSLib import (
    SEMANTIC_MODELS,
    InstallLogic,
    Parameter,
    SegmentationLogic,
    defaultModelsFolder,
)


class SlicerSPINEPS(ScriptedLoadableModule):
    """Module metadata."""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SPINEPS"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Bernardo Dominguez"]
        self.parent.helpText = (
            "Whole-spine segmentation of MRI (T2w, T1w, Vibe) and CT using SPINEPS.\n\n"
            "Requires 3D Slicer 5.10 or newer. No other extension is needed: press "
            "'Install SPINEPS dependencies' once before the first run and everything, "
            "including a PyTorch build matched to this machine, is installed for you."
        )
        self.parent.acknowledgementText = (
            "This module was developed by Bernardo Dominguez.\n\n"
            "It wraps SPINEPS, developed by Hendrik Moeller et al. "
            "(https://github.com/Hendrik-code/spineps). Please cite the SPINEPS paper "
            "if you use this module in your research."
        )


class SlicerSPINEPSWidget(ScriptedLoadableModuleWidget):
    """Minimal setup and debugging UI."""

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = None
        self.installLogic = None
        self._segmentationLogic = None

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.logic = SlicerSPINEPSLogic()
        self.installLogic = InstallLogic()
        self.installLogic.progressInfo.connect(self.onProgressInfo)

        self._segmentationLogic = self.logic.segmentationLogic
        self._segmentationLogic.progressInfo.connect(self.onProgressInfo)
        self._segmentationLogic.errorOccurred.connect(self.onErrorOccurred)
        self._segmentationLogic.inferenceFinished.connect(self.onInferenceFinished)

        formLayout = qt.QFormLayout()

        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = True
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        formLayout.addRow("Input volume:", self.inputSelector)

        self.semanticModelSelector = qt.QComboBox()
        self.semanticModelSelector.addItems(SEMANTIC_MODELS)
        formLayout.addRow("Semantic model:", self.semanticModelSelector)

        self.forceTwelveThoracicCheckBox = qt.QCheckBox()
        self.forceTwelveThoracicCheckBox.toolTip = (
            "Forbid SPINEPS from concluding the subject has no T12.\n\n"
            "Its labeling search may skip the level below T11 to model transitional "
            "anatomy. On a lumbar field of view, where only one or two thoracic "
            "vertebrae are visible, that skip is usually wrong and shifts every "
            "thoracic label by one level.\n\n"
            "Enable it when the vertebrae are evenly spaced across a gap in the label "
            "numbers. Leave it off if the patient may genuinely have a transitional "
            "vertebra."
        )
        formLayout.addRow("Assume 12 thoracic vertebrae:", self.forceTwelveThoracicCheckBox)

        self.deviceSelector = qt.QComboBox()
        self.deviceSelector.addItems(["Auto", "Force CPU"])
        formLayout.addRow("Device:", self.deviceSelector)

        self.layout.addLayout(formLayout)

        self.installButton = qt.QPushButton("Install SPINEPS dependencies")
        self.installButton.connect("clicked()", self.onInstallClicked)
        self.layout.addWidget(self.installButton)

        # Make the install outcome visible without anyone typing into a console:
        # a CPU-only torch on a CUDA machine has no symptom other than being slow.
        self.statusLabel = qt.QLabel()
        self.statusLabel.wordWrap = True
        self.layout.addWidget(self.statusLabel)

        # The weights are downloaded on first use, not by the installer, so an operator
        # who is about to work offline needs to be able to see whether they are present
        # before starting a case.
        self.weightsLabel = qt.QLabel()
        self.weightsLabel.wordWrap = True
        self.weightsLabel.textInteractionFlags = qt.Qt.TextSelectableByMouse
        self.layout.addWidget(self.weightsLabel)

        self.applyButton = qt.QPushButton("Run segmentation")
        self.applyButton.connect("clicked()", self.onApplyClicked)
        self.layout.addWidget(self.applyButton)

        self.cancelButton = qt.QPushButton("Cancel")
        self.cancelButton.enabled = False
        self.cancelButton.connect("clicked()", self.onCancelClicked)
        self.layout.addWidget(self.cancelButton)

        self.logTextEdit = qt.QTextEdit()
        self.logTextEdit.readOnly = True
        self.logTextEdit.setMinimumHeight(240)
        self.layout.addWidget(self.logTextEdit)

        self.layout.addStretch(1)
        self.updateInstallStatus()

    def cleanup(self):
        if self._segmentationLogic is not None:
            self._segmentationLogic.stopSegmentation()
        ScriptedLoadableModuleWidget.cleanup(self)

    def updateInstallStatus(self):
        self.weightsLabel.text = self.describeWeightsState()

        installedVersion = self.installLogic.getInstalledSpinepsVersion()
        if installedVersion is None:
            self.installButton.text = "Install SPINEPS dependencies"
            self.statusLabel.text = "SPINEPS is not installed yet."
            return

        self.installButton.text = f"Reinstall SPINEPS (installed: {installedVersion})"
        self.statusLabel.text = self.describeInstallState(installedVersion)

    def describeWeightsState(self) -> str:
        """Report whether the model weights are already on disk, and where.

        SPINEPS downloads them on first use rather than at install time, so an install
        can report success while the module is still unusable without a network. On an
        air-gapped machine this folder is what has to be copied across.
        """
        folder = self.logic.defaultModelsFolder()
        try:
            models = sorted(p.name for p in folder.iterdir() if p.is_dir())
        except OSError:
            models = []

        if not models:
            return (
                f"Model weights: none yet. The first segmentation downloads them "
                f"(a few hundred MB per model) into {folder} and needs an internet "
                "connection. To work offline, copy that folder from a machine that "
                "already has them."
            )
        return f"Model weights ({len(models)}) in {folder}: {', '.join(models)}"

    def describeInstallState(self, installedVersion) -> str:
        """Return a one-line, human-readable summary of what is installed.

        Deliberately never imports torch. Importing it loads DLLs that Windows then
        keeps locked, which makes any later pip uninstall or reinstall of torch fail
        halfway and leave a damaged install behind. Everything below is derived from
        package metadata instead.
        """
        status = InstallLogic.torchStatus()

        if status["damaged"]:
            return (
                "PyTorch is damaged (a previous install or uninstall did not complete).\n"
                "Close Slicer completely and repair it from a terminal - see the "
                "SlicerSPINEPS README - then reopen this module."
            )
        if not status["installed"]:
            return (
                f"SPINEPS {installedVersion}, but PyTorch is missing - press Reinstall "
                "to complete the installation."
            )

        if not status["cpuOnly"]:
            device = f"GPU-capable PyTorch {status['version']}"
        elif status["cudaGpuPresent"]:
            device = (
                f"CPU-only PyTorch {status['version']} - an NVIDIA GPU was detected but "
                "this build cannot use it. Segmentation will work, but slowly. See the "
                "README to switch to a CUDA build."
            )
        else:
            device = (
                f"CPU-only PyTorch {status['version']} (no NVIDIA GPU detected) - "
                "expect several minutes per scan."
            )

        importable = "imports OK" if InstallLogic.isSpinepsImportable() else "IMPORT FAILS"
        return f"SPINEPS {installedVersion} ({importable}). Inference device: {device}"

    def onInstallClicked(self):
        if InstallLogic.isTorchInstallDamaged():
            slicer.util.errorDisplay(
                "PyTorch is in a damaged state and cannot be repaired from inside Slicer, "
                "because this process holds its files open.\n\n"
                "Close Slicer completely, then run the repair commands documented in the "
                "SlicerSPINEPS README, and start Slicer again."
            )
            return

        # Force when SPINEPS is already present, otherwise the button labelled
        # "Reinstall" would return immediately without doing anything.
        force = self.installLogic.getInstalledSpinepsVersion() is not None

        with slicer.util.tryWithErrorDisplay("Failed to install SPINEPS.", waitCursor=True):
            self.installLogic.setupPythonRequirements(force=force)
            self.updateInstallStatus()

    def onApplyClicked(self):
        volumeNode = self.inputSelector.currentNode()
        if volumeNode is None:
            slicer.util.errorDisplay("Select an input volume first.")
            return

        parameter = Parameter(
            semanticModel=self.semanticModelSelector.currentText,
            useCpu=self.deviceSelector.currentText == "Force CPU",
            forceTwelveThoracic=self.forceTwelveThoracicCheckBox.checked,
            modelsFolder=self.logic.defaultModelsFolder(),
        )
        self._segmentationLogic.setParameter(parameter)

        self.logTextEdit.clear()
        self.applyButton.enabled = False
        self.cancelButton.enabled = True
        self._segmentationLogic.startSegmentation(volumeNode)

    def onCancelClicked(self):
        self._segmentationLogic.stopSegmentation()

    def onInferenceFinished(self, *_):
        self.applyButton.enabled = True
        self.cancelButton.enabled = False
        with slicer.util.tryWithErrorDisplay("Failed to load the SPINEPS result."):
            self._segmentationLogic.loadVertebraeSegmentation()

    def onErrorOccurred(self, text):
        self.applyButton.enabled = True
        self.cancelButton.enabled = False
        self.onProgressInfo(f"ERROR: {text}")

    def onProgressInfo(self, text):
        self.logTextEdit.append(str(text).rstrip())
        self.logTextEdit.ensureCursorVisible()
        slicer.app.processEvents()


class SlicerSPINEPSLogic(ScriptedLoadableModuleLogic):
    """Owns the shared SegmentationLogic instance and the weights location."""

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.segmentationLogic = SegmentationLogic()

    @staticmethod
    def defaultModelsFolder() -> Path:
        """Return the folder SPINEPS should store its weights in."""
        return defaultModelsFolder()


class SlicerSPINEPSTest(ScriptedLoadableModuleTest):
    """Test cases for the SPINEPS module."""

    def runTest(self):
        self.setUp()
        self.test_ParameterArgList()
        self.test_InstallLogicVersionGate()
        self.test_DenylistMatchesWholeNamesOnly()
        self.test_DependenciesSurviveDamagedMetadata()
        self.test_RawMasksAreNeverPickedUp()

    def test_RawMasksAreNeverPickedUp(self):
        """Regression test: the final mask must win over the intermediate one.

        SPINEPS writes ``output_raw_<format>/..._seg-vert-raw_msk.nii.gz`` next to the
        final ``..._seg-vert_msk.nii.gz``. The raw mask holds top-to-bottom instance ids
        instead of anatomical vertebra labels, so picking it up produces a segmentation
        that looks correct in 3D but labels L1 as "1" -- which silently breaks the
        anatomical vertebra naming that consumers key on. A plain sorted() picked it
        first, because "output_raw_" sorts before the final file name.
        """
        from SlicerSPINEPSLib.SegmentationLogic import SEG_VERT_PATTERN, SegmentationLogic

        logic = SegmentationLogic()
        root = Path(logic._tmpDir.path()).joinpath("derivatives_seg", "sub-spineps")
        final = root.joinpath("sub-spineps_acq-sag_mod-T2w_seg-vert_msk.nii.gz")
        raw = root.joinpath(
            "output_raw_nii.gz", "sub-spineps_acq-sag_mod-T2w_seg-vert-raw_msk.nii.gz"
        )
        for path in (final, raw):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        self.assertEqual(logic._findOutputFile(SEG_VERT_PATTERN, "vertebra instance"), final)
        self.delayDisplay("Raw mask exclusion test passed")

    def test_DenylistMatchesWholeNamesOnly(self):
        """Regression test: the skip list must match distribution names, not prefixes.

        "torchmetrics" and "pytorch-lightning" both contain "torch" but are ordinary
        SPINEPS dependencies. A prefix or substring test silently drops them, and the
        resulting install fails only later with ModuleNotFoundError.
        """
        from SlicerSPINEPSLib.InstallLogic import PACKAGES_TO_SKIP, isPackageToSkip

        mustInstall = [
            "torchmetrics (>=1.1.2,<2.0.0)",
            "pytorch-lightning (>=2.0.8,<3.0.0)",
            "TPTBox",
            "nnunetv2 (>=2.4.2,<3.0.0)",
            "requests-toolbelt",
        ]
        for requirement in mustInstall:
            self.assertFalse(
                isPackageToSkip(requirement, PACKAGES_TO_SKIP),
                f"{requirement} must be installed, not skipped",
            )

        mustSkip = ["torch (>=2.0)", "SimpleITK", "requests (>=2.0)", "torchvision"]
        for requirement in mustSkip:
            self.assertTrue(
                isPackageToSkip(requirement, PACKAGES_TO_SKIP),
                f"{requirement} must be skipped",
            )

        self.delayDisplay("Denylist whole-name matching test passed")

    def test_DependenciesSurviveDamagedMetadata(self):
        """The resolved dependency list must be complete even if METADATA is damaged.

        An earlier installer stripped pytorch-lightning and torchmetrics from the
        installed SPINEPS metadata. The override table must put them back, otherwise
        the install silently succeeds and SPINEPS fails to import.
        """
        from SlicerSPINEPSLib.InstallLogic import (
            DEPENDENCY_OVERRIDES,
            PACKAGES_TO_PIN,
            requirementName,
        )

        for required in ["pytorch-lightning", "torchmetrics", "tptbox", "acvl-utils"]:
            self.assertIn(
                required,
                DEPENDENCY_OVERRIDES,
                f"{required} must be listed so a damaged or loose metadata entry "
                "cannot drop or under-constrain it",
            )
            self.assertEqual(requirementName(DEPENDENCY_OVERRIDES[required]), required)

        # Anything shared with MOOSE or owned by Slicer must be pinned, so that
        # installing SPINEPS cannot move it underneath the CT segmentation path.
        for pinned in ["torch", "nnunetv2", "SimpleITK", "numpy"]:
            self.assertIn(pinned, PACKAGES_TO_PIN)

        self.delayDisplay("Dependency override and pinning test passed")

    def test_ParameterArgList(self):
        parameter = Parameter(semanticModel="t2w", useCpu=True)
        args = parameter.asArgList(Path("/tmp/sub-spineps_acq-sag_T2w.nii.gz"))

        self.assertIn("sample", args)
        self.assertIn("--model-semantic", args)
        self.assertEqual(args[args.index("--model-semantic") + 1], "t2w")
        self.assertIn("-cpu", args)
        self.assertEqual(parameter.inputFileName(), "sub-spineps_acq-sag_T2w.nii.gz")

        # Off by default: forcing 12 thoracic vertebrae onto a patient who genuinely has
        # a transitional one is as wrong as the skip it prevents.
        self.assertNotIn("--enforce-12-thoracic", args)
        self.assertIn(
            "--enforce-12-thoracic",
            Parameter(semanticModel="t2w", forceTwelveThoracic=True).asArgList(
                Path("/tmp/sub-spineps_acq-sag_T2w.nii.gz")
            ),
        )
        # The CT semantic model has its own instance and labeling models. SPINEPS' own
        # "sample" command does not pair them (only "dataset" resolves "auto"), and it
        # accepts a mismatch silently, so selecting "ct" must switch all three.
        ctArgs = Parameter(semanticModel="ct").asArgList(Path("/tmp/x.nii.gz"))
        self.assertEqual(ctArgs[ctArgs.index("-model_instance") + 1], "ct_instance")
        self.assertEqual(ctArgs[ctArgs.index("-model_labeling") + 1], "ct_labeling")
        self.assertEqual(
            Parameter(semanticModel="ct").inputFileName(), "sub-spineps_acq-iso_ct.nii.gz"
        )
        # An explicit model must still win over the pairing.
        self.assertEqual(
            Parameter(semanticModel="ct", instanceModel="instance").resolvedInstanceModel(),
            "instance",
        )
        self.delayDisplay("Parameter argument list test passed")

    def test_InstallLogicVersionGate(self):
        # Guards against silently targeting a Slicer that is too old.
        import sys

        expected = sys.version_info >= (3, 12)
        self.assertEqual(InstallLogic.isPythonVersionSupported(), expected)
        self.delayDisplay("Install logic Python version gate test passed")
