cmake_minimum_required(VERSION 3.13.4)

project(SlicerSPINEPS)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/BerDom-Ing/SlicerSPINEPS")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Bernardo Dominguez")
set(EXTENSION_DESCRIPTION "Whole-spine segmentation of MRI (T2w, T1w, Vibe) and CT using SPINEPS. Installs its own dependencies, including a PyTorch build matched to the machine, from a single button press.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/BerDom-Ing/SlicerSPINEPS/main/SlicerSPINEPS.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/BerDom-Ing/SlicerSPINEPS/main/Screenshots/1.png https://raw.githubusercontent.com/BerDom-Ing/SlicerSPINEPS/main/Screenshots/2.png https://raw.githubusercontent.com/BerDom-Ing/SlicerSPINEPS/main/Screenshots/3.png")
# No extension dependencies on purpose. InstallLogic installs PyTorch itself, via the
# light-the-torch package rather than the PyTorch extension, so that adding this one
# extension and pressing "Install SPINEPS dependencies" is the entire procedure.
# Depending on PyTorch here would force an extension install and a Slicer restart before
# the user could even start. An already-installed PyTorch (from the PyTorch extension,
# SlicerMOOSE, or an earlier run) is reused as is.
set(EXTENSION_DEPENDS "NA")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SlicerSPINEPS)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
