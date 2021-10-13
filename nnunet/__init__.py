from __future__ import absolute_import
print("\n\nPlease cite the following paper when using nnUNet:\n\nIsensee, F., Jaeger, P.F., Kohl, S.A.A. et al. "
      "\"nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.\" "
      "Nat Methods (2020). https://doi.org/10.1038/s41592-020-01008-z\n\n")
print("If you have questions or suggestions, feel free to open an issue at https://github.com/MIC-DKFZ/nnUNet\n")


print("")
print("+-----------------------------------------+")
print("|      nnU-Net (with customizations)      |")
print("+-----------------------------------------+")
print("edited by Chenghao Liu (2021-09-13)")
print("the following changes were made:")
print("==============")
print("== TRAINING ==")
print("==============")
print("+ added dynamic inference (trace training progress).")
print("* disabled data unpacking to speed up training.")
print("* changed default training epochs from 1000 to 300.")
print("* changed 'pin_memory' to 'False'.")
print("* when fold='all', 20% of the training samples will be randomly selected and used in validation.")
print("* 'nnUNetTrainerV2' now support setting the number of training epochs and batches manually (use '-b' and '-e').")
print("* now user can turn off validation set during training (use '--noval').")
print("* use '--custom_val_cases XXX YYY ...' and set fold to 'all' to manually assign validation set cases.")
print("* use '--save_every_epoch' to save models from all epochs.")
print("* add '--softmax' to save softmax values of validation set samples.")
print("===============")
print("== INFERENCE ==")
print("===============")
print("* add '--selected_cases XXX YYY ...' when inference will only predict the selected test subjects (XXX YYY ...) .")
print("* add '--softmax' to save softmax values of test set samples.")
print("===========")
print("== OTHER ==")
print("===========")
print("* removed unused codes (deleted unused trainers).")
print("* some bug fixes...")
print("")

from . import *

