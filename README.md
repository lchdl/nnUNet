Customized nnU-Net by Chenghao Liu. The following changes were made:

## TRAINING
* disabled data unpacking to speed up training (reduced IO).
* changed default training epochs from 1000 to 300.
* changed 'pin_memory' to 'False'.
* when fold='all', 20% of the training samples will be randomly selected and used in validation.
* 'nnUNetTrainerV2' now support manually setting the number of training epochs and batches (use '-b' and '-e').
* add '--noval' to tell the program not use validation set during training.
* use '--custom_val_cases XXX YYY ...' and set fold to 'all' to manually assign validation set cases.
* use '--save_every_epoch' to save models from all epochs.
* add '--softmax' to save softmax values of validation set samples.

## INFERENCE
* add '--selected_cases XXX YYY ...' when inference will only predict the selected test subjects (XXX YYY ...) .
* add '--softmax' to save softmax values of test set samples.

## OTHERS
* removed unused codes.
* minor bug fixes...
