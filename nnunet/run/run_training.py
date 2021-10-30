#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import argparse
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.network_training.nnUNetTrainerV2_DynamicInference import nnUNetTrainerV2_DynamicInference
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration
from nnunet.paths import default_plans_identifier
from nnunet.run.load_pretrained_weights import load_pretrained_weights
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")
    parser.add_argument("task", help="can be task name or task id")
    parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)
    parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", 
                        help="if set then nnUNet will "
                                "export npz files of "
                                "predicted segmentations "
                                "in the validation as well. "
                                "This is needed to run the "
                                "ensembling step so unless "
                                "you are developing nnUNet "
                                "you should enable this")
    parser.add_argument("--softmax", required=False, default=False, action="store_true", 
                        help="if set then nnUNet will save softmax during validation.")
    parser.add_argument("--valbest", required=False, default=False, action="store_true",
                        help="hands off. This is not intended to be used")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                             "will be removed at the end of the training). Useful for development when you are "
                             "only interested in the results and want to save some disk space")
    parser.add_argument("--disable_postprocessing_on_folds", required=False, action='store_true',
                        help="Running postprocessing on each fold only makes sense when developing with nnU-Net and "
                             "closely observing the model performance on specific configurations. You do not need it "
                             "when applying nnU-Net because the postprocessing for this will be determined only once "
                             "all five folds have been trained and nnUNet_find_best_configuration is called. Usually "
                             "running postprocessing on each fold is computationally cheap, but some users have "
                             "reported issues with very large images. If your images are large (>600x600x600 voxels) "
                             "you should consider setting this flag.")
    parser.add_argument('--val_disable_overwrite', action='store_false', default=True,
                        help='Validation does not overwrite existing segmentations')
    parser.add_argument('--disable_next_stage_pred', action='store_true', default=False,
                        help='do not predict next stage')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')

    parser.add_argument('--max_epochs', '-e', type=int, required=False, default=300, help='Max number of epochs for training (default 300).')
    parser.add_argument('--num_train_batches', '-b', type=int, required=False, default=250, help='Number of batches for training in each epoch (default 250).')
    parser.add_argument('--noval', action='store_true', required=False, default=False, help='No validation set.')

    parser.add_argument('--custom_val_cases', type=str, required=False, help='Assign customized validation set cases.', nargs='+')

    parser.add_argument('--save_every_epoch', action='store_true', required=False, default=False, 
                        help='Model will be saved once an epoch is finished.')



    args = parser.parse_args()

    task = args.task
    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    plans_identifier = args.p
    disable_postprocessing_on_folds = args.disable_postprocessing_on_folds
    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data
    deterministic = args.deterministic
    valbest = args.valbest
    fp32 = args.fp32
    run_mixed_precision = not fp32
    val_folder = args.val_folder
    max_epochs = args.max_epochs
    num_train_batches = args.num_train_batches
    noval = args.noval
    custom_val_cases = args.custom_val_cases

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
    trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")

    if network == "3d_cascade_fullres":
        assert issubclass(trainer_class, (nnUNetTrainerCascadeFullRes, nnUNetTrainerV2CascadeFullRes)), \
            "If running 3d_cascade_fullres then your " \
            "trainer class must be derived from " \
            "nnUNetTrainerCascadeFullRes"
    else:
        assert issubclass(trainer_class,
                          nnUNetTrainer), "network_trainer was found but is not derived from nnUNetTrainer"

    if issubclass(trainer_class, nnUNetTrainerV2):
        # nnUNetTrainerV2 class support manual assignment of training epochs and number of training batches.
        trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                                batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                                deterministic=deterministic,
                                fp16=run_mixed_precision, 
                                max_num_epochs=max_epochs, num_batches_per_epoch=num_train_batches,no_validation=noval)
    else:
        # otherwise use ordinary constructor
        trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                                batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                                deterministic=deterministic,
                                fp16=run_mixed_precision)
    
    if args.disable_saving:
        trainer.save_final_checkpoint = False # whether or not to save the final checkpoint
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to
        # self.best_val_eval_criterion_MA
        trainer.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
        # the training chashes
        trainer.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

    if args.save_every_epoch:
        trainer.save_latest_only = False

    if isinstance(trainer, nnUNetTrainerV2):
        if custom_val_cases is not None and len(custom_val_cases)>0:
            trainer.set_custom_validation_set(custom_val_cases)
            if noval:
                print('WARNING: you assigned customized validation set cases but --noval is True. '
                   'Validation set is disabled.')

    trainer.initialize(not validation_only)

    if isinstance(trainer, nnUNetTrainerV2_DynamicInference):
        print('trainer object is an instance of "nnUNetTrainerV2_DynamicInference".')
        print('initializing dynamic inference...')
        trainer.init_dynamic_inference(task)

    if not validation_only:
        if args.continue_training:
            # -c was set, continue a previous training and ignore pretrained weights
            trainer.load_latest_checkpoint()
        elif (not args.continue_training) and (args.pretrained_weights is not None):
            # we start a new training. If pretrained_weights are set, use them
            load_pretrained_weights(trainer.network, args.pretrained_weights)
        else:
            # new training without pretraine weights, do nothing
            pass

        trainer.run_training()
    else:
        if valbest:
            trainer.load_best_checkpoint(train=False)
        else:
            trainer.load_final_checkpoint(train=False)

    trainer.network.eval()

    # predict validation
    if noval == False: # has val :)
        save_softmax = args.npz or args.softmax
        trainer.validate(save_softmax=save_softmax, validation_folder_name=val_folder,
                            run_postprocessing_on_folds=not disable_postprocessing_on_folds,
                            overwrite=args.val_disable_overwrite)

    if network == '3d_lowres' and not args.disable_next_stage_pred:
        print("predicting segmentations for the next stage of the cascade")
        predict_next_stage(trainer, join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))

    if isinstance(trainer, nnUNetTrainerV2_DynamicInference):
        print('making visualizations, please wait...')
        trainer.visualize_training()
        print('OK.')


if __name__ == "__main__":
    main()
