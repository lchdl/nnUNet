from nnunet.utilities.data_io import load_nifti_simple, load_pkl, save_nifti_simple, save_pkl
import os
import numpy as np
import matplotlib
matplotlib.use("agg")
import torch
import torch.backends.cudnn as cudnn
from _warnings import warn
from time import time
from genericpath import isfile
from multiprocessing import Pool
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax, save_segmentation_softmax
from nnunet.utilities.progress_bar import Timer, minibar
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.file_ops import cp, gfn, join_path, ls, mkdir, dir_exist
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from nnunet.utilities.plot import multi_curve_plot

def multilabel_dice_hard(y, y_hat):
    y = y.astype('int')
    y_hat = y_hat.astype('int')
    labels_dice = []
    dice_dict = {}
    for ind in np.unique(y):
        onehot_y = (y==ind).astype('float32')
        onehot_y_hat = (y_hat==ind).astype('float32')
        dsc = 2 * np.sum(onehot_y * onehot_y_hat) / (np.sum(onehot_y) + np.sum(onehot_y_hat) + 0.000001 )
        labels_dice.append(dsc)
        dice_dict[ind]=dsc
    return labels_dice, dice_dict
    
class nnUNetTrainerV2_DynamicInference(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=False, deterministic=True, fp16=False, max_num_epochs=None, num_batches_per_epoch=None,
                 no_validation=False):
        super().__init__(plans_file, fold, output_folder=output_folder, dataset_directory=dataset_directory, 
            batch_dice=batch_dice, stage=stage, unpack_data=unpack_data, deterministic=deterministic, fp16=fp16,
            max_num_epochs=max_num_epochs, num_batches_per_epoch=num_batches_per_epoch, no_validation=no_validation)

        # customized member variables
        self.task_name = None
        self.gt_segmentations = {}

    
    def init_dynamic_inference(self, task_name):
        print('set task name to "%s".' % task_name)
        self.set_task_name(task_name)
        print('setup gt paths...')
        self.setup_gt_paths()
        print('OK.')

    def setup_gt_paths(self):
        nnUNet_preprocessed_folder = os.environ.get('nnUNet_preprocessed')
        assert nnUNet_preprocessed_folder is not None, 'error, "nnUNet_preprocessed" variable is not perperly set.'
        full_task_name = self.get_task_name()
        assert full_task_name is not None, 'cannot find a valid task name!'
        gt_segmentation_folders = join_path(nnUNet_preprocessed_folder, full_task_name, 'gt_segmentations')
        gt_segmentations = [item for item in ls(gt_segmentation_folders, full_path=True) if item[-7:]=='.nii.gz']
        gt_dict = {}
        for gt_segmentation in gt_segmentations:
            case_name = gfn(gt_segmentation, no_extension=True)
            case_path = gt_segmentation
            gt_dict[case_name] = case_path
        self.gt_segmentations = gt_dict

    def set_task_name(self, task_name):
        self.task_name = task_name

    def get_task_name(self):
        return self.task_name

    def run_training(self):
        cp(self.plans_file, join_path(self.output_folder_base, "plans.pkl")) # copy plans file to destination
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we

        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        mkdir(self.output_folder)        
        # self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch+1)
            epoch_start_time = time()

            # train
            self.network.train()
            train_losses_epoch = []
            timer = Timer()
            for current_batch in range(self.num_batches_per_epoch):
                l = self.run_iteration(self.tr_gen, True)
                train_losses_epoch.append(l)
                minibar( msg='Epoch %d/%d'%(self.epoch+1, self.max_num_epochs), a = current_batch+1, b = self.num_batches_per_epoch, time = timer.elapsed())
            print('')
            self.all_tr_losses.append(np.mean(train_losses_epoch))

            #####################
            # dynamic inference #
            #####################
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])
            self.print_to_log_file("inferencing training set, this may take a while...")
            self.inference_training_set(epoch=self.epoch+1, save_softmax=False, 
                do_mirroring=False, use_gaussian=False, use_sliding_window=True)
            self.print_to_log_file("now analyzing training progress...")
            self.analyze_training(epoch=self.epoch+1)
            self.print_to_log_file("visualizing training progress...")
            self.visualize_training(self.epoch+1,False)
            self.print_to_log_file("done.")


            # val
            self.print_to_log_file("start validation...")
            with torch.no_grad():
                self.network.eval()
                val_losses = []
                timer = Timer()
                for current_batch in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
                    minibar( msg='Epoch %d/%d'%(self.epoch+1, self.max_num_epochs), a = current_batch+1, b = self.num_val_batches_per_epoch, time = timer.elapsed())
                print('')
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %.2f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: 
            self.save_checkpoint(join_path(self.output_folder, "model_final_checkpoint.model"))


    def inference_training_set(self, epoch=None, do_mirroring=True, use_sliding_window=True, step_size=0.5,
                 save_softmax=True, use_gaussian=False, overwrite=True, all_in_gpu=False,
                 segmentation_export_kwargs=None):

        current_mode = self.network.training
        self.network.eval()
        assert epoch is not None, 'epoch must be properly set.'
        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_tr is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        output_folder = mkdir(join_path(self.output_folder, 'dynamic_inference', 'epoch_%04d' % epoch))
        if save_softmax:
            output_folder_softmax = mkdir(join_path(self.output_folder, 'dynamic_inference_softmax', 'epoch_%04d' % epoch))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []
        default_num_threads = 4
        export_pool = Pool(default_num_threads)
        results = []

        timer = Timer()
        finished = 0
        for k in self.dataset_tr.keys():
            properties = load_pickle(self.dataset[k]['properties_file'])
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]

            data = np.load(self.dataset[k]['data_file'])['data'] # load preprocessed npz
            
            data[-1][data[-1] == -1] = 0

            softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data[:-1],
                do_mirroring=do_mirroring, mirror_axes=mirror_axes, use_sliding_window=use_sliding_window, step_size=step_size,
                use_gaussian=use_gaussian, all_in_gpu=all_in_gpu, mixed_precision=self.fp16, verbose=False)[1]

            softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

            """There is a problem with python process communication that prevents us from communicating obejcts
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
            communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
            filename or np.ndarray and will handle this automatically"""
            if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be safe
                np.save(join_path(output_folder, fname + ".npy"), softmax_pred)
                softmax_pred = join_path(output_folder, fname + ".npy")

            results.append(
                export_pool.starmap_async(
                    save_segmentation_nifti_from_softmax,
                    (
                        (softmax_pred, join_path(output_folder, fname + ".nii.gz"),
                        properties, interpolation_order, self.regions_class_order,
                        None, None,
                        None, None, force_separate_z,
                        interpolation_order_z, False),
                    )
                )
            )
            if save_softmax:
                results.append(
                    export_pool.starmap_async(
                        save_segmentation_softmax,
                        (
                            (softmax_pred, join_path(output_folder_softmax, fname + ".nii.gz"),
                            properties, interpolation_order, self.regions_class_order,
                            None, force_separate_z,
                            interpolation_order_z),
                        )
                    )
                )

            pred_gt_tuples.append([join_path(output_folder, fname + ".nii.gz"),
                                   join_path(self.gt_niftis_folder, fname + ".nii.gz")])
            finished+=1
            minibar( msg='Epoch %d/%d'%(self.epoch+1, self.max_num_epochs), a = finished, b = len(list(self.dataset_tr.keys())), time = timer.elapsed())
        print('')

        _ = [i.get() for i in results]
        self.print_to_log_file("finished")

        self.network.train(current_mode)

        return pred_gt_tuples

    def analyze_training(self, epoch=None):
        assert epoch is not None, 'epoch must be properly set.'
        epoch_inference_folder = join_path(self.output_folder, 'dynamic_inference', 'epoch_%04d' % epoch)
        assert dir_exist(epoch_inference_folder)
        pairs_need_to_analyze = [ (gfn(item, no_extension=True) , item)  for item in ls(epoch_inference_folder,full_path=True) if item[-7:]=='.nii.gz']


        epoch_summary = {}
        epoch_summary_file = join_path(epoch_inference_folder, 'summary.pkl')

        timer = Timer()

        for i in range(len(pairs_need_to_analyze)):
            case_name, seg_path = pairs_need_to_analyze[i]
            gt_path = self.gt_segmentations.get(case_name)
            assert gt_path is not None, 'cannot find gt segmentation for case "%s".' % case_name
            y = load_nifti_simple(gt_path)
            y_hat = load_nifti_simple(seg_path)
            labels_dice, dice_dict = multilabel_dice_hard(y, y_hat)
            # remove bg label
            labels_dice = labels_dice[1:]
            dice_dict = dice_dict.pop(0)
            #
            case_mean_dice = np.mean(labels_dice)
            epoch_summary[case_name]={
                'mean_dice': case_mean_dice
            }
            minibar( msg='Epoch %d/%d'%(self.epoch+1, self.max_num_epochs), a = i+1, b = len(pairs_need_to_analyze), time = timer.elapsed())
        print('')
        save_pkl(epoch_summary, epoch_summary_file)

    def visualize_training(self,end_epoch:int=-1,verbose:bool=True):
        # load each curve
        di_folder = join_path(self.output_folder, 'dynamic_inference')
        di_epochs_folder = [item for item in ls(di_folder, full_path=True) if dir_exist(item)]
        full_epoch_summary = {}
        if end_epoch<0: 
            end_epoch=self.max_num_epochs
        for epoch_inference_folder in di_epochs_folder:
            epoch_name = gfn(epoch_inference_folder)
            epoch_num = int(epoch_name[-4:])
            if epoch_num>end_epoch: continue
            epoch_summary = load_pkl(join_path(epoch_inference_folder, 'summary.pkl'))
            full_epoch_summary[epoch_name] = epoch_summary # load curve data
        save_pkl(full_epoch_summary, join_path(di_folder, 'summary.pkl')) # save all curve data into disk
        # collect curve data
        all_case_names = list(full_epoch_summary['epoch_0001'].keys())
        all_epoch_names = list(full_epoch_summary.keys())
        all_case_dice_curves = {}
        for case_name in all_case_names:
            all_case_dice_curves[case_name]={
                'x':[], 'y':[],
                'label': False, # dont show legend
                'lw': 0.7 # line width
            }
            curve = []
            for epoch_name in all_epoch_names:
                epoch_num = int(epoch_name[-4:])
                epoch_dice = full_epoch_summary[epoch_name][case_name]['mean_dice']
                curve.append( (epoch_num,epoch_dice) )
            sorted_curve = sorted(curve, key=lambda tup: tup[0])
            all_case_dice_curves[case_name]['x'] = [tup[0] for tup in sorted_curve]
            all_case_dice_curves[case_name]['y'] = [tup[1] for tup in sorted_curve]
        # assign random color for each curve
        curve_sorted = sorted( [ (case_name, all_case_dice_curves[case_name]['y'][-1]) for case_name in list(all_case_dice_curves.keys()) ], key=lambda tup:tup[1]  )
        gen_uniform = lambda start, end: np.random.rand() * (end-start) + start
        for curve_id in range(len(curve_sorted)):
            case_name, _ = curve_sorted[curve_id]
            assigned_color = [0,0,0]
            if   curve_id%6==0: assigned_color=[gen_uniform(0.7,1.0), gen_uniform(0.0,0.3), gen_uniform(0.0,0.3)] # R
            elif curve_id%6==1: assigned_color=[gen_uniform(0.0,0.3), gen_uniform(0.7,1.0), gen_uniform(0.0,0.3)] # G
            elif curve_id%6==2: assigned_color=[gen_uniform(0.0,0.3), gen_uniform(0.0,0.3), gen_uniform(0.7,1.0)] # B
            elif curve_id%6==3: assigned_color=[gen_uniform(0.7,1.0), gen_uniform(0.7,1.0), gen_uniform(0.0,0.3)] # RG
            elif curve_id%6==4: assigned_color=[gen_uniform(0.7,1.0), gen_uniform(0.0,0.3), gen_uniform(0.7,1.0)] # RB
            elif curve_id%6==5: assigned_color=[gen_uniform(0.0,0.3), gen_uniform(0.7,1.0), gen_uniform(0.7,1.0)] # GB
            all_case_dice_curves[case_name]['color'] = assigned_color
        # then plot
        save_file = join_path(self.output_folder, 'dynamic_inference_dice.png')
        multi_curve_plot(all_case_dice_curves,save_file,
            fig_size=(8,6),dpi=300,title='Training Sample Dice Plot',xlabel='Epoch',ylabel='Dice Coefficient', ylim=[-0.05,1.05])
        # print sorted Dice
        if verbose:
            self.print_to_log_file('training set Dice coefficients in last epoch (low to high): ')
            case_dice_tups = []
            for case in list( full_epoch_summary['epoch_%04d' % self.max_num_epochs].keys() ):
                dice = full_epoch_summary['epoch_%04d' % self.max_num_epochs][case]['mean_dice']
                case_dice_tups.append( (case,dice) )
            case_dice_tups = sorted( case_dice_tups, key=lambda tup: tup[1] )
            for case,dice in case_dice_tups:
                self.print_to_log_file('"%s", %.4f' % (  case, dice ))
        
