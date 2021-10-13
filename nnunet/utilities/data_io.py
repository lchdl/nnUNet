# defines some utility functions for reading/writing 
# nifti(*.nii)/pickle(*.pkl)/csv(*.csv) files

import csv
import pickle
import nibabel as nib
import numpy as np
from nibabel import processing as nibproc

import xlsxwriter
from xlsxwriter.format import Format

def load_csv_simple(file_path, key_names):
    parsed_dataset = {}
    
    with open(file_path, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',',quotechar='"')
        table_head = None
        for row in csv_reader:
            if table_head is None: 
                table_head = row
            else:
                for key in key_names:
                    if key not in table_head:
                        raise Exception('Cannot find key name "%s" in CSV file "%s". Expected key names can be %s.' % \
                            (key,file_path, table_head))
                    else:
                        column_index = table_head.index(key)
                        if key not in parsed_dataset:
                            parsed_dataset[key] = [] # create list
                        parsed_dataset[key].append(row[column_index])

    return parsed_dataset

def write_csv_simple(file_path, csv_dict):
    keys = list(csv_dict.keys())
    lines=0
    for key in keys:
        if len(csv_dict[key]) > lines:
            lines = len(csv_dict[key])
    with open(file_path,'w') as f:
        table_head = ''
        for key in keys: table_head += key+','
        f.write(table_head+'\n')
        for i in range(lines):
            for key in keys:
                if i >= len(csv_dict[key]):
                    f.write(',')
                else:
                    f.write('%s,' % csv_dict[key][i])
            f.write('\n')

def save_pkl(obj, pkl_path):
    with open(pkl_path,'wb') as f:
        pickle.dump(obj,f)

def load_pkl(pkl_path):
    content = None
    with open(pkl_path,'rb') as f:
        content = pickle.load(f)
    return content


#
# NIFTI file operations
#

def load_nifti_simple(path, return_type='float32'):
    if return_type is None:
        return nib.load(path).get_fdata()
    else:
        return nib.load(path).get_fdata().astype(return_type)

def save_nifti_simple(data,path): # save NIFTI using the default header
	nib.save(nib.Nifti1Image(data.astype('float32'),affine=np.eye(4)),path)

def get_nifti_header(nii_path):
    return nib.load(nii_path).header.copy()

def get_nifti_data(nii_path, return_type='float32'):
    return nib.load(nii_path).get_fdata().astype(return_type)

def save_nifti_with_header(data, header, path):
    nib.save(nib.nifti1.Nifti1Image(data.astype('float32'), None, header=header),path)

# synchronize NIFTI file header
def sync_nifti_header(source_path, target_path, output_path):
    target_header = nib.load(target_path).header.copy()
    source_data = nib.load(source_path).get_fdata()
    save_nifti_with_header(source_data, target_header, output_path)

# get physical resolution
def get_nifti_pixdim(nii_path):
    nii = nib.load(nii_path)
    nii_dim = list(nii.header['dim'])
    nii_pixdim = list(nii.header['pixdim'])
    actual_dim = list(nii.get_fdata().shape)
    physical_resolution = []
    for v in actual_dim:
        physical_resolution.append(nii_pixdim[nii_dim.index(v)])
    return physical_resolution

def resample_nifti(source_path, new_resolution, output_path):
    input_img = nib.load(source_path)
    resampled_img = nibproc.resample_to_output(input_img, new_resolution, order=0)
    nib.save(resampled_img, output_path)

#
# Excel file operations
#

# a simple excel writer class 
class ExcelWriter(object):
    def __init__(self, file_path, worksheet_names='default'):

        if file_path[-5:]!='.xlsx':
            raise RuntimeError('Invalid file name. File path must ends with ".xlsx".')
        self.file_path = file_path

        if isinstance(worksheet_names, str):
            self.worksheet_names = [worksheet_names]
        elif isinstance(worksheet_names, list):
            self.worksheet_names = worksheet_names
        else:
            raise RuntimeError('Only str or list type are accepted. Got type "%s".' % type(worksheet_names).__name__)

        # create workbook and worksheet(s)
        self.workbook = xlsxwriter.Workbook(self.file_path)
        self.worksheets = {}
        for worksheet_name in self.worksheet_names:
            self.worksheets[worksheet_name] = self.workbook.add_worksheet(worksheet_name)

    def _is_closed(self):
        return self.workbook is None
    
    def _check_closed(self):
        if self._is_closed():
            raise RuntimeError('Excel file is already closed and saved, which cannot be written anymore!')

    def _close(self):
        self.workbook.close()
        self.workbook = None

    def new_format(self,bold=False,italic=False,underline=False,font_color='#000000',bg_color='#FFFFFF'):

        self._check_closed()
        cell_format = self.workbook.add_format()
        cell_format.set_bold(bold)
        cell_format.set_italic(italic)
        cell_format.set_underline(underline)
        cell_format.set_font_color(font_color)
        cell_format.set_bg_color(bg_color)

        return cell_format

    def set_column_width(self, pos, width, worksheet_name='default'):
        self._check_closed()
        if worksheet_name not in self.worksheet_names:
            raise RuntimeError('Cannot find worksheet with name "%s".' % worksheet_name)
        if isinstance(pos, int):
            self.worksheets[worksheet_name].set_column(pos,pos, width)
        elif isinstance(pos,str):
            self.worksheets[worksheet_name].set_column(pos, width)
        elif isinstance(pos,tuple) or isinstance(pos,list):
            assert len(pos)==2, 'Invalid position setting.'
            start,end=pos
            self.worksheets[worksheet_name].set_column(start,end,width)

    def write(self, cell_name_or_pos, content, worksheet_name='default', format=None):
        self._check_closed()        
        if format is not None:
            assert isinstance(format, Format), 'Invalid cell format.'
        if worksheet_name not in self.worksheet_names:
            raise RuntimeError('Cannot find worksheet with name "%s".' % worksheet_name)
        if isinstance(cell_name_or_pos, tuple) or isinstance(cell_name_or_pos, list):
            assert len(cell_name_or_pos) == 2, 'Invalid cell position.'
            row,col = cell_name_or_pos
            if format:
                self.worksheets[worksheet_name].write(row,col,content,format)
            else:
                self.worksheets[worksheet_name].write(row,col,content)
        elif isinstance(cell_name_or_pos, str):
            if format:
                self.worksheets[worksheet_name].write(cell_name_or_pos,content,format)
            else:
                self.worksheets[worksheet_name].write(cell_name_or_pos,content)
        else:
            raise RuntimeError('Invalid cell name or position. Accepted types are: tuple, list or str but got "%s".' % \
                type(cell_name_or_pos).__name__)

    def save_and_close(self):
        self._close()
