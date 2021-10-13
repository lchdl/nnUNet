from genericpath import isfile
import os

def run_shell(command,log_command=True,force_continue=False):
    if log_command:
        print(command)
    retval = os.system(command)
    if retval!=0:
        print('Unexpected return value "%d" from command:' % retval)
        print(command)
        if force_continue:
            return retval
        else:
            print('Main process will exit now.')
            exit() 
    else:
        return retval

def ls_tree(folder,depth=2,stat_size=False,closed=False):
    '''
    similar to "ls" command in unix-based systems but can print file hierarchies
    param closed: open/close directory
    '''

    def __get_size_in_bytes(start_path):
        '''
        from https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python
        '''
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size
    
    def __size_format(size_in_bytes):
        if size_in_bytes < 1024:
            return '%dB' % size_in_bytes
        elif size_in_bytes >=1024 and size_in_bytes <=1024*1024-1:
            return '%dKB' % (size_in_bytes//1024)
        elif size_in_bytes >=1024*1024 and size_in_bytes <=1024*1024*1024-1:
            return '%.2fMB' % (size_in_bytes/(1024*1024))
        else:
            return '%.2fGB' % (size_in_bytes/(1024*1024*1024))

    def __abbr_filename(filename, length):
        if len(filename)<=length: return filename
        else:
            return '...'+filename[len(filename)-length+3:]


    def __max_item_len(l):
        maxl = 0
        for item in l:
            if len(item)> maxl:
                maxl = len(item)
        return maxl

    def __rreplace_once(s,subs,replace):
        return replace.join(s.rsplit(subs, 1))

    def __ls_ext_from(folder,indent,cur_depth,max_depth,max_len=24):
        # indent: spaces added in each indent level
        # folder: folder path in current search
        # max_len: maximum length of file name

        prespaces =  ('|' + ' '*(indent-1)) * cur_depth
        terminal_cols = os.get_terminal_size()[0]
        
        items_all = os.listdir(folder)
        items_f = sorted([item for item in items_all if os.path.isfile(os.path.join(folder, item)) == True])
        items_d = sorted([item for item in items_all if os.path.isfile(os.path.join(folder, item)) == False])

        abbr_items_f_repr = [ __abbr_filename(item,max_len-1)+' ' for item in items_f ]
        abbr_items_d_repr = [ __abbr_filename(item,max_len-1)+'/' for item in items_d ]

        if cur_depth == max_depth:
            max_repr_len = __max_item_len(abbr_items_f_repr + abbr_items_d_repr)

            disp_cols = (terminal_cols - indent*cur_depth) // (max_repr_len+2)
            if disp_cols < 1: disp_cols = 1
            repr_string = '' + prespaces
            abbr_items_repr = abbr_items_f_repr + abbr_items_d_repr
            for tid in range(len(abbr_items_repr)):
                repr_string += abbr_items_repr[tid]
                if ( (tid+1) % disp_cols == 0 ) and (tid != len(abbr_items_repr)-1): # newline
                    repr_string += '\n' + prespaces
                else:
                    pad = ' ' * (max_repr_len-len(abbr_items_repr[tid]))
                    repr_string += pad + '  '
            if len(abbr_items_repr)>0 and closed == False:
                print(repr_string)
        else:
            max_repr_len = __max_item_len(abbr_items_f_repr) if len(abbr_items_f_repr)>0 else max_len
            # print files
            disp_cols = (terminal_cols - indent*cur_depth) // (max_repr_len+2)
            if disp_cols < 1: disp_cols = 1
            repr_string = '' + prespaces
            abbr_items_repr = abbr_items_f_repr
            for tid in range(len(abbr_items_repr)):
                repr_string += abbr_items_repr[tid]
                if ( (tid+1) % disp_cols == 0 ) and (tid != len(abbr_items_repr)-1): # newline
                    repr_string += '\n' + prespaces
                else:
                    pad = ' ' * (max_repr_len-len(abbr_items_repr[tid]))
                    repr_string += pad + '  '
            if len(abbr_items_repr)>0:
                print(repr_string)

            # recursive print dirs
            for it, it_repr in zip(items_d, abbr_items_d_repr):
                if it_repr[-1] == '/': # is directory
                    fc = len(os.listdir(os.path.join(folder, it))) # number of files in directory
                    subs = '|' + ' '*(indent-1)
                    replace = '+' + '-'*(indent-1)
                    # replace "|  " to "+--"
                    prespaces2 = __rreplace_once(prespaces,subs,replace)
                    if stat_size:
                        size_str = __size_format(__get_size_in_bytes(os.path.join(folder, it)))
                        print( prespaces2 + it + ' +%d item(s)' % fc + ', ' + size_str)
                    else:
                        print( prespaces2 + it + ' +%d item(s)' % fc)
                    __ls_ext_from(os.path.join(folder, it),indent, cur_depth+1,max_depth,max_len=max_len)
                else:
                    print(prespaces + it_repr)

    # check if folder exists
    folder = os.path.abspath(folder)
    if os.path.exists(folder) and os.path.isdir(folder) == False:
        raise FileNotFoundError('folder not exist: "%s".' % os.path.abspath(folder))
    folder = folder + os.path.sep

    foldc = os.listdir(folder)
    if stat_size:
        size_str = __size_format(__get_size_in_bytes(folder))
        print('file(s) in "%s"  +%d item(s), %s' % (folder, len(foldc), size_str))
    else:
        print('file(s) in "%s"  +%d item(s)' % (folder, len(foldc)))
    if len(foldc) == 0: # empty root folder
        print('(folder is empty)')
        return
    else:
        __ls_ext_from(folder, 3, 1, max_depth=depth)
