import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.ticker

###########################
## common plot functions ##
###########################

def single_curve_plot(x,y,save_file,fig_size=(8,6),dpi=80,
    title=None,xlabel=None,ylabel=None,log_x=False,log_y=False,
    xlim=None, ylim=None):

    plt.figure('figure',figsize=fig_size,dpi=dpi)
    plt.plot(x, y, color=[235/255,64/255,52/255])
    if title is not None: plt.title(title)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    if log_x:
        plt.xscale('log')
        locmin = matplotlib.ticker.LogLocator(base=10,subs=(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),numticks=100)
        plt.gca().xaxis.set_minor_locator(locmin)
        plt.gca().xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if log_y: 
        plt.yscale('log')
        locmin = matplotlib.ticker.LogLocator(base=10,subs=(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),numticks=100)
        plt.gca().yaxis.set_minor_locator(locmin)
        plt.gca().yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.grid(which='both',ls='--',lw=1,color=[200/255,200/255,200/255])
    plt.savefig(save_file)
    plt.close('figure')

def multi_curve_plot(curve_dict,save_file,fig_size=(8,6),dpi=80,
    title=None,xlabel=None,ylabel=None,log_x=False,log_y=False,
    xlim=None, ylim=None):
    '''
    multi_curve_plot: display multiple 2d curves in a single plot

    curve_dict: containing data for drawing. Each key-value pair in this dictionary 
    represents a single curve, the properties are: 
    'x': list of x coordinates
    'y': list of y coordinates
    'color': curve color
    'label': show label in legend, can be True or False
    'ls': line style
    'lw': line width
    '''

    plt.figure('figure',figsize=fig_size,dpi=dpi)

    need_legend = False

    curve_names = list(curve_dict.keys())
    for cname in curve_names:
        x = curve_dict[cname]['x']
        y = curve_dict[cname]['y']
        color = curve_dict[cname]['color']
        ls = '-' if 'ls' not in curve_dict[cname] else curve_dict[cname]['ls']
        label = cname if ('label' not in curve_dict[cname]) or (curve_dict[cname]['label']==True) else None
        if label is not None:
            need_legend = True
        lw = 1.5 if 'lw' not in curve_dict[cname] else curve_dict[cname]['lw']
        plt.plot(x, y, color=color,label=label,ls=ls,lw=lw)
    if title is not None: plt.title(title)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    if log_x:
        plt.xscale('log')
        locmin = matplotlib.ticker.LogLocator(base=10,subs=(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),numticks=100)
        plt.gca().xaxis.set_minor_locator(locmin)
        plt.gca().xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if log_y: 
        plt.yscale('log')
        locmin = matplotlib.ticker.LogLocator(base=10,subs=(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),numticks=100)
        plt.gca().yaxis.set_minor_locator(locmin)
        plt.gca().yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.grid(which='both',ls='--',lw=1,color=[200/255,200/255,200/255])
    if need_legend:
        plt.legend()
    plt.savefig(save_file)
    plt.close('figure')
