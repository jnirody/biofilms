###########################################################################
#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import os, skimage, cv2, math, scipy, sys, glob
from skimage import io, filters, morphology, measure, color, exposure, segmentation, feature, img_as_ubyte
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu
from PIL import Image, ImageChops
from optparse import OptionParser
import cv2
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import itertools

#########################################################################
def show_images(before, after):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes.ravel()
    ax[0].imshow(before, cmap='gray')
    ax[0].set_title("Original image")
    ax[1].imshow(after, cmap='gray')
    plt.tight_layout()
    plt.show()
#########################################################################
def crop(image_stack):
    crop_thresh = skimage.filters.threshold_otsu(image_stack[int(len(image_stack)/2)])
    max_int = []
    for column in range(len(image_stack[int(len(image_stack)/2)])):
        temp = sorted(image_stack[int(len(image_stack)/2)][column])
        max_int.append(temp[-15])
    crop_boundaries_col = np.where(max_int > crop_thresh)[0]
    max_int = []
    for row in range(len(np.transpose(image_stack[int(len(image_stack)/2)]))):
        temp = sorted(np.transpose(image_stack[int(len(image_stack)/2)])[row])
        max_int.append(temp[-15])
        #max_int.append(np.max(np.transpose(image_stack[0])[row]))
    crop_boundaries_row = np.where(max_int > crop_thresh)[0]
    if len(image_stack) > 1:
        image_stack = image_stack[:, min(crop_boundaries_col):max(crop_boundaries_col), min(crop_boundaries_row):max(crop_boundaries_row)]
    else:
        image_stack = [image_stack[0][min(crop_boundaries_col):max(crop_boundaries_col), min(crop_boundaries_row):max(crop_boundaries_row)]]
    
    return image_stack
#########################################################################
def run(options):

    files = []
    if options.dir:
        if options.grouped == 1:
            subdirs = [f.path for f in os.scandir(options.dir) if f.is_dir()]
            subdirs = [subdir for subdir in subdirs if len(glob.glob(subdir + '/*.tif'))>0]
            for subdir in subdirs:
                if len(glob.glob(subdir + '/*.tif')) > 0:
                    files.append(glob.glob(subdir + '/*.tif'))
        else:
            subdirs = [options.dir]
            files.append(glob.glob(options.dir + '/*.tif'))
    elif options.file:
        files = [[options.file]]
    elif options.list:
        if options.grouped == 1:
            with open(options.list) as f:
                subdirs = f.readlines()
                print(subdirs)
                subdirs = [subdir[:-2] for subdir in subdirs]
                print(subdirs)
                for subdir in subdirs:
                    files.append(glob.glob(subdir + '/*.tif'))
        else:
            files.append(glob.glob(options.list + '*.tif'))
    else:
        print('No images provided!')
        sys.exit()
    if len(files[0]) == 0:
        print('No image files found!')
        sys.exit()

    composite_zpos = [[] for subdir in range(len(files))]
    normalized_zpos = [[] for subdir in range(len(files))]
    composite_ypos = [[] for subdir in range(len(files))]
    composite_orientation = [[] for subdir in range(len(files))]
    composite_dist = [[] for subdir in range(len(files))]
    composite_area = [[] for subdir in range(len(files))]

    composite_binos = [[] for subdir in range(len(files))]
    composite_binys = [[] for subdir in range(len(files))]
    composite_binas = [[] for subdir in range(len(files))]
    for i in range(len(subdirs)):
        composite_zpos[i] = [[] for file in range(len(files[i]))]
        normalized_zpos[i] = [[] for file in range(len(files[i]))]
        composite_ypos[i] = [[] for file in range(len(files[i]))]
        composite_orientation[i] = [[] for file in range(len(files[i]))]
        composite_area[i] = [[] for file in range(len(files[i]))]

        composite_binos[i] = [[] for file in range(len(files[i]))]
        composite_binys[i] = [[] for file in range(len(files[i]))]
        composite_binas[i] = [[] for file in range(len(files[i]))]
    for i in range(len(subdirs)):
        for f in range(len(files[i])):
            file = files[i][f]
            print(file)
            image_stack = skimage.io.imread(file)
            dims = image_stack.shape
            if len(dims) == 2:
                image_stack = [image_stack]
            else:
                z = np.argmin(dims)
                if z > 0:
                    x = 0
                    y = 1
                    image_stack = np.transpose(image_stack,axes = [z,0,1])

            if options.crop == 1:
                image_stack = crop(image_stack)
            bin_mask = np.zeros(np.array(image_stack).shape)
            bin_mask_cells = np.zeros(np.array(image_stack).shape)
            cells = np.zeros(np.array(image_stack).shape)
            segmented_cells = np.zeros(np.array(image_stack).shape)
            labeled_cells = np.zeros(np.array(image_stack).shape).astype(int) #to fill with binary version for easy labeling

            # clean image slice by slice for easier registration
            for slice in range(len(image_stack)-1):
                print('Segmenting Slice ', slice)
                try:
                    thresh = threshold_multiotsu(image_stack[slice],classes=3)
                    if options.bright == 1:
                        bin_mask[slice] = morphology.remove_small_objects((image_stack[slice] > thresh[-2]), min_size = 10)
                        bin_mask_clumps = morphology.remove_small_objects((image_stack[slice] > thresh[-2]), min_size = 100000000)
                    else:
                        bin_mask[slice] = morphology.remove_small_objects((image_stack[slice] > thresh[-1]), min_size = 10)
                        bin_mask_clumps = morphology.remove_small_objects((image_stack[slice] > thresh[-1]), min_size = 100000000)
                except ValueError:
                    try:
                        thresh = threshold_multiotsu(image_stack[slice],classes=2)
                        bin_mask[slice] = morphology.remove_small_objects((image_stack[slice] > thresh[-1]), min_size = 10)
                        bin_mask_clumps = morphology.remove_small_objects((image_stack[slice] > thresh[-1]), min_size = 100000000)
                    except ValueError:
                        print('Slice has no image!')
                        continue
                bin_mask_cells[slice] = np.subtract(bin_mask[slice], bin_mask_clumps)
                cells[slice] = np.multiply(image_stack[slice],bin_mask_cells[slice])
                distance = ndi.distance_transform_edt(bin_mask_clumps)
                local_maxi = feature.peak_local_max(distance, indices=False,min_distance=10)
                markers = measure.label(local_maxi)
                segmented_cells[slice] = segmentation.watershed(-distance, markers, mask=bin_mask_clumps) + bin_mask_cells[slice]
                labeled_cells[slice] = skimage.measure.label(segmented_cells[slice], background=0)
            colored_cells = skimage.color.label2rgb(labeled_cells, bg_label=0)

            regions = [[] for slice in range(len(labeled_cells))]
            max_label = []
            for slice in range(len(labeled_cells)):
                max_label.append(np.amax(labeled_cells[slice]))
                regions[slice].append(skimage.measure.regionprops_table(np.transpose(labeled_cells[slice]), properties=['label','orientation','centroid','area']))

            zpos = []
            orientation = []
            area = []
            ypos = []
            for slice in range(len(labeled_cells)-1):
                print('Analyzing Slice ', slice)
                for region in range(len(regions[slice])):
                    for comp in range(len(regions[slice][region]['label'])):
                        temp_region_slice = np.array(labeled_cells[slice])
                        temp_region_slice[temp_region_slice==regions[slice][region]['label'][comp]] = max_label[slice]+1
                    zpos.extend(regions[slice][region]['centroid-0'])
                    orientation.extend(regions[slice][region]['orientation'])
                    ypos.extend(regions[slice][region]['centroid-1'])
                    area.extend(regions[slice][region]['area'])
                composite_zpos[i][f].append(zpos)
                normalized_zpos[i][f].append([])
                composite_ypos[i][f].append(ypos)
                composite_orientation[i][f].append(orientation)
                composite_area[i][f].append(area)
            for slice in range(len(composite_zpos[i][f])):
                composite_zpos[i][f][slice] = [x-min(composite_zpos[i][f][slice]) for x in composite_zpos[i][f][slice]]
                normalized_zpos[i][f][slice] = [x/max(composite_zpos[i][f][slice]) for x in composite_zpos[i][f][slice]]
        binsize = 25
        max_zpos = 0
        bin_os = []
        bin_ys = []
        bin_ds = []
        bin_as = []
        for f in range(len(composite_zpos[i])):
            for slice in range(len(composite_zpos[i][f])):
                binsize = max(binsize,int((70*(max(composite_zpos[i][f][slice])-min(composite_zpos[i][f][slice])))/len(composite_zpos[i][f][slice])))
                max_zpos = max(max_zpos,max(composite_zpos[i][f][slice]))
            bin_os.append([[] for bin in range(0,int(max_zpos/binsize)+1)])
            bin_ds.append([[] for bin in range(0,int(max_zpos/binsize)+1)])
            bin_ys.append([[] for bin in range(0,int(max_zpos/binsize)+1)])
            bin_as.append([[] for bin in range(0,int(max_zpos/binsize)+1)])
        for fil in range(len(composite_orientation[i])):
            for slice in range(len(composite_orientation[i][fil])):
                for point in range(len(composite_orientation[i][fil][slice])):
                    bin_os[fil][int(composite_zpos[i][fil][slice][point]/binsize)].extend([composite_orientation[i][fil][slice][point]])
                    bin_ys[fil][int(composite_zpos[i][fil][slice][point]/binsize)].extend([composite_ypos[i][fil][slice][point]])
                    bin_as[fil][int(composite_zpos[i][fil][slice][point]/binsize)].extend([composite_area[i][fil][slice][point]])
                composite_binos[i][fil] = bin_os[fil]
                composite_binys[i][fil] = bin_ys[fil]
                composite_binas[i][fil] = bin_as[fil]

#    fig = plt.figure()
#    ax1 = fig.add_subplot(2,1,1)
#    ax1.imshow(image_stack[int(len(image_stack)/2)],aspect='auto')
#    ax1.set_xticks([])
#    ax1.set_yticks([])
#    ax2 = fig.add_subplot(2,1,2)
#    ax2.imshow(labeled_cells[int(len(labeled_cells)/2)],aspect='auto')
#    ax2.set_xticks([])
#    ax2.set_yticks([])
#    plt.savefig(options.output + '_segmentation.pdf')
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    colors = ['b','r','g','k','y']
    for subdir in range(len(composite_zpos)):
        for file in range(len(composite_zpos[subdir])):
            for slice in range(min(5,len(composite_zpos[subdir][file]))):
                if options.norm == 1:
                    ax1.plot(normalized_zpos[subdir][file][slice],composite_orientation[subdir][file][slice]/max(max(composite_orientation[subdir][file])),'.',color=colors[subdir],alpha=0.2/min(10,len(composite_zpos[subdir][file])))
                else:
                    ax1.plot(composite_zpos[subdir][file][slice],composite_orientation[subdir][file][slice]/max(max(composite_orientation[subdir][file])),'.',color=colors[subdir],alpha=0.2/min(10,len(composite_zpos[subdir][file])))
    for subdir in range(len(composite_zpos)):
        normalized_stds = []
        for file in range(len(composite_zpos[subdir])):
            #normalized_binos.append([j/(len(composite_binos[subdir][file])-1) for j in range(len(composite_binos[subdir][file]))])
           # normalized_binds = [j/(len(composite_binds[subdir][file])-1) for j in range(len(composite_binds[subdir][file]))]
            bin_stds = [np.nanstd(bin) for bin in composite_binos[subdir][file]]
            #bin_stds = [scipy.stats.iqr(bin,nan_policy='omit') for bin in composite_binos[subdir][file]]
            bin_stds = bin_stds[:-1]
            #normalized_stds.append([(bin_std-min(bin_stds))/(max(bin_stds)-min(bin_stds)) for bin_std in bin_stds])
            normalized_stds.append([bin_std for bin_std in bin_stds if math.isnan(bin_std)==0])
        normalized_stds = np.array(list(itertools.zip_longest(*normalized_stds,fillvalue=np.nan))).T
        means = np.nanmean(normalized_stds,axis=0)
        if options.norm == 1:
            normalized_binos = [j/(len(means)-1) for j in range(len(means))]
        else:
            normalized_binos = np.linspace(0,max(composite_zpos[subdir][file][int(len(composite_zpos[subdir][file])/2)]),len(means))
        stdz = np.nanstd(normalized_stds,axis=0)
        ax2.plot(normalized_binos,means,color=colors[subdir],label=subdirs[subdir].split('/')[-1])
        ax2.fill_between(normalized_binos,means-stdz,means+stdz,color=colors[subdir],alpha=0.2)
    if options.norm == 1:
        ax2.set_xlabel('Depth into biofilm')
        ax1.set_xlim([0,1])
        ax2.set_xlim([0,1])
    if options.scale == 1:
        ax1.set_xlim([0,1000])
        ax2.set_xlim([0,1000])
        ax2.set_xlabel('Depth into biofilm (pixel)')
    else:
        ax2.set_xlabel('Depth into biofilm (pixel)')
        ax1.set_xlim([0,image_stack[0].shape[1]])
        ax2.set_xlim([0,image_stack[0].shape[1]])
    ax1.set_ylabel('Orientation')
    ax1.set_xticks([])
    
    ax2.set_ylim([0,1])
    ax1.invert_yaxis()
    ax2.set_ylabel('Std orientation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(options.output + '_orientation.pdf')
##
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex = ax1)
    for subdir in range(len(composite_binys)):
        for file in range(len(composite_binys[subdir])):
            for slice in range(min(5,len(composite_zpos[subdir][file]))):
                if options.norm == 1:
                    ax1.plot(normalized_zpos[subdir][file][slice],composite_ypos[subdir][file][slice]/max(composite_ypos[subdir][file][slice]),'.',color=colors[subdir],alpha=0.2/min(10,len(composite_zpos[subdir][file])))
                else:
                    ax1.plot(composite_zpos[subdir][file][slice],composite_ypos[subdir][file][slice]/max(composite_ypos[subdir][file][slice]),'.',color=colors[subdir],alpha=0.2/min(10,len(composite_zpos[subdir][file])))
    for subdir in range(len(composite_binys)):
        normalized_stds = []
        for file in range(len(composite_binys[subdir])):
            bin_stds = [scipy.stats.entropy(bin) for bin in composite_binys[subdir][file]]
            bin_stds = bin_stds[:-1]
            normalized_stds.append([bin_std for bin_std in bin_stds if math.isnan(bin_std)==0])
        normalized_stds = np.array(list(itertools.zip_longest(*normalized_stds,fillvalue=np.nan))).T
        means = np.nanmean(normalized_stds,axis=0)
        if options.norm == 1:
            normalized_binys = [j/(len(means)-1) for j in range(len(means))]
        else:
            normalized_binys = np.linspace(0,max(composite_zpos[subdir][file][int(len(composite_zpos[subdir][file])/2)]),len(means))
        stdz = np.nanstd(normalized_stds,axis=0)
        ax2.plot(normalized_binys,means,label=subdirs[subdir].split('/')[-1])
        ax2.fill_between(normalized_binys,[i-j for i,j in zip(means,stdz)],[i+j for i,j in zip(means,stdz)],color=colors[subdir],alpha=0.2)
    if options.norm == 1:
        ax1.set_xlim([0,1])
        ax2.set_xlim([0,1])
        ax2.set_xlabel('Depth into biofilm')
    if options.scale == 1:
        ax1.set_xlim([0,1000])
        ax2.set_xlim([0,1000])
        ax2.set_xlabel('Depth into biofilm (pixel)')
    else:
        ax1.set_xlim([0,image_stack[0].shape[1]])
        ax2.set_xlim([0,image_stack[0].shape[1]])
        ax2.set_xlabel('Depth into biofilm (pixel)')
    ax2.set_ylabel('Entropy')

    plt.legend()
    plt.savefig(options.output + '_positions.pdf')
#
#    fig = plt.figure()
#    ax1 = fig.add_subplot(2,1,1)
#    ax2 = fig.add_subplot(2,1,2, sharex = ax1)
#    for subdir in range(len(composite_binas)):
#        for file in range(len(composite_binas[subdir])):
#            #normalized_binys = [j/(len(composite_binys[subdir][file])-2) for j in range(len(composite_binys[subdir][file])-1)]
#            for slice in range(min(5,len(composite_zpos[subdir][file]))):
#                if options.norm == 1:
#                    ax1.plot(normalized_zpos[subdir][file][slice],composite_ypos[subdir][file][slice]/max(composite_ypos[subdir][file][slice]),'.',color=colors[subdir],alpha=0.2/min(10,len(composite_zpos[subdir][file])))
#                else:
#                    ax1.plot(composite_zpos[subdir][file][slice],composite_ypos[subdir][file][slice]/max(composite_ypos[subdir][file][slice]),'.',color=colors[subdir],alpha=0.2/min(10,len(composite_zpos[subdir][file])))
#    for subdir in range(len(composite_binds)):
#        normalized_stds = []
#        for file in range(len(composite_binas[subdir])):
#            bin_stds = [np.nanmean(bin) for bin in composite_binas[subdir][file]]
#            bin_stds = bin_stds[:-1]
#            normalized_stds.append([bin_std for bin_std in bin_stds if math.isnan(bin_std)==0])
#        normalized_stds = np.array(list(itertools.zip_longest(*normalized_stds,fillvalue=np.nan))).T
#        means = np.nanmean(normalized_stds,axis=0)
#        if options.norm == 1:
#            normalized_binas = [j/(len(means)-1) for j in range(len(means))]
#        else:
#            normalized_binas = np.linspace(0,max(composite_zpos[subdir][file][int(len(composite_zpos[subdir][file])/2)]),len(means))
#        stdz = np.nanstd(normalized_stds,axis=0)
#        ax2.plot(normalized_binas,means,label=subdirs[subdir].split('/')[-1])
#        ax2.fill_between(normalized_binas,[i-j for i,j in zip(means,stdz)],[i+j for i,j in zip(means,stdz)],color=colors[subdir],alpha=0.2)
#    if options.norm == 1:
#        ax1.set_xlim([0,1])
#        ax2.set_xlim([0,1])
#        ax2.set_xlabel('Depth into biofilm')
#    else:
#        ax1.set_xlim([0,image_stack[0].shape[1]])
#        ax2.set_xlim([0,image_stack[0].shape[1]])
#        ax2.set_xlabel('Depth into biofilm (pixel)')
#    ax2.set_ylabel('Area')
#
#
#    plt.legend()
#  #  plt.tight_layout()
#    plt.savefig(options.output + '_areas.pdf')
    
##############################################################################
if __name__ == "__main__":
    
    usage="%prog [options]"
    parser = OptionParser(usage)
    
    parser.add_option("-f",action="store",type="string",dest="file",help="File name, with path")
    parser.add_option("-d",action="store",type="string",dest="dir",help="Directory containing all files or subdirectories")
    parser.add_option("-l",action="store",type="string",dest="list",help="List of files or subdirectories")
    parser.add_option("-g",action="store",type="int",dest="grouped",help="Files are grouped into subdirectories")
    parser.add_option("-c",action="store",type="int",dest="crop",default=0,help="Image needs to be cropped")
    parser.add_option("-r",action="store",type="int",dest="rotate",default=0, help="Image needs to be rotated")
    parser.add_option("-b",action="store",type="int",dest="bright", default=0,help="Are there very bright (1) fluorescent regions or not (0)?")
    parser.add_option("-n",action="store",type="int",dest="norm",default=0, help="Normalize depth into biofilm")
    parser.add_option("-s",action="store",type="int",dest="scale",default=0, help="Scale plots for depth into biofilm to be in the range [0,1000]")
    parser.add_option("-o",action="store",type="string",dest="output", default='output',help="Name of output plot")
    (options,args)=parser.parse_args()
    run(options)

