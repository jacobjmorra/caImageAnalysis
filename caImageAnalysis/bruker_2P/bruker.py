from bs4 import BeautifulSoup
from datetime import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import tifffile

from .bruker_utils import round_microseconds
from .markpoints import MarkPoints
from .voltage_output import VoltageOutput
from caImageAnalysis import Fish
from caImageAnalysis.mesm import get_plane_number, load_mesmerize, load_rois, uuid_to_plane
from caImageAnalysis.utils import calculate_fps, load_pickle


class BrukerFish(Fish):
    def __init__(self, folder_path, region='', remove_pulses=None, gavage=False, anatomy=''):
        self.region = region
        self.remove_pulses = remove_pulses
        self.gavage = gavage
        self.anatomy = anatomy
        super().__init__(folder_path)
        
        self.bruker = True

        self.process_bruker_filestructure()
        self.volumetric = self.check_volumetric()

        if self.volumetric:
            try:
                self.fps = calculate_fps(self.data_paths['volumes']['0']['frametimes'])
            except KeyError:
                pass
        else:
            try:
                self.fps = calculate_fps(self.data_paths['frametimes'])
            except:
                pass

        # TODO: self.zoom = self.get_zoom()

    def process_bruker_filestructure(self):
        '''Appends Bruker specific file paths to the data_paths'''
        with os.scandir(self.exp_path) as entries:
            for entry in entries:
                if os.path.isdir(entry.path): 
                    if entry.name.startswith('img_stack_'):
                        self.volumetric = True
                    elif entry.name == 'mesmerize-batch':
                        self.data_paths['mesmerize'] = Path(entry.path)
                    elif entry.name == 'stytra':
                        self.data_paths['stytra'] = Path(entry.path)
                    elif len(self.region) > 0 and entry.name.startswith(self.region):
                        self.data_paths['raw'] = Path(entry.path)
                    elif (len(self.region) == 0 and len(self.anatomy) == 0 and 
                            not entry.name.startswith('SingleImage-')) or (len(self.region) == 0 and len(self.anatomy) > 0 and 
                            not entry.name.startswith('SingleImage-') and not entry.name.startswith(self.anatomy)):
                        self.data_paths['raw'] = Path(entry.path)
                        self.region = entry.name
                    elif len(self.anatomy) > 0 and entry.name.startswith(self.anatomy):
                        with os.scandir(entry.path) as subentries:
                            for subentry in subentries:
                                if len(self.region) == 0 and subentry.name.endswith('.tif'):
                                    self.data_paths['anatomy'] = Path(subentry.path)
                                elif len(self.region) > 0 and subentry.name.endswith('.tif') and subentry.name.startswith(self.region):
                                    self.data_paths['anatomy'] = Path(subentry.path)

                elif len(self.region) > 0 and entry.name.startswith(self.region):  # if there is a specific region given
                    if entry.name.endswith('ch2.tif') and not entry.name.startswith('.'):
                        self.data_paths['raw_image'] = Path(entry.path)
                    elif entry.name.endswith('frametimes.txt') and not entry.name.startswith('.'):
                        self.data_paths['frametimes'] = Path(entry.path)
                        self.raw_text_frametimes_to_df()
                
                else:
                    if entry.name.endswith('.xml'):
                        self.data_paths['combined_log'] = Path(entry.path)
                    elif entry.name.endswith('ch2.tif') and not entry.name.startswith('.') and 'raw_image' not in self.data_paths.keys():
                        self.data_paths['raw_image'] = Path(entry.path)
                    elif entry.name == 'raw_rotated.tif':
                        self.data_paths['rotated'] = Path(entry.path)
                    elif entry.name.endswith('frametimes.h5') and not entry.name.startswith('.'):
                        self.data_paths['frametimes'] = Path(entry.path)
                    elif entry.name.endswith('frametimes.txt') and not entry.name.startswith('.') and 'frametimes' not in self.data_paths.keys():
                        self.data_paths['frametimes'] = Path(entry.path)
                        self.raw_text_frametimes_to_df()
                    elif entry.name == 'opts.pkl':
                        self.data_paths['opts'] = Path(entry.path)
                    elif entry.name == 'clusters.pkl':
                        self.data_paths['clusters'] = Path(entry.path)
                        self.clusters = load_pickle(Path(entry.path))
                    elif entry.name == 'temporal.h5':
                        self.data_paths['temporal'] = Path(entry.path)
                        self.temporal_df = pd.read_hdf(self.data_paths['temporal'])
                    elif entry.name == 'unrolled_temporal.h5':
                        self.data_paths['unrolled_temporal'] = Path(entry.path)
                        self.unrolled_df = pd.read_hdf(self.data_paths['unrolled_temporal'])
                    elif entry.name == 'vol_temporal.pkl':
                        self.data_paths['vol_temporal'] = Path(entry.path)
                        self.vol_temporal = load_pickle(self.data_paths['vol_temporal'])
                    elif entry.name == 'anatomy.tif':
                        self.data_paths['anatomy'] = Path(entry.path)
                    elif 'C_frames' in entry.name and not entry.name.startswith('.'):
                        self.data_paths['C_frames'] = Path(entry.path)
                    elif entry.name == 'analysis_results.hdf5':
                        self.data_paths['analysis_results'] = Path(entry.path)
                    
        if 'raw' in self.data_paths.keys():
            if 'anatomy' not in self.data_paths.keys():
                self.data_paths['anatomy'] = self.get_anatomy()

            with os.scandir(self.data_paths['raw']) as entries:
                for entry in entries:
                    if os.path.isdir(entry.path) and entry.name == 'References':
                        self.data_paths['references'] = Path(entry.path)
                    elif entry.name == self.data_paths['raw'].name + '.xml':
                        self.data_paths['log'] = Path(entry.path)
                    elif (entry.name.endswith('.xml')) and (not entry.name.startswith('.')):
                        if 'MarkPoints' in entry.name:
                            if 'markpoints' in self.data_paths.keys():
                                self.data_paths['markpoints'].append(Path(entry.path))
                            else:
                                self.data_paths['markpoints'] = [Path(entry.path)]
                        elif 'VoltageOutput' in entry.name:
                            self.data_paths['voltage_output'] = Path(entry.path)

        if self.volumetric:
            self.data_paths['volumes'] = dict()
            with os.scandir(self.exp_path) as entries:
                for entry in entries:
                    if entry.name.startswith('img_stack_'):
                        volume_ind = entry.name[entry.name.rfind('_')+1:]
                        self.data_paths['volumes'][volume_ind] = dict()

                        with os.scandir(entry.path) as subentries:
                            for sub in subentries:
                                if sub.name == 'image.tif':
                                    self.data_paths['volumes'][volume_ind]['image'] = Path(sub.path)
                                elif sub.name == 'frametimes.h5':
                                    self.data_paths['volumes'][volume_ind]['frametimes'] = Path(sub.path)

        if 'mesmerize' in self.data_paths.keys():
            self.process_mesmerize_filestructure()

        try:
            if self.gavage:
                if self.volumetric:
                    self.align_pulses_to_frametimes_from_volume()
                else:
                    # Typical pulse range to compare
                    pulses = [1956, 2739, 3521, 4304, 5086]
                    if self.remove_pulses is not None:
                        vals = [pulses[rp-1] for rp in self.remove_pulses]
                        for val in vals:
                            pulses.remove(val)
                    self.align_pulses_to_frametimes(pulses)
            
            elif 'voltage_output' in self.data_paths.keys():
                self.voltage_output = VoltageOutput(self.data_paths['voltage_output'], self.data_paths['log'])
                self.frametimes_df = self.voltage_output.align_pulses_to_frametimes(self.frametimes_df)

            elif 'markpoints' in self.data_paths.keys():
                self.markpoints = dict()
                self.data_paths['markpoints'].sort()

                for mp_path in self.data_paths['markpoints']:
                    mp_path = str(mp_path)
                    cycle = int(mp_path[mp_path.find('Cycle')+5:mp_path.rfind('_')])
                    mp = MarkPoints(mp_path, self.data_paths['log'], cycle=cycle)
                    self.markpoints[cycle] = mp
                    self.frametimes_df = self.markpoints[cycle].align_pulses_to_frametimes(self.frametimes_df)

        except AttributeError:
            # if this is the first time initializing, frametimes.txt might not have been created yet
            pass

    def get_anatomy(self):
        '''Finds the anatomy stack in the raw data folder'''
        imgs = [path for path in os.listdir(self.data_paths['raw']) if path.endswith('.ome.tif')]

        first_img = tifffile.imread(self.data_paths['raw'].joinpath(imgs[0]))
        last_img = tifffile.imread(self.data_paths['raw'].joinpath(imgs[-1]))

        if first_img.shape[1] != last_img.shape[1]:
            return self.data_paths['raw'].joinpath(imgs[-1])
        else:
            return None
        
    def check_volumetric(self):
        '''Checks if the experiment is volumetric'''
        volumetric = False
        
        if 'combined_log' in self.data_paths:
            with open(self.data_paths['combined_log'], 'r') as file:
                log = file.read()
        else:
            with open(self.data_paths['log'], 'r') as file:
                log = file.read()

        Bs_data = BeautifulSoup(log)

        first_sequence = Bs_data.find_all('sequence')[0]
        if 'ZSeries' in first_sequence['type']:
            # if it's a Z-Series, automatically assume volumetric
            volumetric = True
            self.volumetric_type = 'ZSeries'

        elif first_sequence['type'] == 'TSeries Timed Element':
            # this is for "fake volumetric" image sequences
            sequences = Bs_data.find_all('sequence')
            planes = list()
            
            for plane, seq in enumerate(sequences):
                if seq['type'] == "TSeries Timed Element":
                    planes.append(plane)
            
            if len(np.unique(planes)) > 1:
                volumetric = True
                self.volumetric_type = f'fake_volumetric_{len(np.unique(planes))}'

        return volumetric
    
    def create_frametimes_txt(self):
        '''Creates a frametimes.txt file from the log xml file'''
        with open(self.data_paths['log'], 'r') as file:
            log = file.read()

        Bs_data = BeautifulSoup(log)
        frames = Bs_data.find_all('frame')
        first_line = Bs_data.find_all('pvscan')
        str_time = first_line[0]['date'].split(' ')[1]  # start time of the experiment

        dt_time = dt.strptime(str_time, '%H:%M:%S')

        if first_line[0]['date'].split(' ')[2] == 'PM':
            # datetime doesn't handle military time well
            military_adjustment = timedelta(hours=12)
            dt_time = dt_time + military_adjustment

        if len(self.region) > 0:
            frametimes_path = os.path.join(self.exp_path, f'{self.region}_frametimes.txt')
        else:
            frametimes_path = os.path.join(self.exp_path, f'frametimes.txt')
        self.data_paths['frametimes'] = Path(frametimes_path)

        with open(frametimes_path, 'w') as file:
            for i, frame in enumerate(frames):
                if 'anatomy' in frame['parameterset']:  # if the anatomy stack starts
                    break
                else:
                    str_abstime = round_microseconds(frame['relativetime'])
                    dt_abstime = timedelta(seconds=int(str_abstime[:str_abstime.find('.')]),
                                        microseconds=int(str_abstime[str_abstime.find('.')+1:]))
                    str_final_time = dt.strftime(dt_time + dt_abstime, '%H:%M:%S.%f')
                    file.write(str_final_time + '\n')

        self.raw_text_frametimes_to_df()

        if self.gavage:
            # Typical pulse range to compare
            pulses = [1956, 2739, 3521, 4304, 5086]
            if self.remove_pulses is not None:
                vals = [pulses[rp-1] for rp in self.remove_pulses]
                for val in vals:
                    pulses.remove(val)
            self.align_pulses_to_frametimes(pulses)
        elif 'voltage_output' in self.data_paths.keys():
            self.frametimes_df = self.voltage_output.align_pulses_to_frametimes(self.frametimes_df)
        elif 'markpoints' in self.data_paths.keys():
            for cycle in self.markpoints:
                self.frametimes_df = self.markpoints[cycle].align_pulses_to_frametimes(self.frametimes_df)
        else:
            print('no voltage output or markpoints detected')
    
    def combine_channel_images(self, channel):
        '''Combines a channel's images'''
        channels = ['Ch1', 'Ch2']
        if channel not in channels:
            raise ValueError(f'channel needs to be one of {channels}')

        ch_image_paths = []
        for entry in sorted(os.scandir(self.data_paths['raw']), key=lambda e: e.name):
            if entry.name.endswith('.ome.tif') and channel in entry.name:
                ch_image_paths.append(Path(entry.path))

        ch_images = [np.array(tifffile.imread(img_path)) for img_path in ch_image_paths]
        
        if len(self.anatomy) == 0:
            # find out if there is an anatomy stack
            n_planes = ch_images[0].shape[0]
            anatomy_index = [i for i, img in enumerate(ch_images) if img.shape[0] != n_planes]
            if len(anatomy_index) != 0:
                del ch_images[anatomy_index[0]]
                self.data_paths['anatomy'] = ch_image_paths[anatomy_index[0]]

        raw_img = np.concatenate(ch_images)
        
        if len(self.region) > 0:
            ch_image_path = Path(os.path.join(self.exp_path, f'{self.region}_{channel.lower()}.tif'))
        else:
            ch_image_path = Path(os.path.join(self.exp_path, f'{channel.lower()}.tif'))
        tifffile.imsave(ch_image_path, raw_img, bigtiff=True)

        self.data_paths['raw_image'] = ch_image_path

        plt.imshow(raw_img[0])

    def split_bruker_volumes(self, channel, overwrite=True):
        '''Splits volumes to individual planes'''
        channels = ['Ch1', 'Ch2']
        if channel not in channels:
            raise ValueError(f'channel needs to be one of {channels}')

        ch_image_paths = []

        for entry in sorted(os.scandir(self.data_paths['raw']), key=lambda e: e.name):
            if entry.name.endswith('.ome.tif') and channel in entry.name:
                ch_image_paths.append(Path(entry.path))

        if 'rotated' in self.data_paths.keys():
            img = tifffile.memmap(self.data_paths['rotated'])
        else:
            img = tifffile.memmap(self.data_paths['raw_image'])

        if self.volumetric_type == 'ZSeries':
            n_planes = tifffile.imread(ch_image_paths[0]).shape[0]
        elif self.volumetric_type.startswith('fake_volumetric'):
            n_planes = int(self.volumetric_type[self.volumetric_type.rfind('_')+1:])
            len_plane = int(img.shape[0]/n_planes)  # number of frames in each plane
        
        for plane in range(n_planes):
            plane_folder_path = os.path.join(self.exp_path, f'img_stack_{plane}')
            if not os.path.exists(plane_folder_path):
                os.mkdir(plane_folder_path)

            if self.volumetric_type == 'ZSeries':
                plane_img = img[plane::n_planes]
                plane_frametimes = self.frametimes_df[plane::n_planes].copy()
            elif self.volumetric_type.startswith('fake_volumetric'):
                plane_img = img[plane*len_plane:(plane+1)*len_plane]
                plane_frametimes = self.frametimes_df[plane*len_plane:(plane+1)*len_plane].copy()

            if overwrite:
                tifffile.imsave(os.path.join(plane_folder_path, 'image.tif'), plane_img, bigtiff=True)

                plane_frametimes = plane_frametimes.reset_index(drop=True)
                plane_frametimes.to_hdf(os.path.join(plane_folder_path, 'frametimes.h5'), 'frametimes')
            else:
                try:
                    prev_plane_img = tifffile.memmap(os.path.join(plane_folder_path, 'image.tif'))
                    new_img = np.concatenate(prev_plane_img, plane_img)
                    tifffile.imsave(os.path.join(plane_folder_path, 'image.tif'), new_img, bigtiff=True)

                    prev_plane_fts = pd.read_hdf(os.path.join(plane_folder_path, 'frametimes.h5'))
                    new_fts = pd.concat(prev_plane_fts, plane_frametimes)
                    new_fts = new_fts.reset_index(drop=True)
                    new_fts.to_hdf(os.path.join(plane_folder_path, 'frametimes.h5'), 'frametimes')
                except FileNotFoundError:
                    # if an image.tif doesn't exist yet
                    tifffile.imsave(os.path.join(plane_folder_path, 'image.tif'), plane_img, bigtiff=True)

                    plane_frametimes = plane_frametimes.reset_index(drop=True)
                    plane_frametimes.to_hdf(os.path.join(plane_folder_path, 'frametimes.h5'), 'frametimes')

        self.process_bruker_filestructure()

    def get_pulse_frames(self):
        '''Gets frame indices for each pulse (for bruker recordings)
        Picks the most common frame across all planes'''
        if not hasattr(self, 'temporal_df'):
            raise AttributeError('Requires a temporal_df: Run temporal.py/save_temporal')
        
        pulse_frames = list()

        n_pulses = len(self.temporal_df.pulse_frames[0])
        for i in range(n_pulses):
            pulse_frames.append(np.argmax(np.bincount([pulses[i] for pulses in self.temporal_df.pulse_frames])))

        return pulse_frames
    
    def get_zoom(self):
        '''Extracts the zoom info from log'''
        with open(self.data_paths['log'], 'r') as file:
            log = file.read()

        Bs_data = BeautifulSoup(log)
        values = Bs_data.find_all('pvstatevalue')
        
        for val in values:
            if val['key'] == 'opticalZoom':
                print(val['value'])
        return
    
    def align_pulses_to_frametimes(self, pulses):
        '''Aligns manual entry of pulse frames to the frametimes dataframe'''
        self.frametimes_df['pulse'] = 0
        curr_pulse = 0
        
        for i, _ in self.frametimes_df.iterrows():
            try:
                if i >= pulses[curr_pulse]:
                    curr_pulse += 1
                    self.frametimes_df.loc[i, 'pulse'] = curr_pulse
                else:
                    self.frametimes_df.loc[i, 'pulse'] = curr_pulse
            except IndexError:
                # exception for the last pulse
                self.frametimes_df.loc[i, 'pulse'] = curr_pulse

    def align_pulses_to_frametimes_from_volume(self):
        '''Aligns frametimes_df from the split frametimes h5 files'''
        pulses = dict()
        n_planes = len(self.data_paths['volumes'])

        for plane in self.data_paths['volumes']:
            df = pd.read_hdf(self.data_paths['volumes'][plane]['frametimes'])

            for pulse in df.pulse.unique():
                if pulse not in pulses:
                    pulses[pulse] = list()

                pulses[pulse].append(df[df.pulse == pulse].index[0])

        # find the first plane that comes immediately after the pulses
        first_pulse_planes = [np.where(frames == np.min(frames))[0][0] for frames in pulses.values()]

        # calculate where the first pulse frame would be in the combined frametimes
        start_frames = [n_planes * np.min(pulses[i]) + plane for i, plane in enumerate(first_pulse_planes)]

        if start_frames[0] == 0:
            start_frames.remove(0)

        self.align_pulses_to_frametimes(start_frames)

    def combine_regions(self, regions, remove_region_files=False, file_prefix=''):
        '''If a single imaging session consists of multiple "region"s, combine the combined tif files and the frametimes txts here.
        remove_region_files: if True, it will delete the separate region files to save space
        file_prefix: the prefix for the names of the combined files'''
        if not isinstance(regions, list):
            raise TypeError('regions needs to be a list of folder names, aka regions')

        img_paths = list()
        frametimes_paths = list()

        for region in regions:
            for entry in os.listdir(self.exp_path):
                if entry.startswith(region) and entry.endswith('_ch2.tif'):
                    img_paths.append(self.exp_path.joinpath(entry))
                elif entry.startswith(region) and entry.endswith('_frametimes.txt'):
                    frametimes_paths.append(self.exp_path.joinpath(entry))

        imgs = [tifffile.memmap(img_path) for img_path in img_paths]
        img = np.concatenate(imgs)

        if remove_region_files:
            for img_path in img_paths:
                os.remove(img_path)

        tifffile.imsave(self.exp_path.joinpath(f'{file_prefix}_ch2.tif'), img, bigtiff=True)

        with open(self.exp_path.joinpath(f'{file_prefix}_frametimes.txt'), 'w') as outfile:
            for ft_path in frametimes_paths:
                with open(ft_path) as infile:
                    for line in infile:
                        outfile.write(line)

        if remove_region_files:
            for ft_path in frametimes_paths:
                os.remove(ft_path)

        self.process_bruker_filestructure()


    def save_temporal(self):
        '''Saves the temporal components of final ROIs as a temporal.h5 file
        Also calculates the dF/F0 and adds it to the dataframe'''
        mes_df = uuid_to_plane(load_mesmerize(self))
        final_rois = load_rois(self)

        planes = list()
        raw_temporal = list()
        temporal = list()
        roi_indices = list()
        pulse_frames = list()

        for i, row in mes_df.iterrows():
            if row.algo == 'cnmf':

                try:
                    plane = get_plane_number(row)

                    name = row['item_name']
                    if name not in final_rois.keys():
                        continue

                    indices = final_rois[name]

                    raw_temp = row.cnmf.get_temporal("good", add_residuals=True)  # raw temporal responses: C+YrA
                    raw_temporal.append(raw_temp[indices])

                    planes.append(int(plane))
                    roi_indices.append(indices)

                    temp = row.cnmf.get_temporal('good')  # denoised temporal responses: C
                    temporal.append(temp[indices])

                    fts = pd.read_hdf(self.data_paths['volumes'][plane]['frametimes'])
                    try:
                        pulses = [fts[fts.pulse == pulse].index.values[0] for pulse in fts.pulse.unique() if pulse != fts.loc[0, 'pulse']]
                    except:
                        pulses = [0]

                    if 'DOI' in str(self.data_paths['raw']) and pulses == [0]:
                        with os.scandir(self.exp_path) as entries:
                            for entry in entries:
                                if '_pre' in entry.name:
                                    pre_path = Path(entry.path)

                        pulses = [len([file for file in os.listdir(pre_path) if file.endswith('.ome.tif')])]

                    pulse_frames.append(pulses)

                    print(f'finished plane {plane}')
                
                except ValueError:
                    # if none of the cells made it
                    pass

        temporal_df = pd.DataFrame({'plane': planes,
                                    'raw_temporal': raw_temporal,
                                    'temporal': temporal,
                                    'roi_indices': roi_indices,
                                    'pulse_frames': pulse_frames})
        temporal_df.sort_values(by=['plane'], ignore_index=True, inplace=True)
        temporal_df.to_hdf(self.exp_path.joinpath('temporal.h5'), key='temporal')

        self.process_bruker_filestructure()


    def normalize_temporaldf(self):
        '''Normalizes both the raw and denoised traces between 0 and 1'''
        self.temporal_df['norm_temporal'] = None
        self.temporal_df['raw_norm_temporal'] = None

        for i, row in self.temporal_df.iterrows():
            norm_temporals = list()
            raw_norm_temporals = list()

            for comp in row.temporal:
                norm_temporal = (comp - min(comp)) / (max(comp) - min(comp))
                norm_temporals.append(norm_temporal)

            for comp in row.raw_temporal:
                raw_norm_temporal = (comp - min(comp)) / (max(comp) - min(comp))
                raw_norm_temporals.append(raw_norm_temporal)

            self.temporal_df['norm_temporal'][i] = norm_temporals
            self.temporal_df['raw_norm_temporal'][i] = raw_norm_temporals

        self.temporal_df.to_hdf(self.exp_path.joinpath('temporal.h5'), key='temporal')

        return self.temporal_df
    

    def add_coms_to_temporaldf(self):
        '''Adds a column for centers of mass for each "good" neuron'''
        self.temporal_df["coms"] = None

        mes_df = uuid_to_plane(load_mesmerize(self))
        for i, row in self.temporal_df.iterrows():
            plane = f'img_stack_{row.plane}'
            mes_row = mes_df[(mes_df.algo == 'cnmf') & (mes_df.item_name == plane)].iloc[0]

            _, coms = mes_row.cnmf.get_contours('good', swap_dim=False)  
            coms = np.array(coms)
            coms = coms[row.roi_indices]  # get the accepted components
            
            self.temporal_df["coms"][i] = coms

        self.temporal_df.to_hdf(self.exp_path.joinpath('temporal.h5'), key='temporal')

        return self.temporal_df
    
    def get_microns_per_pixel(self):
        """
        Extracts the microns per pixel values for the X, Y, and (if volumetric) Z axes from a Bruker 2P log file.
        Returns:
            tuple: A tuple containing the microns per pixel values for the X, Y, and (if volumetric) Z axes.
        """
        with open(self.data_paths["log"], 'r') as file:
            log = file.read()

        Bs_data = BeautifulSoup(log, 'html.parser')

        microns_per_pixel = Bs_data.find_all('pvstatevalue', {'key': 'micronsPerPixel'})
        x_microns_per_pixel = float(microns_per_pixel[0].find('indexedvalue', {'index': 'XAxis'})['value'])
        y_microns_per_pixel = float(microns_per_pixel[0].find('indexedvalue', {'index': 'YAxis'})['value'])

        if self.volumetric:
            z_microns_per_pixel = float(microns_per_pixel[0].find('indexedvalue', {'index': 'ZAxis'})['value'])
            return (x_microns_per_pixel, y_microns_per_pixel, z_microns_per_pixel)
        else:
            return (x_microns_per_pixel, y_microns_per_pixel)
