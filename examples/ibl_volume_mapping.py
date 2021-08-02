import numpy as np
import pandas as pd
import pickle
import argparse

from iblviewer.launcher import IBLViewer
from iblviewer import utils

class DataViewer():

    def __init__(self, file_path=None, aggregator='median', grouper='acronym', color_map='viridis'):
        """
        Constructor
        """
        self.file_path = file_path
        self.aggregator = aggregator
        self.color_map = color_map
        self.grouper = grouper
        self.df = None
        self.aggregated_df = None
        self.min_value = None
        self.max_value = None
        self.load_data(file_path)

    def load_data(self, file_path=None, silent=True):
        if file_path is None:
            test_data = './stimonR_top10_rawpoints.p'
            file_path = str(utils.EXAMPLES_DATA_FOLDER.joinpath(test_data))
            self.file_path = file_path
        
        df = pickle.load(open(file_path, 'rb'))
        
        # For testing data given by Guido
        # df['r_over_chance'] = df['r_prior'] - df['r_prior_null']
        # filtered_df = df.groupby('region').median()['r_over_chance']

        raw_df = df['rawpoints']
        copy_df = raw_df.copy()
        agg_df = copy_df.groupby(self.grouper).agg({'value': self.aggregator})
        agg_df.dropna(inplace=True)
        self.min_value = float(np.amin(agg_df, axis=0).to_numpy()[0])
        self.max_value = float(np.amax(agg_df, axis=0).to_numpy()[0])

        if not silent:
            print('Min prior value ' + str(self.min_value))
            print('Max prior value ' + str(self.max_value))

        self.df = raw_df
        self.df.dropna(inplace=True)
        self.df.sort_values(by='acronym', key=lambda col: col.str.lower())
        self.aggregated_df = agg_df
        
    #pLeft_iti_scores_n_gt_50
    # Data given by Berk Gerçek, International Brain Laboratory
    def get_scalars_map(self, viewer):
        """
        Process priors data and get color map and scalar values
        """
        scalars_map = [None]*viewer.ibl_model.get_num_regions()

        # This code is to be modified if you have split data for left and right hemispheres
        # The concept is pretty simple: scalars_map is a 1D list that maps to brain regions.
        # With the lateralized brain mapping, the standard region id in Allen CCF is negated
        # on the right hemisphere. 
        # Currently this code uses standard acronym lookup, which yields a region on both
        # hemispheres. The value you assign to an acronym will thus be mirrored.

        # Or for i in range(0, len(df)): which preserves data types
        for acronym, row in self.aggregated_df.iterrows():
            value = row.iloc[0]
            if value is None:
                continue
            region_ids, row_ids = viewer.ibl_model.get_region_and_row_id(acronym)
            if region_ids is None:
                print('Acronym', acronym, 'was not found in Atlas')
                continue
            for r_id in range(len(region_ids)):
                region_id = region_ids[r_id]
                row_id = row_ids[r_id]
                if region_id is None:
                    print('Error, could not find acronym', acronym, '...ignoring it)')
                    continue
                if row_id == 0: #or value.isnull().values.any():
                    # We ignore void acronym (which is equal to row_id 0) on Allen Mouse CCF v3
                    continue
                #print('Setting row', int(row_id), 'with value', value)
                scalars_map[int(row_id)] = value
        return scalars_map
        
    def on_viewer_initialized(self, viewer):
        """
        Method called when the viewer is initialized and ready to accept further code by the user.
        In this case we map prior data to the Allen Mouse brain atlas.
        :param viewer: MouseBrainViewer instance (mandatory)
        """
        scalars_map = self.get_scalars_map(viewer)
        # Ensure we have the right selection, here 0 is the first registered object: the atlas
        viewer.select(-1)
        viewer.assign_scalars(scalars_map, [self.min_value, self.max_value], self.color_map)

    def on_statistics_update(self, statistics, viewer):
        """
        Method called when statistics are updated. Here we use a scatter plot.
        :param statistics: MplCanvas instance (mandatory)
        :param viewer: MouseBrainViewer instance (mandatory)
        """
        # Clear previous plot
        statistics.axes.clear()

        # Prepare new one
        agg_df = self.aggregated_df
        statistics.axes.scatter(self.df.acronym, self.df.value, alpha=0.2, s=8)
        # There are multiple ways to retrieve the acronym, here's one
        acronyms = agg_df.index[agg_df.value == viewer.model.selection_related_value].tolist()
        if acronyms is None or len(acronyms) < 1:
            return
        acronym = acronyms[0]
        selected_data = self.df.value[self.df.acronym == acronym]
        selected_data.dropna(inplace=True)
        statistics.axes.scatter(['']*len(selected_data), selected_data, color='yellow', s=32)
        statistics.axes.set_xlabel('Brain regions')


if __name__ == '__main__':

    # More parsing options are added in parse_args() method below.
    # -> Please check that you don't override any existing argument name!
    parser = argparse.ArgumentParser(description='International Brain Viewer based on VTK')
    parser.add_argument('-f', dest='file_path', type=str, default=None, 
    help='File path to your data. None will load a default example.')
    parser.add_argument('-g', dest='grouper', type=str, default='acronym', 
    help='Grouper type. Defaults to "acronym" column.')
    parser.add_argument('-a', dest='aggregator', type=str, default='median', 
    help='Aggregator type. Defaults to median.')

    iblviewer = IBLViewer()
    # First retrieve command-line arguments (default ones + custom ones above)
    args = iblviewer.parse_args(parser)

    # Then update our custom class with those arguments
    data = DataViewer(args.file_path, args.aggregator, args.grouper, args.color_map)

    # Finally, launch the UI and 3D viewer
    iblviewer.launch(data.on_viewer_initialized, data.on_statistics_update, args)