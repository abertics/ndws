# Modified Next Day Wildfire Spread 

# uses higher resolution VIIRS data and adds latent fuel variables, water, impervious features

# George Hulsey, Sep. 2024

# Built on and utilizes a majority of the code associated with:

# F. Huot, R. L. Hu, N. Goyal, T. Sankar, M. Ihme and Y. -F. Chen, 
# 	"Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading From Remote-Sensing Data," 
# 	in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-13, 2022, Art no. 4412513, 
# 	doi: 10.1109/TGRS.2022.3192974.

# Original git repo: https://github.com/google-research/google-research/tree/762395598d6935dfbaa5ecb862145a34509b2c7c/simulation_research/next_day_wildfire_spread


"""Modified set of Earth Engine utility functions."""

import ee
import geemap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime, timedelta

import os
from os import path as osp
from tqdm.auto import tqdm

import enum
import math
import random
import glob

# Included to make the shuffling in split_days_into_train_eval_test
# deterministic.
random.seed(123)



class DataType(enum.Enum):
	ELEVATION_SRTM = 1
	CHILI = 2
	IMPERVIOUS = 3
	WTR_COVER = 4
	POPULATION = 5
	LATENT_FUELS = 6
	
	VEGETATION_VIIRS = 7
	DROUGHT_GRIDMET = 8 
	WEATHER_GRIDMET = 9
	WEATHER_RTMA = 10
	
	FIRE_VIIRS = 11

	FIRE_MODIS = 12
	

DATA_SOURCES = {
	DataType.ELEVATION_SRTM: 'USGS/SRTMGL1_003', #  ee.Image
	DataType.CHILI: 'CSP/ERGo/1_0/US/CHILI', #  ee.Image
	DataType.IMPERVIOUS: 'USGS/NLCD_RELEASES/2019_REL/NLCD', #  ee.ImageCollection
	DataType.WTR_COVER: 'JRC/GSW1_4/GlobalSurfaceWater', #  ee.Image
	DataType.POPULATION: 'CIESIN/GPWv411/GPW_Population_Density', #  ee.ImageCollection
	DataType.LATENT_FUELS: 'projects/ee-georgethulsey/assets/latent_fuels_CONUSwest', # ee.Image
	
	DataType.VEGETATION_VIIRS: 'NASA/VIIRS/002/VNP13A1', #  ee.ImageCollection
	DataType.DROUGHT_GRIDMET: 'GRIDMET/DROUGHT', #  ee.ImageCollection
	DataType.WEATHER_GRIDMET: 'IDAHO_EPSCOR/GRIDMET', #  ee.ImageCollection
	DataType.WEATHER_RTMA: 'NOAA/NWS/RTMA',

	DataType.FIRE_MODIS: 'MODIS/061/MOD14A1', 
}

DATA_BANDS = {
	DataType.ELEVATION_SRTM: ['elevation'],
	DataType.CHILI: ['constant'],
	DataType.IMPERVIOUS: ['impervious'],
	DataType.WTR_COVER: ['occurrence'],
	DataType.POPULATION: ['population_density'],
	DataType.LATENT_FUELS: ['b1','b2','b3'],
	
	DataType.VEGETATION_VIIRS: ['NDVI'],
	DataType.DROUGHT_GRIDMET: ['pdsi'],
	DataType.WEATHER_GRIDMET: [
		#'pr',
		#'sph',
		#'th',
		#'tmmn',
		#'tmmx',
		#'vs',
		'erc',
		'bi',
	],
	DataType.WEATHER_RTMA: [
		'TMP',
		'UGRD',
		'VGRD',
		'SPFH',
		'WDIR',
		'WIND',
		'GUST',
		'ACP01',
	],
	
	DataType.FIRE_MODIS: ['FireMask'],
	DataType.FIRE_VIIRS: ['viirs_FireMask'],
}


# The time unit is 'days'.
DATA_TIME_SAMPLING = {
	DataType.VEGETATION_VIIRS: 16, # changed this as NDVI was failing in some instances
	DataType.DROUGHT_GRIDMET: 5,
	DataType.WEATHER_GRIDMET: 1,
	DataType.FIRE_MODIS: 1,
	DataType.FIRE_VIIRS: 1,
	DataType.WEATHER_RTMA: 1,
}


RESAMPLING_SCALE = {DataType.WEATHER_GRIDMET: 10000,
					DataType.DROUGHT_GRIDMET: 10000,
					DataType.WEATHER_RTMA: 5000,
}

DETECTION_BAND = 'detection'
DEFAULT_KERNEL_SIZE = 64
DEFAULT_SAMPLING_RESOLUTION = 500 # Units: meters
DEFAULT_EVAL_SPLIT = 0.2
DEFAULT_LIMIT_PER_EE_CALL = 60
DEFAULT_SEED = 123

VIIRS_DATA_FOLDER = 'VIIRS_data' # script-relative path to folder containing VIIRS csv data
VIIRS_SCALE = 375 # Units: meters

COORDINATES = {
	# Used as input to ee.Geometry.Rectangle().
	'US': [-124, 24, -73, 49],
	'SIERRA_NEVADA': [-121.0, 37.0, -119.0, 39.0],
	'OREGON': [-124.6, 41.9, -116.4, 46.3],
	'US_WEST': [-125, 26, -100, 49], # These coordinates cut off at the 'arid' meridian 100 W
}


def read_viirs_csv(file_path):
	"""
	Reads VIIRS csv files and processes the dataframe. 

	The data product is desribed at https://www.earthdata.nasa.gov/learn/find-data/near-real-time/firms/viirs-i-band-375-m-active-fire-data

	Args: 
		file_path: script-relative path for the VIIRS csv file

	Returns: 
		Cleaned and processed DataFrame

	"""
	df = pd.read_csv(file_path)
	df['datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'].astype(str).str.zfill(4), format='%Y-%m-%d %H%M')
	
	# keep nominal and high confidence as positive and otherwise set to zero
	df['confidence'] = df['confidence'].apply(lambda x: 1 if x == 'n' or x == 'h' else 0)
	# Currently, this script takes all observations, even if they are low confidence predictions. 
	
	# filtering step
	df = df[df['confidence'] == 1]
	
	df.drop(columns = ['scan','track','instrument','version','type','daynight','satellite'],inplace = True)
	return df


def viirs_df_to_ee_image(df, date):
	"""
	Takes a VIIRS dataframe and a date and constructs an ee.Image object of all observations on that date. 

	This code should be optimized by converting all the VIIRS dataframes into an ee.ImageCollection object, 
	but this isn't necessary at all. 

	Args:
		df: VIIRS dataframe
		date: formatted as %Y-%m-%d %H%M

	Returns:
		ee.Image with mask removed

	"""
	
	viirs_scale = VIIRS_SCALE
	
	df_date = df[df['acq_date'] == date]
	
	# Create a feature collection from the filtered dataframe
	features = [ee.Feature(
		ee.Geometry.Point([row['longitude'], row['latitude']]),
		{
			'fire_mask': 1 #if row['confidence'] == 1 and row['frp'] > 0 else 0
		}
	) for _, row in df_date.iterrows()]
	
	# Create a FeatureCollection
	fc = ee.FeatureCollection(features)

	# Get the bounding box of the FeatureCollection
	bbox = fc.geometry().bounds()

	# Create a 375m resolution grid of zeros
	grid = ee.Image.constant(0).clip(bbox)

	# Create a binary image from the point data
	point_image = ee.Image().byte().paint(fc, 1)

	# Reproject the point image to match the grid
	point_image = point_image.reproject(ee.Projection('EPSG:4326').atScale(viirs_scale))

	# Combine the grid and point image
	fire_mask_image = grid.max(point_image).rename('viirs_FireMask')
	
	return fire_mask_image.set('system:time_start', ee.Date(date).millis()).unmask()

def get_image(data_type):
  """Gets an image corresponding to `data_type`.

  Args:
	data_type: A specifier for the type of data.

  Returns:
	The EE image correspoding to the selected `data_type`.
  """
  return ee.Image(DATA_SOURCES[data_type]).select(DATA_BANDS[data_type])


def get_image_collection(data_type,viirs_info = None):
	viirs_folder = VIIRS_DATA_FOLDER
	"""Gets an image collection corresponding to `data_type`.

	In the (special) case that we want the VIIRS data, we have to pass the start date and end date, and we construct an
	ee.ImageCollection from the VIIRS csv data. 

	Otherwise, this function is very straightforward. 
	
	Args:
	data_type: A specifier for the type of data.
	
	Returns:
	The EE image collection corresponding to `data_type`.
	"""
	
	if data_type == DataType.FIRE_VIIRS:
		start_date,end_date = viirs_info
		if not viirs_folder:
			raise ValueError("VIIRS folder path is required for FIRE_VIIRS data type")
		
		start_year = int(start_date.split('-')[0])
		end_year = int(end_date.split('-')[0])
		csv_files = glob.glob(os.path.join(viirs_folder, '*.csv'))
		
		print("Reading VIIRS dataframes...")
		dfs = []
		for file in csv_files:
			year = int(file.split('_')[2])
			if start_year <= year <= end_year:
				df = read_viirs_csv(file)
				# Filter by date range and geographic bounds
				df = df[(df['datetime'] >= pd.to_datetime(start_date)) & 
						(df['datetime'] <= pd.to_datetime(end_date))]
				dfs.append(df)
		
		df = pd.concat(dfs)
		
		unique_dates = df['acq_date'].unique()
		images = [viirs_df_to_ee_image(df[df['acq_date'] == date], date) for date in tqdm(unique_dates)]
		return ee.ImageCollection(images)
	else:
		return ee.ImageCollection(DATA_SOURCES[data_type]).select(DATA_BANDS[data_type])


def remove_mask(image):
  """Removes the mask from an EE image.

  Args:
	image: The input EE image.

  Returns:
	The EE image without its mask.
  """
  mask = ee.Image(1)
  return image.updateMask(mask)

def eeDate_to_string(ee_date):
	dt = datetime.fromtimestamp(ee_date.millis().getInfo()/1000)
	return dt.strftime('%Y-%m-%d %H:%M:%S')


def export_feature_collection(
	feature_collection,
	description,
	bucket,
	folder,
	bands,
	file_format = 'TFRecord',
):
  """Starts an EE task to export `feature_collection` to TFRecords.

  Args:
	feature_collection: The EE feature collection to export.
	description: The filename prefix to use in the export.
	bucket: The name of the Google Cloud bucket.
	folder: The folder to export to.
	bands: The list of names of the features to export.
	file_format: The output file format. 'TFRecord' and 'GeoTIFF' are supported.

  Returns:
	The EE task associated with the export.
  """
  task = ee.batch.Export.table.toCloudStorage(
	  collection=feature_collection,
	  description=description,
	  bucket=bucket,
	  fileNamePrefix=os.path.join(folder, description),
	  fileFormat=file_format,
	  selectors=bands)
  task.start()
  return task


def convert_features_to_arrays(
	image_list,
	kernel_size = DEFAULT_KERNEL_SIZE,
	):
	"""Converts a list of EE images into `(kernel_size x kernel_size)` tiles.
	Args:
	image_list: The list of EE images.
	kernel_size: The size of the tiles (kernel_size x kernel_size).

	Returns:
	An EE image made of (kernel_size x kernel_size) tiles.
	"""
	feature_stack = ee.Image.cat(image_list).float()
	kernel_list = ee.List.repeat(1, kernel_size)  # pytype: disable=attribute-error
	kernel_lists = ee.List.repeat(kernel_list, kernel_size)  # pytype: disable=attribute-error
	kernel = ee.Kernel.fixed(kernel_size, kernel_size, kernel_lists)
	return feature_stack.neighborhoodToArray(kernel)

def get_detection_count(
	detection_image,
	geometry,
	sampling_scale = DEFAULT_SAMPLING_RESOLUTION,
	detection_band = DETECTION_BAND,
):
	"""Counts the total number of positive pixels in the detection image.

	Assumes that the pixels in the `detection_band` of `detection_image` are
	zeros and ones.

	This is used to determine where fires have been detected. 

	Args:
	detection_image: An EE image with a detection band.
	geometry: The EE geometry over which to count the pixels.
	sampling_scale: The sampling scale used to count pixels.
	detection_band: The name of the image band to use.

	Returns:
	The number of positive pixel counts or -1 if EE throws an error.
	"""
	detection_stats = detection_image.reduceRegion(
	  reducer=ee.Reducer.sum(), geometry=geometry, scale=sampling_scale)
	try:
		detection_count = int(detection_stats.get(detection_band).getInfo())
	except ee.EEException:
		# If the number of positive pixels cannot be counted because of a server-
		# side error, return -1.
		detection_count = -1
	return detection_count


def extract_samples(
	image,
	detection_count,
	geometry,
	detection_band='detection',
	sampling_limit_per_call=DEFAULT_LIMIT_PER_EE_CALL,
	resolution=DEFAULT_SAMPLING_RESOLUTION,
	seed=DEFAULT_SEED,
	date=None
):
	"""Samples an EE image for positive and negative samples.

	Extracts `detection_count` positive examples and (`sampling_ratio` x
	`detection_count`) negative examples. Assumes that the pixels in the
	`detection_band` of `detection_image` are zeros and ones.

	Args:
		image: The EE image to extract samples from.
		detection_count: The number of positive samples to extract.
		geometry: The EE geometry over which to sample.
		sampling_ratio: If sampling negatives examples, samples (`sampling_ratio` x
		  `detection_count`) negative examples. When extracting only positive
		  examples, set this to zero.
		detection_band: The name of the image band to use to determine sampling
		  locations.
		sampling_limit_per_call: The limit on the size of EE calls. Can be used to
		  avoid memory errors on the EE server side. To disable this limit, set it
		  to `detection_count`.
		resolution: The resolution in meters at which to scale.
		seed: The number used to seed the random number generator. Used when
		  sampling less than the total number of pixels.
		date: The date of the image, to be included as metadata.

	Returns:
		An EE feature collection with all the extracted samples.
	"""
	feature_collection = ee.FeatureCollection([])
	num_per_call = sampling_limit_per_call 

	# Add date as a band to the image
	if date:
		image = image.addBands(ee.Image.constant(ee.Date(date).millis()).rename('date'))
	# Date metadata is added to aid in finding specific fires

	# The sequence of sampling calls is deterministic, so calling stratifiedSample
	# multiple times never returns samples with the same center pixel.
	for _ in range(math.ceil(detection_count / num_per_call)):

		samples = image.stratifiedSample(
			region=geometry,
			numPoints=0,
			classBand=detection_band,
			scale=resolution,
			seed=seed,
			classValues=[0, 1],
			classPoints=[0, num_per_call],
			dropNulls=True
		)
		
		feature_collection = feature_collection.merge(samples)
		
	
	# # Add overall geometry as metadata to the feature collection
	# feature_collection = feature_collection.set('overall_geometry', geometry)
	# this is not currently implemented, but can be used to retain the geolocation of the fire. 
	
	return feature_collection

def split_days_into_train_eval_test(
	start_date,
	end_date,
	split_ratio = DEFAULT_EVAL_SPLIT,
	window_length_days = 8,
):
	"""Splits the days into train / eval / test sets.

	Splits the interval between  `start_date` and `end_date` into subintervals of
	duration `window_length` days, and divides them into train / eval / test sets.

	Args:
	start_date: The start date.
	end_date: The end date.
	split_ratio: The split ratio for the divide between sets, such that the
	  number of eval time chunks and test time chunks are equal to the total
	  number of time chunks x `split_ratio`. All the remaining time chunks are
	  training time chunks.
	window_length_days: The length of the time chunks (in days).

	Returns:
	A dictionary containing the list of start day indices of each time chunk for
	each set.
	"""
	random.seed(123)
	num_days = int(
	  ee.Date.difference(end_date, start_date, unit='days').getInfo())  # pytype: disable=attribute-error
	days = list(range(num_days))
	days = days[::window_length_days]
	random.shuffle(days)
	num_eval = int(len(days) * split_ratio)
	split_days = {}
	split_days['train'] = days[:-2 * num_eval]
	split_days['eval'] = days[-2 * num_eval:-num_eval]
	split_days['test'] = days[-num_eval:]
	return split_days
