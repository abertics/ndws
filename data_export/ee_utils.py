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

from datetime import datetime

import os
from os import path as osp
from tqdm.auto import tqdm

import enum
import math
import random

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path().absolute().parent / 'modified_ndws.env')

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
	LANDFIRE_FUELS = 7
	
	VEGETATION_VIIRS = 8
	DROUGHT_GRIDMET = 9 
	WEATHER_GRIDMET = 10
	WEATHER_RTMA = 11
	
	FIRE_VIIRS = 12
	FIRE_MODIS = 13
	

DATA_SOURCES = {
	DataType.ELEVATION_SRTM: 'USGS/SRTMGL1_003', #  ee.Image
	DataType.CHILI: 'CSP/ERGo/1_0/US/CHILI', #  ee.Image
	DataType.IMPERVIOUS: 'USGS/NLCD_RELEASES/2019_REL/NLCD', #  ee.ImageCollection
	DataType.WTR_COVER: 'JRC/GSW1_4/GlobalSurfaceWater', #  ee.Image
	DataType.POPULATION: 'CIESIN/GPWv411/GPW_Population_Density', #  ee.ImageCollection
	
	DataType.LATENT_FUELS: os.environ.get("LATENT_FUEL_ASSETS").split(','), # this asset is hosted on Earth Engine
	DataType.LANDFIRE_FUELS: os.environ.get("LANDFIRE_ASSETS").split(','), # this asset is hosted on Earth Engine

	DataType.VEGETATION_VIIRS: 'MODIS/061/MOD13Q1', #  ee.ImageCollection
	DataType.DROUGHT_GRIDMET: 'GRIDMET/DROUGHT', #  ee.ImageCollection
	DataType.WEATHER_GRIDMET: 'IDAHO_EPSCOR/GRIDMET', #  ee.ImageCollection
	DataType.WEATHER_RTMA: 'NOAA/NWS/RTMA',

	DataType.FIRE_MODIS: 'MODIS/061/MOD14A1', 
	DataType.FIRE_VIIRS:os.environ.get("VIIRS_ASSETS").split(',') # this asset is hosted on Earth Engine
}

DATA_BANDS = {
	DataType.ELEVATION_SRTM: ['elevation'],
	DataType.CHILI: ['constant'],
	DataType.IMPERVIOUS: ['impervious'],
	DataType.WTR_COVER: ['occurrence'],
	DataType.POPULATION: ['population_density'],

	DataType.LATENT_FUELS: ['b1','b2','b3'],
	DataType.LANDFIRE_FUELS: ['b1'],
	
	DataType.VEGETATION_VIIRS: ['NDVI'],
	DataType.DROUGHT_GRIDMET: ['pdsi'],
	DataType.WEATHER_GRIDMET: [
		'pr',
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
		# 'ACP01',
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
	DataType.WEATHER_RTMA: 1, # this data is sampled hourly, but we want to aggregate over the past day
}


RESAMPLING_SCALE = {DataType.WEATHER_GRIDMET: 10000,
					DataType.DROUGHT_GRIDMET: 10000,
					DataType.WEATHER_RTMA: 10000,
}

DETECTION_BAND = 'detection'
DEFAULT_KERNEL_SIZE = 64
DEFAULT_SAMPLING_RESOLUTION = 500 # Units: meters
DEFAULT_EVAL_SPLIT = 0.2
DEFAULT_LIMIT_PER_EE_CALL = 30
DEFAULT_SEED = 123

VIIRS_SCALE = 375 # Units: meters (not in use currently)

COORDINATES = {
	# Used as input to ee.Geometry.Rectangle().
	'US': [-124, 24, -73, 49],
	'SIERRA_NEVADA': [-121.0, 37.0, -119.0, 39.0],
	'OREGON': [-124.6, 41.9, -116.4, 46.3],
	'US_WEST': [-125, 26, -100, 49], # These coordinates cut off at the 'arid' meridian 100 W
	'WEST_COAST':[-124.848974, 32.528832, -113.915989, 49.002494],
}



def viirs_fire_to_imagecollection(table):
	"""
	Convert VIIRS fire detection points to a 500m binary grid ImageCollection for CONUS.
	
	Args:
	table (ee.FeatureCollection): FeatureCollection of VIIRS fire detections.
	
	Returns:
	ee.ImageCollection: ImageCollection of binary grids indicating fire detections.
	"""
	
	# Define CONUS boundary
	conus = ee.Geometry.Rectangle([-124.848974, 24.396308, -66.885444, 49.384358])

	def create_image_for_date(date):
		# Filter points for this date, CONUS, and confidence levels 'n' or 'h'
		points_for_date = table.filter(ee.Filter.And(
			ee.Filter.eq('ACQ_DATE', date),
			ee.Filter.bounds(conus),
			ee.Filter.Or(
				ee.Filter.eq('CONFIDENCE', 'n'), # keep nominal and high confidence detections
				ee.Filter.eq('CONFIDENCE', 'h')
			)))
		
		# Create a binary image from the points
		binary_image = ee.Image().byte().paint(points_for_date, 1)

		# Create a fixed-size kernel (approximately 375m)
		kernel = ee.Kernel.circle(radius=375, units='meters')
		
		# Apply the kernel and threshold to create a binary mask
		kernelized_image = binary_image.convolve(kernel).gt(0)
		
		# Resample to 500m resolution and unmask
		resampled_image = kernelized_image.reproject(
			crs='EPSG:5070',  # USA Contiguous Albers Equal Area Conic
			scale=500
		).unmask(0)  # Set masked areas to 0
		
		# Set properties
		ee_date = ee.Date(date)
		formatted_date = ee_date.format('YYYY-MM-dd')
		
		return resampled_image.set({
			'date': formatted_date,
			'system:time_start': ee_date.millis()}).byte()  # Ensure the image is byte type

	# Get unique dates
	dates = table.filter(ee.Filter.And(
		ee.Filter.bounds(conus),
		ee.Filter.Or(
			ee.Filter.eq('CONFIDENCE', 'n'),
			ee.Filter.eq('CONFIDENCE', 'h')
		)
	)).aggregate_array('ACQ_DATE').distinct()

	# Create an image for each date
	image_collection = ee.ImageCollection(dates.map(create_image_for_date))

	return image_collection

def get_image(data_type):
  """Gets an image corresponding to `data_type`.

  Args:
	data_type: A specifier for the type of data.

  Returns:
	The EE image correspoding to the selected `data_type`.
  """
  return ee.Image(DATA_SOURCES[data_type]).select(DATA_BANDS[data_type])


def get_image_collection(data_type,viirs_info = None):
	# viirs_folder = VIIRS_DATA_FOLDER
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
		
		colls = []
		for table_id in DATA_SOURCES[data_type]:
			features = ee.FeatureCollection(table_id)
			coll = viirs_fire_to_imagecollection(features)
			colls.append(coll)

		final_coll = colls[0]
		for x in colls[1:]:
			final_coll = final_coll.merge(x)
		return final_coll.select('constant')

	# elif data_type == DataType.LANDFIRE_FUELS:
	# 	images = []
	# 	for image_id in DATA_SOURCES[data_type]:
	# 		if '2014' in image_id:
	# 			year = 2014
	# 		elif '2019' in image_id:
	# 			year = 2019
	# 		img = ee.Image(image_id).select(DATA_BANDS[data_type])
	# 		img = img.set('system:time_start', ee.Date.fromYMD(year, 1, 1).millis())
	# 		images.append(img)
	# 	return ee.ImageCollection(images)
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
	sampling_scale=DEFAULT_SAMPLING_RESOLUTION,
	detection_band=DETECTION_BAND,
	):
	'''
	Note: this function has been modified from the original structure used by Huot et al. 
	The reason is that pulling the sum of detections across a large region can throw an error 
	due to the data requested being too large. This is avoided here by using server-side logic. 
	I feel that the old method (using a .sum() reducer) could be modified to avoid this error, but leave this
	for future development. 
	'''
	distinct_values = detection_image.select(detection_band) \
		.reduceRegion(
			reducer=ee.Reducer.countDistinct(),
			geometry=geometry,
			scale=sampling_scale,
			maxPixels=1e9
		).get(detection_band)

	return 1 if ee.List(distinct_values).contains(1) else 0

def extract_samples(
	image,
	detection_count,
	geometry,
	detection_band='detection',
	sampling_limit_per_call=DEFAULT_LIMIT_PER_EE_CALL,
	resolution=DEFAULT_SAMPLING_RESOLUTION,
	seed=DEFAULT_SEED,
	date=True
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
			dropNulls=True,
			geometries=True  # This ensures the geometry is included in the output
		)
		
		# Add lat/lon as properties
		samples = samples.map(lambda f: f.set({
			'longitude': f.geometry().coordinates().get(0),
			'latitude': f.geometry().coordinates().get(1)
		}))
		
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
	fire_season_only = False,
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
		ee.Date.difference(end_date, start_date, unit='days').getInfo())
	
	# Create list of all days and apply window length filtering
	days = list(range(num_days))[::window_length_days]
	random.shuffle(days)
	
	# Split into train/eval/test sets
	num_eval = int(len(days) * split_ratio)
	split_days = {}
	split_days['train'] = days[:-2 * num_eval]
	split_days['eval'] = days[-2 * num_eval:-num_eval]
	split_days['test'] = days[-num_eval:]
	return split_days
