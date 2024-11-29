import random

import ee

import ee_utils # library of functions

# Included to make the shuffling in split_days_into_train_eval_test
# deterministic.
random.seed(123)


def _get_all_feature_bands():
    """Returns list of all bands corresponding to features."""
    return (ee_utils.DATA_BANDS[ee_utils.DataType.ELEVATION_SRTM] +
            ['chili'] +
            ee_utils.DATA_BANDS[ee_utils.DataType.IMPERVIOUS] + 
            ['water'] + 
            ['population'] + 
            ['fuel1','fuel2','fuel3'] + ['landfire_fuel'] +

            ee_utils.DATA_BANDS[ee_utils.DataType.VEGETATION_VIIRS] +
            ee_utils.DATA_BANDS[ee_utils.DataType.DROUGHT_GRIDMET] +
            ee_utils.DATA_BANDS[ee_utils.DataType.WEATHER_GRIDMET] +
            [
                'avg_sph',
                'tmp_day',
                'tmp_75',
                'gust_med',
                'wind_avg',
                'wind_75',
                'wdir_wind',
                'wdir_gust'
            ] +
            # ['modis_PrevFireMask']+
            ['viirs_PrevFireMask']
            )

def _get_all_response_bands():
    """Returns list of all bands corresponding to labels."""
    return ['viirs_FireMask'] #+ ['modis_FireMask'] 


def _add_index(i, bands):
    """Appends the index number `i` at the end of each element of `bands`."""
    return [f'{band}_{i}' for band in bands]

def _get_all_image_collections(viirs_info = None):
    """Gets all the image collections and corresponding time sampling."""
    image_collections = {
      'drought':
          ee_utils.get_image_collection(ee_utils.DataType.DROUGHT_GRIDMET),
      'vegetation':
          ee_utils.get_image_collection(ee_utils.DataType.VEGETATION_VIIRS),
      'weather_gridmet':
          ee_utils.get_image_collection(ee_utils.DataType.WEATHER_GRIDMET),
      'weather_rtma':
          ee_utils.get_image_collection(ee_utils.DataType.WEATHER_RTMA),
      # 'modis_fire':
      #     ee_utils.get_image_collection(ee_utils.DataType.FIRE_MODIS),
      'viirs_fire':
          ee_utils.get_image_collection(ee_utils.DataType.FIRE_VIIRS),
    }
    time_sampling = {
      'drought':
          ee_utils.DATA_TIME_SAMPLING[ee_utils.DataType.DROUGHT_GRIDMET],
      'vegetation':
          ee_utils.DATA_TIME_SAMPLING[ee_utils.DataType.VEGETATION_VIIRS],
      'weather_gridmet':
          ee_utils.DATA_TIME_SAMPLING[ee_utils.DataType.WEATHER_GRIDMET],
      'weather_rtma':
          ee_utils.DATA_TIME_SAMPLING[ee_utils.DataType.WEATHER_RTMA],
      # 'modis_fire':
      #     ee_utils.DATA_TIME_SAMPLING[ee_utils.DataType.FIRE_MODIS],
      'viirs_fire':
          ee_utils.DATA_TIME_SAMPLING[ee_utils.DataType.FIRE_VIIRS],
    }
    return image_collections, time_sampling

def _verify_feature_collection(
    feature_collection
    ):
    """Verifies the feature collection is valid.
    
    If the feature collection is invalid, resets the feature collection.
    
    Args:
    feature_collection: An EE feature collection.
    
    Returns:
    `(feature_collection, size)` a tuple of the verified feature collection and
    its size.
    """
    # WARNING: this is a bit of a hack, and may not work for large feature collections
    # This often throws an error due to the size of the feature collection. For exporting data over large areas, this is a problem.
    try:
        size = int(feature_collection.size().getInfo())
        
        print("Verified a feature collection of size",size) # debugging
        
    except ee.EEException:
        print('EE exception thrown, resetting') # debugging
        
        # Reset the feature collection
        feature_collection = ee.FeatureCollection([])
        size = 0
    
    return feature_collection, size


def _get_time_slices(
    window_start,
    window = 1,
    projection = ee.Projection('EPSG:4326'),  # Defer calling until called by test code
    resampling_scale = 500, # units: meters
    lag = 0,
):
    """Extracts the time slice features.
    
    Args:
    window_start: Start of the time window over which to extract data.
    window: Length of the window (in days).
    projection: projection to reproject all data into.
    resampling_scale: length scale to resample data to.
    lag: Number of days before the fire to extract the features.
    
    Returns:
    A list of the extracted EE images.
    """
    projection = projection.atScale(resampling_scale) # reprojection flow is a bit confusing
    
    window_end = window_start.advance(window, 'day')
    viirs_info = (ee_utils.eeDate_to_string(window_start),ee_utils.eeDate_to_string(window_end)) 

    image_collections, time_sampling = _get_all_image_collections(viirs_info = viirs_info)

    # aggregate out-of-the-box features
    drought = image_collections['drought'].filterDate(
      window_start.advance(-lag - time_sampling['drought'], 'day'),
      window_start.advance(-lag, 'day')).median().reproject(
        projection.atScale(ee_utils.RESAMPLING_SCALE[ee_utils.DataType.DROUGHT_GRIDMET])
            ).resample('bicubic')
    
    vegetation = image_collections['vegetation'].filterDate(
      window_start.advance(-lag - time_sampling['vegetation'], 'day'),
      window_start.advance(
          -lag, 'day')).median().unmask()#.reproject(projection).resample('bicubic')
    
    weather_gridmet = image_collections['weather_gridmet'].filterDate(
      window_start.advance(-lag - time_sampling['weather_gridmet'], 'day'),
      window_start.advance(-lag, 'day')).median().reproject(
          projection.atScale(ee_utils.RESAMPLING_SCALE[ee_utils.DataType.WEATHER_GRIDMET])
            ).resample('bicubic')

    # Create bespoke weather features from hourly data
    avg_humidity = image_collections['weather_rtma'].filterDate(
      window_start.advance(-lag - time_sampling['weather_rtma'], 'day'),
      window_start.advance(-lag, 'day')).select('SPFH').mean().reproject(
          projection.atScale(ee_utils.RESAMPLING_SCALE[ee_utils.DataType.WEATHER_RTMA])
            ).resample('bicubic').rename('avg_sph')

    daytime_temp = image_collections['weather_rtma'].filterDate(
        window_start.advance(-lag - time_sampling['weather_rtma'], 'day'),
        window_start.advance(-lag, 'day')) \
        .select('TMP') \
        .filter(ee.Filter.calendarRange(16, 4, 'hour')) \
        .median() \
        .reproject(
            projection.atScale(ee_utils.RESAMPLING_SCALE[ee_utils.DataType.WEATHER_RTMA])
        ).resample('bicubic').rename('tmp_day')

    q75_temp = image_collections['weather_rtma'].filterDate(
        window_start.advance(-lag - time_sampling['weather_rtma'], 'day'),
        window_start.advance(-lag, 'day')) \
        .select('TMP') \
        .reduce(ee.Reducer.percentile([75])) \
        .reproject(
            projection.atScale(ee_utils.RESAMPLING_SCALE[ee_utils.DataType.WEATHER_RTMA])
        ).resample('bicubic').rename('tmp_75')

    median_gust = image_collections['weather_rtma'].filterDate(
        window_start.advance(-lag - time_sampling['weather_rtma'], 'day'),
        window_start.advance(-lag, 'day')) \
        .select('GUST') \
        .median() \
        .reproject(
            projection.atScale(ee_utils.RESAMPLING_SCALE[ee_utils.DataType.WEATHER_RTMA])
        ).resample('bicubic').rename('gust_med')

    avg_wind = image_collections['weather_rtma'].filterDate(
        window_start.advance(-lag - time_sampling['weather_rtma'], 'day'),
        window_start.advance(-lag, 'day')) \
        .select('WIND') \
        .mean() \
        .reproject(
            projection.atScale(ee_utils.RESAMPLING_SCALE[ee_utils.DataType.WEATHER_RTMA])
        ).resample('bicubic').rename('wind_avg')

    q75_wind = image_collections['weather_rtma'].filterDate(
        window_start.advance(-lag - time_sampling['weather_rtma'], 'day'),
        window_start.advance(-lag, 'day')) \
        .select('WIND') \
        .reduce(ee.Reducer.percentile([75])) \
        .reproject(
            projection.atScale(ee_utils.RESAMPLING_SCALE[ee_utils.DataType.WEATHER_RTMA])
        ).resample('bicubic').rename('wind_75')

    weighted_wind_dir = image_collections['weather_rtma'].filterDate(
        window_start.advance(-lag - time_sampling['weather_rtma'], 'day'),
        window_start.advance(-lag, 'day')) \
        .select(['UGRD', 'VGRD', 'WIND']) \
        .map(lambda img: img.select('UGRD').addBands(img.select('WIND'))) \
        .reduce(ee.Reducer.mean().splitWeights()) \
        .addBands(
            image_collections['weather_rtma'].filterDate(
                window_start.advance(-lag - time_sampling['weather_rtma'], 'day'),
                window_start.advance(-lag, 'day')) \
            .select(['UGRD', 'VGRD', 'WIND']) \
            .map(lambda img: img.select('VGRD').addBands(img.select('WIND'))) \
            .reduce(ee.Reducer.mean().splitWeights())
        ) \
        .expression('atan2(b("mean_1"), b("mean"))') \
        .select([0]) \
        .reproject(
            projection.atScale(ee_utils.RESAMPLING_SCALE[ee_utils.DataType.WEATHER_RTMA])
        ).resample('bicubic').rename('wdir_wind')

    weighted_gust_dir = image_collections['weather_rtma'].filterDate(
        window_start.advance(-lag - time_sampling['weather_rtma'], 'day'),
        window_start.advance(-lag, 'day')) \
        .select(['UGRD', 'VGRD', 'GUST']) \
        .map(lambda img: img.select('UGRD').addBands(img.select('GUST'))) \
        .reduce(ee.Reducer.mean().splitWeights()) \
        .addBands(
            image_collections['weather_rtma'].filterDate(
                window_start.advance(-lag - time_sampling['weather_rtma'], 'day'),
                window_start.advance(-lag, 'day')) \
            .select(['UGRD', 'VGRD', 'GUST']) \
            .map(lambda img: img.select('VGRD').addBands(img.select('GUST'))) \
            .reduce(ee.Reducer.mean().splitWeights())
        ) \
        .expression('atan2(b("mean_1"), b("mean"))') \
        .select([0]) \
        .reproject(
            projection.atScale(ee_utils.RESAMPLING_SCALE[ee_utils.DataType.WEATHER_RTMA])
        ).resample('bicubic').rename('wdir_gust')
    
    custom_weather_features = [
        avg_humidity,
        daytime_temp,
        q75_temp,
        median_gust,
        avg_wind,
        q75_wind,
        weighted_wind_dir,
        weighted_gust_dir
    ]

    # modis_prev_fire = image_collections['modis_fire'].filterDate(
    #   window_start.advance(-lag - time_sampling['modis_fire'], 'day'),
    #   window_start.advance(-lag, 'day')).map(
    #       ee_utils.remove_mask).max().rename('modis_PrevFireMask')

    viirs_prev_fire = image_collections['viirs_fire'].filterDate(
      window_start.advance(-lag - time_sampling['viirs_fire'], 'day'),
      window_start.advance(-lag, 'day')).map(
          ee_utils.remove_mask).max().rename('viirs_PrevFireMask')
    
    # modis_fire = image_collections['modis_fire'].filterDate(window_start, window_end).map(
    #   ee_utils.remove_mask).max().rename('modis_FireMask')

    viirs_fire = image_collections['viirs_fire'].filterDate(window_start,window_end).map(
      ee_utils.remove_mask).max().rename('viirs_FireMask')
    
    detection1 = viirs_prev_fire.rename('detection') # can use the MODIS data for detection, but have to subtract 6 from all measurements
    detection2 = viirs_fire.rename('detection')
    
    final_slices = [vegetation, drought, weather_gridmet] + custom_weather_features + [viirs_prev_fire, viirs_fire, detection1, detection2]
    
    return final_slices


def _export_dataset(
    bucket,
    folder,
    prefix,
    start_date,
    start_days,
    geometry,
    kernel_size,
    sampling_scale,
    num_samples_per_file,
    verbose = True,
):
    """Exports the dataset TFRecord files for wildfire risk assessment.
    
    Args:
    bucket: Google Cloud bucket
    folder: Folder to which to export the TFRecords.
    prefix: Export file name prefix.
    start_date: Start date for the EE data to export.
    start_days: Start day of each time chunk to export.
    geometry: EE geometry from which to export the data.
    kernel_size: Size of the exported tiles (square).
    sampling_scale: Resolution at which to export the data (in meters).
    num_samples_per_file: Approximate number of samples to save per TFRecord
      file.
    """
    
    def _verify_and_export_feature_collection(
      num_samples_per_export,
      feature_collection,
      file_count,
      features,
    ):
        """Wraps the verification and export of the feature collection.
        
        Verifies the size of the feature collection and triggers the export when
        it is larger than `num_samples_per_export`. Resets the feature collection
        and increments the file count at each export.
        
        Args:
          num_samples_per_export: Approximate number of samples per export.
          feature_collection: The EE feature collection to export.
          file_count: The TFRecord file count for naming the files.
          features: Names of the features to export.
        
        Returns:
          `(feature_collection, file_count)` tuple of the current feature collection
            and file count.
        """
        feature_collection, size_count = _verify_feature_collection(
            feature_collection)
        if size_count > num_samples_per_export:
            print("Exporting a feature collection of size",size_count) # debugging
            ee_utils.export_feature_collection(
              feature_collection,
              description=prefix + '_{:03d}'.format(file_count),
              bucket=bucket,
              folder=folder,
              bands=features,
            )
            file_count += 1
            feature_collection = ee.FeatureCollection([])
        return feature_collection, file_count

    # Get all feature names (inputs and responses)
    features = _get_all_feature_bands() + _get_all_response_bands()
    if verbose:
        print("Features to be extracted:")
        print(features)

    # Next, obtain the data which are not time-sliced, or not time-dependent
    
    elevation = ee_utils.get_image(ee_utils.DataType.ELEVATION_SRTM) # Elevation: only a single image
    
    chili = ee_utils.get_image(ee_utils.DataType.CHILI).rename('chili') # CHILI: only a single image
    
    impervious_coll = ee_utils.get_image_collection(ee_utils.DataType.IMPERVIOUS) # impervious: use 2016 data
    impervious = impervious_coll.filterDate('2016-01-01', '2017-01-01').first()
    
    water_cover = ee_utils.get_image(ee_utils.DataType.WTR_COVER).rename('water') # water cover: single image
    water_cover = water_cover.unmask() # unmasking is crucial for this dataset, or extract samples will return empty
    
    # population density
    population_coll = ee_utils.get_image_collection(ee_utils.DataType.POPULATION)
    population = population_coll.filterDate('2015-01-01',
                                     '2018-01-01').first().rename('population')

    # fuel types: manually choose the correct year
    curr_year = start_date.get('year').getInfo()
    if curr_year < 2019:
        fuel = ee.Image(ee_utils.DATA_SOURCES[ee_utils.DataType.LATENT_FUELS][0])
        landfire_fuel = ee.Image(ee_utils.DATA_SOURCES[ee_utils.DataType.LANDFIRE_FUELS][0])
    else:
        fuel = ee.Image(ee_utils.DATA_SOURCES[ee_utils.DataType.LATENT_FUELS][1])
        landfire_fuel = ee.Image(ee_utils.DATA_SOURCES[ee_utils.DataType.LANDFIRE_FUELS][1])

    # manually resample the landfire fuel map down to 500m resolution
    landfire_fuel = landfire_fuel.select('b1').rename('landfire_fuel')
    resampled_landfire = landfire_fuel.reduceResolution(
        reducer=ee.Reducer.mode(),
        maxPixels=16**2
    )
    landfire_fuel = resampled_landfire.reproject(
        ee.Projection('EPSG:4326').atScale(500)
    )
        
    fuel1 = fuel.select('b1').rename('fuel1')
    fuel2 = fuel.select('b2').rename('fuel2')
    fuel3 = fuel.select('b3').rename('fuel3')

    
    all_days = []
    for day in start_days:
        for i in range(7):
            all_days.append(day + i)
    
    window = 1
    sampling_limit_per_call = 30
    
    
    file_count = 0
    total_feature_collection = ee.FeatureCollection([])
    
    for start_day in all_days:
        window_start = start_date.advance(start_day, 'days')
        if verbose:
            print("Looking at date",window_start.format().getInfo())

        # aggregate the features as a list of images, with collections time-sliced
        time_slices = _get_time_slices(window_start)
        # print(len(time_slices)-2)

        image_list = [elevation, chili, impervious,water_cover,population,fuel1,fuel2,fuel3,landfire_fuel] 
        # print('before adding time slices',len(image_list))
        image_list += time_slices[:-2]
        # print('after adding time slices',len(image_list))
        # return image_list
        detection1,detection2 = time_slices[-2:]
        
        arrays = ee_utils.convert_features_to_arrays(image_list, kernel_size)
        to_sample = detection1.addBands(arrays)

        if verbose: print("Getting detection count....")
        fire_count_prev = ee_utils.get_detection_count(
            detection1,
            geometry=geometry,
            sampling_scale= 20 * sampling_scale, # sampling scale is 10km = 20*500m here, which defines the rule to discriminate 
                                                #        spatially separated/distinct fires
        )
        fire_count_post = ee_utils.get_detection_count(
            detection2,
            geometry=geometry,
            sampling_scale= 20 * sampling_scale, # sampling scale is 10km = 20*500m here, which defines the rule to discriminate 
                                                #        spatially separated/distinct fires
        )
        if verbose: print("Prev fire count found:",fire_count_prev)
        if verbose: print("Post fire count found:",fire_count_post)
        
        if fire_count_prev > 0 and fire_count_post > 0:
            if verbose: print("Extracting samples...")
  
            samples = ee_utils.extract_samples(
              to_sample,
              fire_count_prev,
              geometry, 
              sampling_limit_per_call=sampling_limit_per_call,
              resolution=sampling_scale,
              date = window_start,
            )
            # return samples
            
            total_feature_collection = total_feature_collection.merge(samples)

            # this is where the feature collection size error will be thrown
            total_feature_collection, file_count = _verify_and_export_feature_collection(
              num_samples_per_file, total_feature_collection, file_count, features)
            
    # Export the remaining feature collection
    _verify_and_export_feature_collection(0, total_feature_collection, file_count,
                                    features)
    

def export_ml_datasets(
    bucket,
    folder,
    start_date,
    end_date,
    prefix = '',
    kernel_size = 64,
    sampling_scale = 500,
    eval_split_ratio = 0.125,
    num_samples_per_file = 1000,
    coordinates = ee_utils.COORDINATES['OREGON'],
    fire_season_only = True,
    ):
    """Exports the ML dataset TFRecord files for wildfire risk assessment.
    
    Export is to Google Cloud Storage.
    
    Args:
    bucket: Google Cloud bucket
    folder: Folder to which to export the TFRecords.
    start_date: Start date for the EE data to export.
    end_date: End date for the EE data to export.
    prefix: File name prefix to use.
    kernel_size: Size of the exported tiles (square).
    sampling_scale: Resolution at which to export the data (in meters).
    eval_split_ratio: Split ratio for the divide between training and evaluation
      datasets.
    num_samples_per_file: Approximate number of samples to save per TFRecord
      file.
    """
    
    split_days = ee_utils.split_days_into_train_eval_test(
      start_date, end_date, split_ratio=eval_split_ratio, window_length_days=8,
        fire_season_only = fire_season_only
    )
    
    for mode in ['train', 'eval', 'test']:
        print("Beginning export process in mode",mode)
        sub_prefix = f'{mode}_{prefix}'
        _export_dataset(
            bucket=bucket,
            folder=folder,
            prefix=sub_prefix,
            start_date=start_date,
            start_days=split_days[mode],
            geometry=ee.Geometry.Rectangle(coordinates),
            kernel_size=kernel_size,
            sampling_scale=sampling_scale,
            num_samples_per_file=num_samples_per_file)