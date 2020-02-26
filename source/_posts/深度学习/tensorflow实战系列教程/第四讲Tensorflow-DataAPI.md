---
title: Tensorflow系列教程4-TF中的关于Data的API
categories: [TensorFlow系列教程]
tags: [Tensorflow]
typora-root-url: 第四讲Tensorflow-DataAPI
typora-copy-images-to: 第四讲Tensorflow-DataAPI
---

​    ![](/bg.jpg)   



<!--more-->

# Tensorflow系列教程4-TF中的关于Data的API

## Dataset基本API的使用

- tf.data.Dataset.from_tensor_slices

  from_tensor_slices支持元组数据、list数据、字典数据和numpy数据

  ```python
  # 支持numpy数据
  dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
  print(dataset)
  
  # 支持list数据
  dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4])
  for item in dataset:
      print(item)
      
  x = np.array([[1, 2], [3, 4], [5, 6]])
  y = np.array(['cat', 'dog', 'fox'])
  # 支持元组
  dataset3 = tf.data.Dataset.from_tensor_slices((x, y))
  for item_x, item_y in dataset3:
      print(item_x.numpy(), item_y.numpy())
  
  # 支持字典
  dataset4 = tf.data.Dataset.from_tensor_slices({"feature": x, "label": y})
  for item in dataset4:
      print(item['feature'].numpy(), item['label'].numpy())
  ```

- repeat/batch/interleacve/map/shuffle/list_files

  repeat/batch

  ```python
  dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
  dataset2 = dataset.repeat(3).batch(7)# 全部数据重复3次 每组数据取7个值
  for item in dataset2:
      print(item)
  ```

  ```python
  tf.Tensor([0 1 2 3 4 5 6], shape=(7,), dtype=int32)
  tf.Tensor([7 8 9 0 1 2 3], shape=(7,), dtype=int32)
  tf.Tensor([4 5 6 7 8 9 0], shape=(7,), dtype=int32)
  tf.Tensor([1 2 3 4 5 6 7], shape=(7,), dtype=int32)
  tf.Tensor([8 9], shape=(2,), dtype=int32)
  ```

  interleacve

  ```python
  dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
  dataset2 = dataset.repeat(3).batch(7)
  dataset3 = dataset2.interleave(
      # map_fun
      lambda v: tf.data.Dataset.from_tensor_slices(v),
      # cycle_length 并行程度
      cycle_length=5,
      # block_length
      block_length=5 # 对每一个epoch的数据优先取前5个
  )
  for item in dataset3:
      print(item)
  ```

  ```
  tf.Tensor(0, shape=(), dtype=int32)
  tf.Tensor(1, shape=(), dtype=int32)
  tf.Tensor(2, shape=(), dtype=int32)
  tf.Tensor(3, shape=(), dtype=int32)
  tf.Tensor(4, shape=(), dtype=int32)
  tf.Tensor(7, shape=(), dtype=int32)
  tf.Tensor(8, shape=(), dtype=int32)
  tf.Tensor(9, shape=(), dtype=int32)
  tf.Tensor(0, shape=(), dtype=int32)
  tf.Tensor(1, shape=(), dtype=int32)
  tf.Tensor(4, shape=(), dtype=int32)
  tf.Tensor(5, shape=(), dtype=int32)
  tf.Tensor(6, shape=(), dtype=int32)
  tf.Tensor(7, shape=(), dtype=int32)
  tf.Tensor(8, shape=(), dtype=int32)
  tf.Tensor(1, shape=(), dtype=int32)
  tf.Tensor(2, shape=(), dtype=int32)
  tf.Tensor(3, shape=(), dtype=int32)
  tf.Tensor(4, shape=(), dtype=int32)
  tf.Tensor(5, shape=(), dtype=int32)
  tf.Tensor(8, shape=(), dtype=int32)
  tf.Tensor(9, shape=(), dtype=int32)
  tf.Tensor(5, shape=(), dtype=int32)
  tf.Tensor(6, shape=(), dtype=int32)
  tf.Tensor(2, shape=(), dtype=int32)
  tf.Tensor(3, shape=(), dtype=int32)
  tf.Tensor(9, shape=(), dtype=int32)
  tf.Tensor(0, shape=(), dtype=int32)
  tf.Tensor(6, shape=(), dtype=int32)
  tf.Tensor(7, shape=(), dtype=int32)
  ```

## Dataset读取csv文件

- tf.data.TextLineDataset,tf.io.decode_csv

  首先生成csv文件

  ```python
  import matplotlib as mpl
  import matplotlib.pyplot as plt
  import numpy as np
  import sklearn
  import pandas as pd
  import os
  import sys
  import time
  import tensorflow as tf
  from tensorflow import keras
  
  print(tf.__version__)
  print(sys.version_info)
  for module in mpl, np, pd, sklearn, tf, keras:
      print(module.__name__, module.__version__)
  
  from sklearn.datasets import fetch_california_housing
  
  housing = fetch_california_housing()
  print(housing.DESCR)
  print(housing.data.shape)
  print(housing.target.shape)
  
  from sklearn.model_selection import train_test_split
  
  x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=7)
  x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=11)
  print(x_train.shape, y_train.shape)
  print(x_valid.shape, y_valid.shape)
  print(x_test.shape, y_test.shape)
  
  from sklearn.preprocessing import StandardScaler
  
  scaler = StandardScaler()
  x_train_scaled = scaler.fit_transform(x_train)
  x_valid_scaled = scaler.fit_transform(x_valid)
  x_test_scaled = scaler.fit_transform(x_test)
  
  output_dir = "generate_csv"
  if not os.path.exists(output_dir):
      os.mkdir(output_dir)
  
  
  def save_to_csv(output_dir, data, name_prefix,
                  header=None, n_parts=10):
      path_format = os.path.join(output_dir, "{}_{:02d}.csv")
      filenames = []
      for file_idx, row_indices in enumerate(np.array_split(np.arange(len(data)), n_parts)):
          part_csv = path_format.format(name_prefix, file_idx)
          filenames.append(part_csv)
          with open(part_csv, 'wt', encoding='utf-8') as f:
              if header is not None:
                  f.write(header + '\n')
              for row_index in row_indices:
                  f.write(",".join([repr(col) for col in data[row_index]]))
                  f.write('\n')
      return filenames
  
  
  train_data = np.c_[x_train_scaled, y_train]
  valid_data = np.c_[x_valid_scaled, y_valid]
  test_data = np.c_[x_test_scaled, y_test]
  
  header_cols = housing.feature_names + ['MidianHouseValue']
  header_str = ','.join(header_cols)
  
  train_filenames = save_to_csv(output_dir, train_data, "train", header_str, n_parts=20)
  valid_filenames = save_to_csv(output_dir, valid_data, "valid", header_str, n_parts=10)
  test_filenames = save_to_csv(output_dir, test_data, "test", header_str, n_parts=10)
  
  ```

  ```python
  # decode_csv解析字符串
  sample_str = "1,2,3,4,5"
  # 默认类型
  record_defaults = [
      tf.constant(0, dtype=tf.int32),
      0,
      np.nan,
      "hello",
      tf.constant([])
  
  ]
  parsed_fields = tf.io.decode_csv(sample_str, record_defaults=record_defaults)
  print(parsed_fields)
  
  # 字符串与record_defaults不匹配的两种情况会报错
  try:
      tf.io.decode_csv(",,,,,,", record_defaults=record_defaults)
  except tf.errors.InvalidArgumentError as e:
      print(e)
  
  try:
      tf.io.decode_csv("1,2,3,4,5,6,7", record_defaults=record_defaults)
  except tf.errors.InvalidArgumentError as e:
      print(e)
  ```

  读取csv文件并进行训练

  ```python
  n_readers = 4
  
  def parse_csv_line(line, n_fields=9): # 默认csv文件一行有9个字段
      defs = [tf.constant(np.nan)] * n_fields # 默认数据类型
      parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
      x = tf.stack(parsed_fields[0:-1])
      y = tf.stack(parsed_fields[-1:])
      return x, y
  
  
  def csv_reader_dataset(filenames, n_readers, batch_size=32, n_parse_threads=5, shuffle_buffer_size=1000):
      # 将所有文件转换为dataset
      dataset = tf.data.Dataset.list_files(filenames)
      # 扩展数据量
      dataset = dataset.repeat()
      # interleave 对每一个元素生成一个dataset
      dataset = dataset.interleave(
          lambda filename: tf.data.TextLineDataset(filename).skip(1),
          cycle_length=n_readers
      )
      dataset.shuffle(shuffle_buffer_size)  # 打乱顺序
      dataset = dataset.map(parse_csv_line, num_parallel_calls=n_parse_threads)
      dataset = dataset.batch(batch_size=batch_size)
      return dataset
  
  batch_size = 32
  train_set = csv_reader_dataset(train_filenames, n_readers=n_readers, batch_size=batch_size)
  valid_set = csv_reader_dataset(valid_filenames, n_readers=n_readers, batch_size=batch_size)
  test_set = csv_reader_dataset(test_filenames, n_readers=n_readers, batch_size=batch_size)
  
  # 定义model 进行fit
  model = keras.models.Sequential(
      [
          keras.layers.Dense(30, activation="relu", input_shape=[8]),
          keras.layers.Dense(1)
      ]
  )
  model.compile(loss="mse", optimizer="SGD")
  callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
  history = model.fit(train_set,
                      validation_data=valid_set,
                      steps_per_epoch=11160 // batch_size,
                      validation_steps=3870 // batch_size,
                      epochs=100,
                      callbacks=callbacks
                      )
  model.evaluate(test_set, steps=5160 // batch_size)
  
  ```

## Dataset读取tfrecord文件

- api的基本使用

    > tf.train.FloatList,tf.train.Int64List,tf.train.BytesList
    >
    > tf.train.Feature,tf.train.Features,tf.train.Example
    >
    > example.SerializeToString
>
    > tf.io.ParaseSingleExample
    >
    > tf.io.VarLenFeature,tf.io.FixedLenFeature
    >
    > tf.data.TFRecordDataset,tf.io.TFRecordOptions

    ```python
    # tfrecord是一个格式
    # -> tf.train.Example
#   -> tf.train.Feature -> {"key":tf.train.Feature}
    #      -> tf.train.Feature -> tf.train.ByteList?FloatList/Int64List
    
    # 创建一个字符串数组
favotite_books = [name.encode("utf-8") for name in ["meachine learing", "cc150"]]
    # 转为tf的BytesList类型
    favotite_books_bytelist = tf.train.BytesList(value=favotite_books)
    print(favotite_books_bytelist)
    
    # 创建一个浮点数数组并转为FloatList类型
    hours_floatlist = tf.train.FloatList(value=[15.5, 9.5])
    print(hours_floatlist)
    
    # 创建一个Int64List类型的数组并
age_int64list = tf.train.Int64List(value=[15])
    print(age_int64list)
    
    # 创建一个Features,其中里面包括多个feature
features = tf.train.Features(
        feature={
            "favorite_books": tf.train.Feature(bytes_list=favotite_books_bytelist),
            "hours": tf.train.Feature(float_list=hours_floatlist),
        "age": tf.train.Feature(int64_list=age_int64list)
        }
    )
    print(features)

    # 创建一个样本
    example = tf.train.Example(features=features)
    print(example)
    
    # 将样本数据序列化，节省空间
    serialized_example = example.SerializeToString()
    print(serialized_example)
    
    output_dir = "tfrecord_basic"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

# 正常数据持久化
    filename = "test.tfrecords"
    filename_fullpath = os.path.join(output_dir, filename)
    with tf.io.TFRecordWriter(filename_fullpath) as writer:
        for i in range(3):
        writer.write(serialized_example)
    # 数据压缩持久化
    filename_fullpath_zip = os.path.join(output_dir, filename) + ".zip"
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(filename_fullpath_zip, options) as writer:
        for i in range(3):
            writer.write(serialized_example)
    
    # 创建一个TFRecordDataset解析数据
dataset = tf.data.TFRecordDataset解析数据([filename_fullpath])
    # 创建一个TFRecordDataset解析压缩数据
    dataset_zip = tf.data.TFRecordDataset([filename_fullpath_zip], compression_type="GZIP")
    
    for serialized_example_tensor in dataset:
        print(serialized_example_tensor)
    # 要声明要解析的数据格式
expected_features = {
        "favorite_books": tf.io.VarLenFeature(dtype=tf.string),
        "hours": tf.io.VarLenFeature(dtype=tf.float32),
        "age": tf.io.FixedLenFeature([], dtype=tf.int64)
    }
    
# 解析tfrecord单个example的feature
    for serialized_example_tensor in dataset:
        example = tf.io.parse_single_example(serialized_example_tensor, expected_features)
        books = tf.sparse.to_dense(example['favorite_books'], default_value=b"")
        for book in books:
            print(book.numpy().decode("utf-8"))
    
    for serialized_example_tensor in dataset_zip:
        example = tf.io.parse_single_example(serialized_example_tensor, expected_features)
        books = tf.sparse.to_dense(example['favorite_books'], default_value=b"")
        for book in books:
            print(book.numpy().decode("utf-8"))
    
    ```

- 将现有的数据集转为tfrecord文件

  ```python
  
  source_dir = "./generate_csv"
  filenames = os.listdir(source_dir)
  
  
  def get_filename_by_prefix(source_dir, prefix_name):
      all_files = os.listdir(source_dir)
      results = []
      for filename in all_files:
          if filename.startswith(prefix_name):
              results.append(os.path.join(source_dir, filename))
      return results
  
  
  train_filenames = get_filename_by_prefix(source_dir, "train")
  valid_filenames = get_filename_by_prefix(source_dir, "valid")
  test_filenames = get_filename_by_prefix(source_dir, "test")
  pprint(train_filenames)
  pprint(valid_filenames)
  pprint(test_filenames)
  
  
  def parse_csv_line(line, n_fields=9):  # 默认csv文件一行有9个字段
      defs = [tf.constant(np.nan)] * n_fields  # 默认数据类型
      parsed_fields = tf.io.decode_csv(line, record_defaults=defs)
      x = tf.stack(parsed_fields[0:-1])
      y = tf.stack(parsed_fields[-1:])
      return x, y
  
  
  def csv_reader_dataset(filenames, n_readers, batch_size=32, n_parse_threads=5, shuffle_buffer_size=1000):
      # 将所有文件转换为dataset
      dataset = tf.data.Dataset.list_files(filenames)
      # 扩展数据量
      dataset = dataset.repeat()
      # interleave 对每一个元素生成一个dataset
      dataset = dataset.interleave(
          lambda filename: tf.data.TextLineDataset(filename).skip(1),
          cycle_length=n_readers
      )
      dataset.shuffle(shuffle_buffer_size)  # 打乱顺序
      dataset = dataset.map(parse_csv_line, num_parallel_calls=n_parse_threads)
      dataset = dataset.batch(batch_size=batch_size)
      return dataset
  
  
  batch_size = 32
  n_readers = 4
  train_set = csv_reader_dataset(train_filenames, n_readers=n_readers, batch_size=batch_size)
  valid_set = csv_reader_dataset(valid_filenames, n_readers=n_readers, batch_size=batch_size)
  test_set = csv_reader_dataset(test_filenames, n_readers=n_readers, batch_size=batch_size)
  
  
  def serialize_example(x, y):
      """Convert x,y to tf.train.Example and serialize"""
      input_features = tf.train.FloatList(value=x)
      label = tf.train.FloatList(value=y)
      features = tf.train.Features(
          feature={
              "input_features": tf.train.Feature(float_list=input_features),
              "label": tf.train.Feature(float_list=label)
          }
      )
      example = tf.train.Example(features=features)
      return example.SerializeToString()
  
  
  def csv_dataset_to_tfrecord(base_filename, dataset, n_shards, steps_per_shard, compression_type=None):
      options = tf.io.TFRecordOptions(compression_type=compression_type)
      all_filenames = []
      for shard_id in range(n_shards):
          filename_fullpath = "{}_{:05d}-of-{:05d}".format(base_filename, shard_id, n_shards)
          with tf.io.TFRecordWriter(filename_fullpath, options) as writer:
              for x_batch, y_batch in dataset.take(steps_per_shard):
                  for x_example, y_example in zip(x_batch, y_batch):
                      writer.write(serialize_example(x_example.numpy(), y_example.numpy()))
          all_filenames.append(filename_fullpath)
      return all_filenames
  
  
  n_shards = 20
  train_steps_per_shard = 11610 // batch_size // n_shards
  valid_steps_per_shard = 3880 // batch_size // n_shards
  test_steps_per_shard = 5170 // batch_size // n_shards
  
  output_dir = "generate_tfrecord"
  if not os.path.exists(output_dir):
      os.mkdir(output_dir)
  
  train_basename = os.path.join(output_dir, "train")
  valid_basename = os.path.join(output_dir, "valid")
  test_basename = os.path.join(output_dir, "test")
  
  train_tfrecord_filenames = csv_dataset_to_tfrecord(train_basename, train_set, n_shards, train_steps_per_shard, None)
  valid_tfrecord_filenames = csv_dataset_to_tfrecord(valid_basename, valid_set, n_shards, valid_steps_per_shard, None)
  test_tfrecord_filenames = csv_dataset_to_tfrecord(test_basename, test_set, n_shards, test_steps_per_shard, None)
  
  ```

- 使用tfrecord文件进行训练

  ```python
  expected_features = {
      "input_features": tf.io.FixedLenFeature([8], dtype=tf.float32),
      "label": tf.io.FixedLenFeature([1], dtype=tf.float32)
  }
  
  
  def parse_example(serialized_example):
      example = tf.io.parse_single_example(serialize_example, expected_features)
      return example['input_features'], example['label']
  
  
  def tfrecord_reader_dataset(filenames, n_readers, batch_size=32, n_parse_threads=5, shuffle_buffer_size=1000):
      dataset = tf.data.Dataset.list_files(filenames)
      dataset = dataset.repeat()
      # interleave 对每一个元素生成一个dataset
      dataset = dataset.interleave(
          lambda filename: tf.data.TFRecordDataset(filename, compression_type="GZIP"),
          cycle_length=n_readers
      )
      dataset.shuffle(shuffle_buffer_size)  # 打乱顺序
      dataset = dataset.map(parse_example, num_parallel_calls=n_parse_threads)
      dataset = dataset.batch(batch_size=batch_size)
      return dataset
  
  
  batch_size = 32
  n_readers = 5
  train_set = tfrecord_reader_dataset(train_tfrecord_filenames, n_readers=n_readers, batch_size=batch_size)
  valid_set = tfrecord_reader_dataset(valid_filenames, n_readers=n_readers, batch_size=batch_size)
  test_set = tfrecord_reader_dataset(test_filenames, n_readers=n_readers, batch_size=batch_size)
  
  
  # 定义model 进行fit
  model = keras.models.Sequential(
      [
          keras.layers.Dense(30, activation="relu", input_shape=[8]),
          keras.layers.Dense(1)
      ]
  )
  model.compile(loss="mse", optimizer="SGD")
  callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
  history = model.fit(train_set,
                      validation_data=valid_set,
                      steps_per_epoch=11160 // batch_size,
                      validation_steps=3870 // batch_size,
                      epochs=100,
                      callbacks=callbacks
                      )
  model.evaluate(test_set, steps=5160 // batch_size)
  ```

  