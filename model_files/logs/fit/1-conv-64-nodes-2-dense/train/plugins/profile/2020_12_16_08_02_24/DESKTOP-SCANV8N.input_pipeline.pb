	���w��@���w��@!���w��@	b5]o?��?b5]o?��?!b5]o?��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$���w��@�z�G��?A�&���@Y�}8gD�?*	33333{�@2F
Iterator::Model=�U����?!��¢�Q@);�O��n�?1����P@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�ZӼ��?!�r%ƑJ<@)f��a���?1r���s�;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���&�?!�J���?)�X�� �?1�#7(���?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��&S�?!>��tɀ?@)8��d�`�?1������?:Preprocessing2U
Iterator::Model::ParallelMapV2� �	��?!)�H�?)� �	��?1)�H�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��ǘ���?!�8��y�?)��ǘ���?1�8��y�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!?!��r��?)ŏ1w-!?1��r��?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���V�/�?!V�|�Ww<@)F%u�k?1�a��c�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9b5]o?��?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�z�G��?�z�G��?!�z�G��?      ��!       "      ��!       *      ��!       2	�&���@�&���@!�&���@:      ��!       B      ��!       J	�}8gD�?�}8gD�?!�}8gD�?R      ��!       Z	�}8gD�?�}8gD�?!�}8gD�?JCPU_ONLYYb5]o?��?b 