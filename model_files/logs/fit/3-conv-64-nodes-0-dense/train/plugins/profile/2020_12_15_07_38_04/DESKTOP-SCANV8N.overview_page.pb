�	{�/L&'�@{�/L&'�@!{�/L&'�@	���	�?���	�?!���	�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails${�/L&'�@��6��?A�o_N�@Y<�R�!N@*	�����Q�@2F
Iterator::ModelX�5�;�@!1��t�J@)sh��|�@1��BHe�J@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�>W[���?!�
+�Q G@)6�>W[��?1ƒ=D��F@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�I+��?!$+�P��?)�5�;Nё?1=D�U��?:Preprocessing2U
Iterator::Model::ParallelMapV2��H�}}?!y8@�}�?)��H�}}?1y8@�}�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���~�:�?!��9�\G@)�HP�x?1�/�!���?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�r?!*H`.=0�?)HP�s�r?1*H`.=0�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�l?!�6�ڝr�?)y�&1�l?1�6�ڝr�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��@����?!�*�RrG@)����Mb`?1�>�����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9���	�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��6��?��6��?!��6��?      ��!       "      ��!       *      ��!       2	�o_N�@�o_N�@!�o_N�@:      ��!       B      ��!       J	<�R�!N@<�R�!N@!<�R�!N@R      ��!       Z	<�R�!N@<�R�!N@!<�R�!N@JCPU_ONLYY���	�?b Y      Y@qk7F����?"�
device�Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 