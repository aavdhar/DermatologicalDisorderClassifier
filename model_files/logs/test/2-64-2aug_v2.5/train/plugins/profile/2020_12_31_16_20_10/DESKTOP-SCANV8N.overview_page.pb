�	@�߾��@@�߾��@!@�߾��@	��(���?��(���?!��(���?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$@�߾��@�:pΈҾ?Aj�q��@Y��:M�?*	33333{w@2F
Iterator::Model��:M��?!�q6�q6R@)�!��u��?1���A�Q@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatTR'����?!spd2@)C��6�?1����0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�D���J�?!���.L@)/�$��?1�z:�[@:Preprocessing2U
Iterator::Model::ParallelMapV2{�G�z�?!;�h(K@){�G�z�?1;�h(K@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�ZӼ��?!m,��<@)��~j�t�?1x{��:@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�&1��?!�9&�9&;@)	�^)ˀ?1J/��u@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora��+ey?!}`X�og�?)a��+ey?1}`X�og�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceU���N@s?!�aO��?)U���N@s?1�aO��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9��(���?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�:pΈҾ?�:pΈҾ?!�:pΈҾ?      ��!       "      ��!       *      ��!       2	j�q��@j�q��@!j�q��@:      ��!       B      ��!       J	��:M�?��:M�?!��:M�?R      ��!       Z	��:M�?��:M�?!��:M�?JCPU_ONLYY��(���?b Y      Y@q�o�1�G�?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 