�	M�O�v�@M�O�v�@!M�O�v�@	ߗӺ�s�?ߗӺ�s�?!ߗӺ�s�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$M�O�v�@2�%��?A}гY�r�@Y�Ǻ���?*	����̜s@2F
Iterator::Model��ǘ���?!� W�l�T@)      �?1�N^���S@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��d�`T�?!�y7,�&@)����o�?1#����#@:Preprocessing2U
Iterator::Model::ParallelMapV2HP�sׂ?!�DUt@)HP�sׂ?1�DUt@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa��+e�?!�g#G̜@)	�^)ˀ?1�i�`��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�#�����?!�|�VMf1@)-C��6z?1)F��P @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��ZӼ�t?!���� �?)��ZӼ�t?1���� �?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�J�4q?!��7j�?)�J�4q?1��7j�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9ޗӺ�s�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	2�%��?2�%��?!2�%��?      ��!       "      ��!       *      ��!       2	}гY�r�@}гY�r�@!}гY�r�@:      ��!       B      ��!       J	�Ǻ���?�Ǻ���?!�Ǻ���?R      ��!       Z	�Ǻ���?�Ǻ���?!�Ǻ���?JCPU_ONLYYޗӺ�s�?b Y      Y@q���r��?"�
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