	V-r��@V-r��@!V-r��@	��=f�?��=f�?!��=f�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$V-r��@�(��?AR�����@Y鷯��?*	23333�x@2F
Iterator::ModelOjM��?!�LzF$�R@)E���JY�?1/<�"R@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�� �rh�?!M����4!@)��j+���?1��TB@@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapO��e�c�?!��j�o-@)���S㥛?1@���S@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateŏ1w-!�?!O��N��@)J+��?1��m�-�@:Preprocessing2U
Iterator::Model::ParallelMapV2U���N@�?!t�KA@)U���N@�?1t�KA@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��^�?!7��n9@)� �	�?1�2��,�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��0�*x?!0>\����?)��0�*x?10>\����?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��ZӼ�t?!]
:xԥ�?)��ZӼ�t?1]
:xԥ�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9��=f�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�(��?�(��?!�(��?      ��!       "      ��!       *      ��!       2	R�����@R�����@!R�����@:      ��!       B      ��!       J	鷯��?鷯��?!鷯��?R      ��!       Z	鷯��?鷯��?!鷯��?JCPU_ONLYY��=f�?b 