	��H�}�@��H�}�@!��H�}�@	��R��?��R��?!��R��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��H�}�@vOjM�?A-���x�@Y�E�����?*	     j�@2F
Iterator::Model�[ A��?!h���N@)@�߾��?16��M@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�ZӼ��?!�$I�$IB@)	�^)��?1~̣��B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���Q��?!@���P@)tF��_�?1�ԓ�ۥ�?:Preprocessing2U
Iterator::Model::ParallelMapV2��_�L�?!�}̣���?)��_�L�?1�}̣���?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipQk�w���?!��ic/�C@)���Q�~?1@���P�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora��+ey?!G�`���?)a��+ey?1G�`���?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice;�O��nr?!N
���-�?);�O��nr?1N
���-�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�46<�?!��aB@)a2U0*�c?1S`��n��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9��R��?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	vOjM�?vOjM�?!vOjM�?      ��!       "      ��!       *      ��!       2	-���x�@-���x�@!-���x�@:      ��!       B      ��!       J	�E�����?�E�����?!�E�����?R      ��!       Z	�E�����?�E�����?!�E�����?JCPU_ONLYY��R��?b 