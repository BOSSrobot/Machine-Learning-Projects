ʈ
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??
z
layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namelayer_1/kernel
s
"layer_1/kernel/Read/ReadVariableOpReadVariableOplayer_1/kernel* 
_output_shapes
:
??*
dtype0
q
layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namelayer_1/bias
j
 layer_1/bias/Read/ReadVariableOpReadVariableOplayer_1/bias*
_output_shapes	
:?*
dtype0
z
layer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namelayer_2/kernel
s
"layer_2/kernel/Read/ReadVariableOpReadVariableOplayer_2/kernel* 
_output_shapes
:
??*
dtype0
q
layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namelayer_2/bias
j
 layer_2/bias/Read/ReadVariableOpReadVariableOplayer_2/bias*
_output_shapes	
:?*
dtype0
{
q_values/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_nameq_values/kernel
t
#q_values/kernel/Read/ReadVariableOpReadVariableOpq_values/kernel*
_output_shapes
:	?*
dtype0
r
q_values/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameq_values/bias
k
!q_values/bias/Read/ReadVariableOpReadVariableOpq_values/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
 
h


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
 
*

0
1
2
3
4
5
*

0
1
2
3
4
5
?
regularization_losses
layer_regularization_losses

layers
trainable_variables
metrics
non_trainable_variables
 layer_metrics
	variables
 
ZX
VARIABLE_VALUElayer_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElayer_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE


0
1
 


0
1
?
	variables
regularization_losses
!layer_regularization_losses

"layers
trainable_variables
#non_trainable_variables
$layer_metrics
%metrics
ZX
VARIABLE_VALUElayer_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElayer_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
regularization_losses
&layer_regularization_losses

'layers
trainable_variables
(non_trainable_variables
)layer_metrics
*metrics
[Y
VARIABLE_VALUEq_values/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEq_values/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
regularization_losses
+layer_regularization_losses

,layers
trainable_variables
-non_trainable_variables
.layer_metrics
/metrics
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
x
serving_default_inpPlaceholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inplayer_1/kernellayer_1/biaslayer_2/kernellayer_2/biasq_values/kernelq_values/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_77316696
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"layer_1/kernel/Read/ReadVariableOp layer_1/bias/Read/ReadVariableOp"layer_2/kernel/Read/ReadVariableOp layer_2/bias/Read/ReadVariableOp#q_values/kernel/Read/ReadVariableOp!q_values/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_save_77316881
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_1/kernellayer_1/biaslayer_2/kernellayer_2/biasq_values/kernelq_values/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__traced_restore_77316909??
?

*__inference_layer_2_layer_call_fn_77316820

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_layer_2_layer_call_and_return_conditional_losses_773165412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_77316696
inp
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinpunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_773164992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
(
_output_shapes
:??????????

_user_specified_nameinp
?
?
!__inference__traced_save_77316881
file_prefix-
)savev2_layer_1_kernel_read_readvariableop+
'savev2_layer_1_bias_read_readvariableop-
)savev2_layer_2_kernel_read_readvariableop+
'savev2_layer_2_bias_read_readvariableop.
*savev2_q_values_kernel_read_readvariableop,
(savev2_q_values_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_layer_1_kernel_read_readvariableop'savev2_layer_1_bias_read_readvariableop)savev2_layer_2_kernel_read_readvariableop'savev2_layer_2_bias_read_readvariableop*savev2_q_values_kernel_read_readvariableop(savev2_q_values_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*N
_input_shapes=
;: :
??:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: 
?
?
+__inference_model_14_layer_call_fn_77316780

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_14_layer_call_and_return_conditional_losses_773166622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_model_14_layer_call_and_return_conditional_losses_77316662

inputs
layer_1_77316646
layer_1_77316648
layer_2_77316651
layer_2_77316653
q_values_77316656
q_values_77316658
identity??layer_1/StatefulPartitionedCall?layer_2/StatefulPartitionedCall? q_values/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCallinputslayer_1_77316646layer_1_77316648*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_layer_1_layer_call_and_return_conditional_losses_773165142!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_77316651layer_2_77316653*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_layer_2_layer_call_and_return_conditional_losses_773165412!
layer_2/StatefulPartitionedCall?
 q_values/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0q_values_77316656q_values_77316658*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_q_values_layer_call_and_return_conditional_losses_773165682"
 q_values/StatefulPartitionedCall?
IdentityIdentity)q_values/StatefulPartitionedCall:output:0 ^layer_1/StatefulPartitionedCall ^layer_2/StatefulPartitionedCall!^q_values/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall2D
 q_values/StatefulPartitionedCall q_values/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_layer_2_layer_call_and_return_conditional_losses_77316541

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_q_values_layer_call_and_return_conditional_losses_77316568

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_model_14_layer_call_fn_77316641
inp
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinpunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_14_layer_call_and_return_conditional_losses_773166262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
(
_output_shapes
:??????????

_user_specified_nameinp
?
?
$__inference__traced_restore_77316909
file_prefix#
assignvariableop_layer_1_kernel#
assignvariableop_1_layer_1_bias%
!assignvariableop_2_layer_2_kernel#
assignvariableop_3_layer_2_bias&
"assignvariableop_4_q_values_kernel$
 assignvariableop_5_q_values_bias

identity_7??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_layer_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_layer_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_layer_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_q_values_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_q_values_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6?

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
F__inference_model_14_layer_call_and_return_conditional_losses_77316585
inp
layer_1_77316525
layer_1_77316527
layer_2_77316552
layer_2_77316554
q_values_77316579
q_values_77316581
identity??layer_1/StatefulPartitionedCall?layer_2/StatefulPartitionedCall? q_values/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCallinplayer_1_77316525layer_1_77316527*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_layer_1_layer_call_and_return_conditional_losses_773165142!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_77316552layer_2_77316554*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_layer_2_layer_call_and_return_conditional_losses_773165412!
layer_2/StatefulPartitionedCall?
 q_values/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0q_values_77316579q_values_77316581*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_q_values_layer_call_and_return_conditional_losses_773165682"
 q_values/StatefulPartitionedCall?
IdentityIdentity)q_values/StatefulPartitionedCall:output:0 ^layer_1/StatefulPartitionedCall ^layer_2/StatefulPartitionedCall!^q_values/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall2D
 q_values/StatefulPartitionedCall q_values/StatefulPartitionedCall:M I
(
_output_shapes
:??????????

_user_specified_nameinp
?
?
+__inference_model_14_layer_call_fn_77316763

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_14_layer_call_and_return_conditional_losses_773166262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_model_14_layer_call_and_return_conditional_losses_77316626

inputs
layer_1_77316610
layer_1_77316612
layer_2_77316615
layer_2_77316617
q_values_77316620
q_values_77316622
identity??layer_1/StatefulPartitionedCall?layer_2/StatefulPartitionedCall? q_values/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCallinputslayer_1_77316610layer_1_77316612*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_layer_1_layer_call_and_return_conditional_losses_773165142!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_77316615layer_2_77316617*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_layer_2_layer_call_and_return_conditional_losses_773165412!
layer_2/StatefulPartitionedCall?
 q_values/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0q_values_77316620q_values_77316622*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_q_values_layer_call_and_return_conditional_losses_773165682"
 q_values/StatefulPartitionedCall?
IdentityIdentity)q_values/StatefulPartitionedCall:output:0 ^layer_1/StatefulPartitionedCall ^layer_2/StatefulPartitionedCall!^q_values/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall2D
 q_values/StatefulPartitionedCall q_values/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_layer_1_layer_call_and_return_conditional_losses_77316514

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_model_14_layer_call_and_return_conditional_losses_77316604
inp
layer_1_77316588
layer_1_77316590
layer_2_77316593
layer_2_77316595
q_values_77316598
q_values_77316600
identity??layer_1/StatefulPartitionedCall?layer_2/StatefulPartitionedCall? q_values/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCallinplayer_1_77316588layer_1_77316590*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_layer_1_layer_call_and_return_conditional_losses_773165142!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_77316593layer_2_77316595*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_layer_2_layer_call_and_return_conditional_losses_773165412!
layer_2/StatefulPartitionedCall?
 q_values/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0q_values_77316598q_values_77316600*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_q_values_layer_call_and_return_conditional_losses_773165682"
 q_values/StatefulPartitionedCall?
IdentityIdentity)q_values/StatefulPartitionedCall:output:0 ^layer_1/StatefulPartitionedCall ^layer_2/StatefulPartitionedCall!^q_values/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall2D
 q_values/StatefulPartitionedCall q_values/StatefulPartitionedCall:M I
(
_output_shapes
:??????????

_user_specified_nameinp
?
?
+__inference_model_14_layer_call_fn_77316677
inp
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinpunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_14_layer_call_and_return_conditional_losses_773166622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
(
_output_shapes
:??????????

_user_specified_nameinp
?

*__inference_layer_1_layer_call_fn_77316800

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_layer_1_layer_call_and_return_conditional_losses_773165142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_model_14_layer_call_and_return_conditional_losses_77316721

inputs*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource*
&layer_2_matmul_readvariableop_resource+
'layer_2_biasadd_readvariableop_resource+
'q_values_matmul_readvariableop_resource,
(q_values_biasadd_readvariableop_resource
identity??layer_1/BiasAdd/ReadVariableOp?layer_1/MatMul/ReadVariableOp?layer_2/BiasAdd/ReadVariableOp?layer_2/MatMul/ReadVariableOp?q_values/BiasAdd/ReadVariableOp?q_values/MatMul/ReadVariableOp?
layer_1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
layer_1/MatMul/ReadVariableOp?
layer_1/MatMulMatMulinputs%layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_1/MatMul?
layer_1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_1/BiasAdd/ReadVariableOp?
layer_1/BiasAddBiasAddlayer_1/MatMul:product:0&layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_1/BiasAddq
layer_1/ReluRelulayer_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
layer_1/Relu?
layer_2/MatMul/ReadVariableOpReadVariableOp&layer_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
layer_2/MatMul/ReadVariableOp?
layer_2/MatMulMatMullayer_1/Relu:activations:0%layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_2/MatMul?
layer_2/BiasAdd/ReadVariableOpReadVariableOp'layer_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_2/BiasAdd/ReadVariableOp?
layer_2/BiasAddBiasAddlayer_2/MatMul:product:0&layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_2/BiasAddq
layer_2/ReluRelulayer_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
layer_2/Relu?
q_values/MatMul/ReadVariableOpReadVariableOp'q_values_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
q_values/MatMul/ReadVariableOp?
q_values/MatMulMatMullayer_2/Relu:activations:0&q_values/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
q_values/MatMul?
q_values/BiasAdd/ReadVariableOpReadVariableOp(q_values_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
q_values/BiasAdd/ReadVariableOp?
q_values/BiasAddBiasAddq_values/MatMul:product:0'q_values/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
q_values/BiasAdds
q_values/TanhTanhq_values/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
q_values/Tanh?
IdentityIdentityq_values/Tanh:y:0^layer_1/BiasAdd/ReadVariableOp^layer_1/MatMul/ReadVariableOp^layer_2/BiasAdd/ReadVariableOp^layer_2/MatMul/ReadVariableOp ^q_values/BiasAdd/ReadVariableOp^q_values/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::2@
layer_1/BiasAdd/ReadVariableOplayer_1/BiasAdd/ReadVariableOp2>
layer_1/MatMul/ReadVariableOplayer_1/MatMul/ReadVariableOp2@
layer_2/BiasAdd/ReadVariableOplayer_2/BiasAdd/ReadVariableOp2>
layer_2/MatMul/ReadVariableOplayer_2/MatMul/ReadVariableOp2B
q_values/BiasAdd/ReadVariableOpq_values/BiasAdd/ReadVariableOp2@
q_values/MatMul/ReadVariableOpq_values/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_model_14_layer_call_and_return_conditional_losses_77316746

inputs*
&layer_1_matmul_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource*
&layer_2_matmul_readvariableop_resource+
'layer_2_biasadd_readvariableop_resource+
'q_values_matmul_readvariableop_resource,
(q_values_biasadd_readvariableop_resource
identity??layer_1/BiasAdd/ReadVariableOp?layer_1/MatMul/ReadVariableOp?layer_2/BiasAdd/ReadVariableOp?layer_2/MatMul/ReadVariableOp?q_values/BiasAdd/ReadVariableOp?q_values/MatMul/ReadVariableOp?
layer_1/MatMul/ReadVariableOpReadVariableOp&layer_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
layer_1/MatMul/ReadVariableOp?
layer_1/MatMulMatMulinputs%layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_1/MatMul?
layer_1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_1/BiasAdd/ReadVariableOp?
layer_1/BiasAddBiasAddlayer_1/MatMul:product:0&layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_1/BiasAddq
layer_1/ReluRelulayer_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
layer_1/Relu?
layer_2/MatMul/ReadVariableOpReadVariableOp&layer_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
layer_2/MatMul/ReadVariableOp?
layer_2/MatMulMatMullayer_1/Relu:activations:0%layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_2/MatMul?
layer_2/BiasAdd/ReadVariableOpReadVariableOp'layer_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_2/BiasAdd/ReadVariableOp?
layer_2/BiasAddBiasAddlayer_2/MatMul:product:0&layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_2/BiasAddq
layer_2/ReluRelulayer_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
layer_2/Relu?
q_values/MatMul/ReadVariableOpReadVariableOp'q_values_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
q_values/MatMul/ReadVariableOp?
q_values/MatMulMatMullayer_2/Relu:activations:0&q_values/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
q_values/MatMul?
q_values/BiasAdd/ReadVariableOpReadVariableOp(q_values_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
q_values/BiasAdd/ReadVariableOp?
q_values/BiasAddBiasAddq_values/MatMul:product:0'q_values/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
q_values/BiasAdds
q_values/TanhTanhq_values/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
q_values/Tanh?
IdentityIdentityq_values/Tanh:y:0^layer_1/BiasAdd/ReadVariableOp^layer_1/MatMul/ReadVariableOp^layer_2/BiasAdd/ReadVariableOp^layer_2/MatMul/ReadVariableOp ^q_values/BiasAdd/ReadVariableOp^q_values/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::2@
layer_1/BiasAdd/ReadVariableOplayer_1/BiasAdd/ReadVariableOp2>
layer_1/MatMul/ReadVariableOplayer_1/MatMul/ReadVariableOp2@
layer_2/BiasAdd/ReadVariableOplayer_2/BiasAdd/ReadVariableOp2>
layer_2/MatMul/ReadVariableOplayer_2/MatMul/ReadVariableOp2B
q_values/BiasAdd/ReadVariableOpq_values/BiasAdd/ReadVariableOp2@
q_values/MatMul/ReadVariableOpq_values/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
E__inference_layer_2_layer_call_and_return_conditional_losses_77316811

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_q_values_layer_call_fn_77316840

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_q_values_layer_call_and_return_conditional_losses_773165682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?"
?
#__inference__wrapped_model_77316499
inp3
/model_14_layer_1_matmul_readvariableop_resource4
0model_14_layer_1_biasadd_readvariableop_resource3
/model_14_layer_2_matmul_readvariableop_resource4
0model_14_layer_2_biasadd_readvariableop_resource4
0model_14_q_values_matmul_readvariableop_resource5
1model_14_q_values_biasadd_readvariableop_resource
identity??'model_14/layer_1/BiasAdd/ReadVariableOp?&model_14/layer_1/MatMul/ReadVariableOp?'model_14/layer_2/BiasAdd/ReadVariableOp?&model_14/layer_2/MatMul/ReadVariableOp?(model_14/q_values/BiasAdd/ReadVariableOp?'model_14/q_values/MatMul/ReadVariableOp?
&model_14/layer_1/MatMul/ReadVariableOpReadVariableOp/model_14_layer_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&model_14/layer_1/MatMul/ReadVariableOp?
model_14/layer_1/MatMulMatMulinp.model_14/layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_14/layer_1/MatMul?
'model_14/layer_1/BiasAdd/ReadVariableOpReadVariableOp0model_14_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model_14/layer_1/BiasAdd/ReadVariableOp?
model_14/layer_1/BiasAddBiasAdd!model_14/layer_1/MatMul:product:0/model_14/layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_14/layer_1/BiasAdd?
model_14/layer_1/ReluRelu!model_14/layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_14/layer_1/Relu?
&model_14/layer_2/MatMul/ReadVariableOpReadVariableOp/model_14_layer_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&model_14/layer_2/MatMul/ReadVariableOp?
model_14/layer_2/MatMulMatMul#model_14/layer_1/Relu:activations:0.model_14/layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_14/layer_2/MatMul?
'model_14/layer_2/BiasAdd/ReadVariableOpReadVariableOp0model_14_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model_14/layer_2/BiasAdd/ReadVariableOp?
model_14/layer_2/BiasAddBiasAdd!model_14/layer_2/MatMul:product:0/model_14/layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_14/layer_2/BiasAdd?
model_14/layer_2/ReluRelu!model_14/layer_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_14/layer_2/Relu?
'model_14/q_values/MatMul/ReadVariableOpReadVariableOp0model_14_q_values_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'model_14/q_values/MatMul/ReadVariableOp?
model_14/q_values/MatMulMatMul#model_14/layer_2/Relu:activations:0/model_14/q_values/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_14/q_values/MatMul?
(model_14/q_values/BiasAdd/ReadVariableOpReadVariableOp1model_14_q_values_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_14/q_values/BiasAdd/ReadVariableOp?
model_14/q_values/BiasAddBiasAdd"model_14/q_values/MatMul:product:00model_14/q_values/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_14/q_values/BiasAdd?
model_14/q_values/TanhTanh"model_14/q_values/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_14/q_values/Tanh?
IdentityIdentitymodel_14/q_values/Tanh:y:0(^model_14/layer_1/BiasAdd/ReadVariableOp'^model_14/layer_1/MatMul/ReadVariableOp(^model_14/layer_2/BiasAdd/ReadVariableOp'^model_14/layer_2/MatMul/ReadVariableOp)^model_14/q_values/BiasAdd/ReadVariableOp(^model_14/q_values/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????::::::2R
'model_14/layer_1/BiasAdd/ReadVariableOp'model_14/layer_1/BiasAdd/ReadVariableOp2P
&model_14/layer_1/MatMul/ReadVariableOp&model_14/layer_1/MatMul/ReadVariableOp2R
'model_14/layer_2/BiasAdd/ReadVariableOp'model_14/layer_2/BiasAdd/ReadVariableOp2P
&model_14/layer_2/MatMul/ReadVariableOp&model_14/layer_2/MatMul/ReadVariableOp2T
(model_14/q_values/BiasAdd/ReadVariableOp(model_14/q_values/BiasAdd/ReadVariableOp2R
'model_14/q_values/MatMul/ReadVariableOp'model_14/q_values/MatMul/ReadVariableOp:M I
(
_output_shapes
:??????????

_user_specified_nameinp
?	
?
E__inference_layer_1_layer_call_and_return_conditional_losses_77316791

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_q_values_layer_call_and_return_conditional_losses_77316831

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
4
inp-
serving_default_inp:0??????????<
q_values0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?{
?#
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
0__call__
1_default_save_signature
*2&call_and_return_all_conditional_losses"? 
_tf_keras_network? {"class_name": "Functional", "name": "model_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_14", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 225]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "inp"}, "name": "inp", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "units": 281, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer_1", "inbound_nodes": [[["inp", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "units": 187, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer_2", "inbound_nodes": [[["layer_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "q_values", "trainable": true, "dtype": "float32", "units": 4, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "q_values", "inbound_nodes": [[["layer_2", 0, 0, {}]]]}], "input_layers": [["inp", 0, 0]], "output_layers": [["q_values", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 225]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 225]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_14", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 225]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "inp"}, "name": "inp", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "units": 281, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer_1", "inbound_nodes": [[["inp", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "units": 187, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer_2", "inbound_nodes": [[["layer_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "q_values", "trainable": true, "dtype": "float32", "units": 4, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "q_values", "inbound_nodes": [[["layer_2", 0, 0, {}]]]}], "input_layers": [["inp", 0, 0]], "output_layers": [["q_values", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "inp", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 225]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 225]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "inp"}}
?


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
3__call__
*4&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "units": 281, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 225}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 225]}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
5__call__
*6&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "units": 187, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 281}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 281]}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
7__call__
*8&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "q_values", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "q_values", "trainable": true, "dtype": "float32", "units": 4, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 187}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 187]}}
 "
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
?
regularization_losses
layer_regularization_losses

layers
trainable_variables
metrics
non_trainable_variables
 layer_metrics
	variables
0__call__
1_default_save_signature
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
,
9serving_default"
signature_map
": 
??2layer_1/kernel
:?2layer_1/bias
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
?
	variables
regularization_losses
!layer_regularization_losses

"layers
trainable_variables
#non_trainable_variables
$layer_metrics
%metrics
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
": 
??2layer_2/kernel
:?2layer_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
regularization_losses
&layer_regularization_losses

'layers
trainable_variables
(non_trainable_variables
)layer_metrics
*metrics
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
": 	?2q_values/kernel
:2q_values/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
regularization_losses
+layer_regularization_losses

,layers
trainable_variables
-non_trainable_variables
.layer_metrics
/metrics
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?2?
+__inference_model_14_layer_call_fn_77316641
+__inference_model_14_layer_call_fn_77316763
+__inference_model_14_layer_call_fn_77316677
+__inference_model_14_layer_call_fn_77316780?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference__wrapped_model_77316499?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *#? 
?
inp??????????
?2?
F__inference_model_14_layer_call_and_return_conditional_losses_77316721
F__inference_model_14_layer_call_and_return_conditional_losses_77316585
F__inference_model_14_layer_call_and_return_conditional_losses_77316746
F__inference_model_14_layer_call_and_return_conditional_losses_77316604?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_layer_1_layer_call_fn_77316800?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_layer_1_layer_call_and_return_conditional_losses_77316791?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_layer_2_layer_call_fn_77316820?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_layer_2_layer_call_and_return_conditional_losses_77316811?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_q_values_layer_call_fn_77316840?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_q_values_layer_call_and_return_conditional_losses_77316831?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_77316696inp"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_77316499l
-?*
#? 
?
inp??????????
? "3?0
.
q_values"?
q_values??????????
E__inference_layer_1_layer_call_and_return_conditional_losses_77316791^
0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_layer_1_layer_call_fn_77316800Q
0?-
&?#
!?
inputs??????????
? "????????????
E__inference_layer_2_layer_call_and_return_conditional_losses_77316811^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_layer_2_layer_call_fn_77316820Q0?-
&?#
!?
inputs??????????
? "????????????
F__inference_model_14_layer_call_and_return_conditional_losses_77316585f
5?2
+?(
?
inp??????????
p

 
? "%?"
?
0?????????
? ?
F__inference_model_14_layer_call_and_return_conditional_losses_77316604f
5?2
+?(
?
inp??????????
p 

 
? "%?"
?
0?????????
? ?
F__inference_model_14_layer_call_and_return_conditional_losses_77316721i
8?5
.?+
!?
inputs??????????
p

 
? "%?"
?
0?????????
? ?
F__inference_model_14_layer_call_and_return_conditional_losses_77316746i
8?5
.?+
!?
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
+__inference_model_14_layer_call_fn_77316641Y
5?2
+?(
?
inp??????????
p

 
? "???????????
+__inference_model_14_layer_call_fn_77316677Y
5?2
+?(
?
inp??????????
p 

 
? "???????????
+__inference_model_14_layer_call_fn_77316763\
8?5
.?+
!?
inputs??????????
p

 
? "???????????
+__inference_model_14_layer_call_fn_77316780\
8?5
.?+
!?
inputs??????????
p 

 
? "???????????
F__inference_q_values_layer_call_and_return_conditional_losses_77316831]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? 
+__inference_q_values_layer_call_fn_77316840P0?-
&?#
!?
inputs??????????
? "???????????
&__inference_signature_wrapper_77316696s
4?1
? 
*?'
%
inp?
inp??????????"3?0
.
q_values"?
q_values?????????