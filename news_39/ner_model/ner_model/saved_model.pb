╒─+
▀F╕F
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	АР
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
ю
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
Ы
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeintИ
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
s
	AssignSub
ref"TА

value"T

output_ref"TА" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

ControlTrigger
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

B
Equal
x"T
y"T
z
"
Ttype:
2	
Р
)
Exit	
data"T
output"T"	
Ttype
,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
8
FloorMod
x"T
y"T
z"T"
Ttype:	
2	
М
Gather
params"Tparams
indices"Tindices
output"Tparams"
validate_indicesbool("
Tparamstype"
Tindicestype:
2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
.
IsFinite
x"T
y
"
Ttype:
2
:
Less
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
$

LogicalAnd
x

y

z
Р
!
LoopCond	
input


output

p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
М
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	Р
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
М
Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Minimum
x"T
y"T
z"T"
Ttype:

2	Р
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
.
Neg
x"T
y"T"
Ttype:

2	
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	
Р
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
)
Rank

input"T

output"	
Ttype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
К
ReverseSequence

input"T
seq_lengths"Tlen
output"T"
seq_dimint"
	batch_dimint "	
Ttype"
Tlentype0	:
2	
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
е

ScatterAdd
ref"TА
indices"Tindices
updates"T

output_ref"TА" 
Ttype:
2	"
Tindicestype:
2	"
use_lockingbool( 
s
	ScatterNd
indices"Tindices
updates"T
shape"Tindices
output"T"	
Ttype"
Tindicestype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
-
Sqrt
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
A

StackPopV2

handle
elem"	elem_type"
	elem_typetypeИ
X
StackPushV2

handle	
elem"T
output"T"	
Ttype"
swap_memorybool( И
S
StackV2
max_size

handle"
	elem_typetype"

stack_namestring И
2
StopGradient

input"T
output"T"	
Ttype
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:И
`
TensorArrayGradV3

handle
flow_in
grad_handle
flow_out"
sourcestringИ
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetypeИ
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
TtypeИ
9
TensorArraySizeV3

handle
flow_in
sizeИ
▐
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring И
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
TtypeИ
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
А
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
┴
UnsortedSegmentSum	
data"T
segment_ids"Tindices
num_segments"Tnumsegments
output"T" 
Ttype:
2	"
Tindicestype:
2	" 
Tnumsegmentstype0:
2	
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.7.02v1.7.0-3-g024aecf414о°$
{
inputsPlaceholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
|
targetsPlaceholder*
dtype0*0
_output_shapes
:                  *%
shape:                  
N
	keep_probPlaceholder*
dtype0*
_output_shapes
:*
shape:
H
Cast/xConst*
value	B : *
dtype0*
_output_shapes
: 
_
NotEqualNotEqualCast/xinputs*0
_output_shapes
:                  *
T0
b
Cast_1CastNotEqual*

SrcT0
*0
_output_shapes
:                  *

DstT0
W
Sum/reduction_indicesConst*
_output_shapes
: *
value	B :*
dtype0
t
SumSumCast_1Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:         
╟
Aembedding_layer/embedding_matrix/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"Д  А   *3
_class)
'%loc:@embedding_layer/embedding_matrix
╣
?embedding_layer/embedding_matrix/Initializer/random_uniform/minConst*
valueB
 *нъ ╝*3
_class)
'%loc:@embedding_layer/embedding_matrix*
dtype0*
_output_shapes
: 
╣
?embedding_layer/embedding_matrix/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *нъ <*3
_class)
'%loc:@embedding_layer/embedding_matrix*
dtype0
г
Iembedding_layer/embedding_matrix/Initializer/random_uniform/RandomUniformRandomUniformAembedding_layer/embedding_matrix/Initializer/random_uniform/shape*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix*
seed2 *
dtype0* 
_output_shapes
:
Д/А*

seed 
Ю
?embedding_layer/embedding_matrix/Initializer/random_uniform/subSub?embedding_layer/embedding_matrix/Initializer/random_uniform/max?embedding_layer/embedding_matrix/Initializer/random_uniform/min*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix*
_output_shapes
: 
▓
?embedding_layer/embedding_matrix/Initializer/random_uniform/mulMulIembedding_layer/embedding_matrix/Initializer/random_uniform/RandomUniform?embedding_layer/embedding_matrix/Initializer/random_uniform/sub*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix* 
_output_shapes
:
Д/А
д
;embedding_layer/embedding_matrix/Initializer/random_uniformAdd?embedding_layer/embedding_matrix/Initializer/random_uniform/mul?embedding_layer/embedding_matrix/Initializer/random_uniform/min*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix* 
_output_shapes
:
Д/А
═
 embedding_layer/embedding_matrix
VariableV2*
shared_name *3
_class)
'%loc:@embedding_layer/embedding_matrix*
	container *
shape:
Д/А*
dtype0* 
_output_shapes
:
Д/А
Щ
'embedding_layer/embedding_matrix/AssignAssign embedding_layer/embedding_matrix;embedding_layer/embedding_matrix/Initializer/random_uniform*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix*
validate_shape(* 
_output_shapes
:
Д/А*
use_locking(
│
%embedding_layer/embedding_matrix/readIdentity embedding_layer/embedding_matrix*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix* 
_output_shapes
:
Д/А
№
 embedding_layer/embedding_lookupGather%embedding_layer/embedding_matrix/readinputs*5
_output_shapes#
!:                  А*
Tindices0*
Tparams0*
validate_indices(*3
_class)
'%loc:@embedding_layer/embedding_matrix
l
*biLSTM_layers/bidirectional_rnn/fw/fw/RankConst*
value	B :*
dtype0*
_output_shapes
: 
s
1biLSTM_layers/bidirectional_rnn/fw/fw/range/startConst*
_output_shapes
: *
value	B :*
dtype0
s
1biLSTM_layers/bidirectional_rnn/fw/fw/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ю
+biLSTM_layers/bidirectional_rnn/fw/fw/rangeRange1biLSTM_layers/bidirectional_rnn/fw/fw/range/start*biLSTM_layers/bidirectional_rnn/fw/fw/Rank1biLSTM_layers/bidirectional_rnn/fw/fw/range/delta*
_output_shapes
:*

Tidx0
Ж
5biLSTM_layers/bidirectional_rnn/fw/fw/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
s
1biLSTM_layers/bidirectional_rnn/fw/fw/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Й
,biLSTM_layers/bidirectional_rnn/fw/fw/concatConcatV25biLSTM_layers/bidirectional_rnn/fw/fw/concat/values_0+biLSTM_layers/bidirectional_rnn/fw/fw/range1biLSTM_layers/bidirectional_rnn/fw/fw/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
┘
/biLSTM_layers/bidirectional_rnn/fw/fw/transpose	Transpose embedding_layer/embedding_lookup,biLSTM_layers/bidirectional_rnn/fw/fw/concat*5
_output_shapes#
!:                  А*
Tperm0*
T0
t
5biLSTM_layers/bidirectional_rnn/fw/fw/sequence_lengthIdentitySum*
T0*#
_output_shapes
:         
Ъ
+biLSTM_layers/bidirectional_rnn/fw/fw/ShapeShape/biLSTM_layers/bidirectional_rnn/fw/fw/transpose*
T0*
out_type0*
_output_shapes
:
Г
9biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
Е
;biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Е
;biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╖
3biLSTM_layers/bidirectional_rnn/fw/fw/strided_sliceStridedSlice+biLSTM_layers/bidirectional_rnn/fw/fw/Shape9biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice/stack;biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice/stack_1;biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
И
FbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
■
BbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/ExpandDims
ExpandDims3biLSTM_layers/bidirectional_rnn/fw/fw/strided_sliceFbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
И
=biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/ConstConst*
dtype0*
_output_shapes
:*
valueB:А
Е
CbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╠
>biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/concatConcatV2BbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/ExpandDims=biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/ConstCbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
И
CbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
П
=biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/zerosFill>biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/concatCbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/zeros/Const*
T0*

index_type0*(
_output_shapes
:         А
К
HbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
В
DbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/ExpandDims_1
ExpandDims3biLSTM_layers/bidirectional_rnn/fw/fw/strided_sliceHbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/ExpandDims_1/dim*
T0*
_output_shapes
:*

Tdim0
К
?biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/Const_1Const*
valueB:А*
dtype0*
_output_shapes
:
К
HbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
В
DbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/ExpandDims_2
ExpandDims3biLSTM_layers/bidirectional_rnn/fw/fw/strided_sliceHbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/ExpandDims_2/dim*
_output_shapes
:*

Tdim0*
T0
К
?biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/Const_2Const*
valueB:А*
dtype0*
_output_shapes
:
З
EbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╘
@biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/concat_1ConcatV2DbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/ExpandDims_2?biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/Const_2EbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
К
EbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Х
?biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/zeros_1Fill@biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/concat_1EbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/zeros_1/Const*
T0*

index_type0*(
_output_shapes
:         А
К
HbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 
В
DbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/ExpandDims_3
ExpandDims3biLSTM_layers/bidirectional_rnn/fw/fw/strided_sliceHbiLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/ExpandDims_3/dim*

Tdim0*
T0*
_output_shapes
:
К
?biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/Const_3Const*
_output_shapes
:*
valueB:А*
dtype0
в
-biLSTM_layers/bidirectional_rnn/fw/fw/Shape_1Shape5biLSTM_layers/bidirectional_rnn/fw/fw/sequence_length*
T0*
out_type0*
_output_shapes
:
в
+biLSTM_layers/bidirectional_rnn/fw/fw/stackPack3biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice*
_output_shapes
:*
T0*

axis *
N
╡
+biLSTM_layers/bidirectional_rnn/fw/fw/EqualEqual-biLSTM_layers/bidirectional_rnn/fw/fw/Shape_1+biLSTM_layers/bidirectional_rnn/fw/fw/stack*
_output_shapes
:*
T0
u
+biLSTM_layers/bidirectional_rnn/fw/fw/ConstConst*
valueB: *
dtype0*
_output_shapes
:
┐
)biLSTM_layers/bidirectional_rnn/fw/fw/AllAll+biLSTM_layers/bidirectional_rnn/fw/fw/Equal+biLSTM_layers/bidirectional_rnn/fw/fw/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
╚
2biLSTM_layers/bidirectional_rnn/fw/fw/Assert/ConstConst*f
value]B[ BUExpected shape for Tensor biLSTM_layers/bidirectional_rnn/fw/fw/sequence_length:0 is *
dtype0*
_output_shapes
: 
Е
4biLSTM_layers/bidirectional_rnn/fw/fw/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
╨
:biLSTM_layers/bidirectional_rnn/fw/fw/Assert/Assert/data_0Const*f
value]B[ BUExpected shape for Tensor biLSTM_layers/bidirectional_rnn/fw/fw/sequence_length:0 is *
dtype0*
_output_shapes
: 
Л
:biLSTM_layers/bidirectional_rnn/fw/fw/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
▄
3biLSTM_layers/bidirectional_rnn/fw/fw/Assert/AssertAssert)biLSTM_layers/bidirectional_rnn/fw/fw/All:biLSTM_layers/bidirectional_rnn/fw/fw/Assert/Assert/data_0+biLSTM_layers/bidirectional_rnn/fw/fw/stack:biLSTM_layers/bidirectional_rnn/fw/fw/Assert/Assert/data_2-biLSTM_layers/bidirectional_rnn/fw/fw/Shape_1*
T
2*
	summarize
╪
1biLSTM_layers/bidirectional_rnn/fw/fw/CheckSeqLenIdentity5biLSTM_layers/bidirectional_rnn/fw/fw/sequence_length4^biLSTM_layers/bidirectional_rnn/fw/fw/Assert/Assert*
T0*#
_output_shapes
:         
Ь
-biLSTM_layers/bidirectional_rnn/fw/fw/Shape_2Shape/biLSTM_layers/bidirectional_rnn/fw/fw/transpose*
_output_shapes
:*
T0*
out_type0
Е
;biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
З
=biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
З
=biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
┴
5biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_1StridedSlice-biLSTM_layers/bidirectional_rnn/fw/fw/Shape_2;biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_1/stack=biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_1/stack_1=biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
Ь
-biLSTM_layers/bidirectional_rnn/fw/fw/Shape_3Shape/biLSTM_layers/bidirectional_rnn/fw/fw/transpose*
_output_shapes
:*
T0*
out_type0
Е
;biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_2/stackConst*
_output_shapes
:*
valueB:*
dtype0
З
=biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
З
=biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_2/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
┴
5biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_2StridedSlice-biLSTM_layers/bidirectional_rnn/fw/fw/Shape_3;biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_2/stack=biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_2/stack_1=biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
v
4biLSTM_layers/bidirectional_rnn/fw/fw/ExpandDims/dimConst*
_output_shapes
: *
value	B : *
dtype0
▄
0biLSTM_layers/bidirectional_rnn/fw/fw/ExpandDims
ExpandDims5biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_24biLSTM_layers/bidirectional_rnn/fw/fw/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
x
-biLSTM_layers/bidirectional_rnn/fw/fw/Const_1Const*
valueB:А*
dtype0*
_output_shapes
:
u
3biLSTM_layers/bidirectional_rnn/fw/fw/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
К
.biLSTM_layers/bidirectional_rnn/fw/fw/concat_1ConcatV20biLSTM_layers/bidirectional_rnn/fw/fw/ExpandDims-biLSTM_layers/bidirectional_rnn/fw/fw/Const_13biLSTM_layers/bidirectional_rnn/fw/fw/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
v
1biLSTM_layers/bidirectional_rnn/fw/fw/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
█
+biLSTM_layers/bidirectional_rnn/fw/fw/zerosFill.biLSTM_layers/bidirectional_rnn/fw/fw/concat_11biLSTM_layers/bidirectional_rnn/fw/fw/zeros/Const*
T0*

index_type0*(
_output_shapes
:         А
w
-biLSTM_layers/bidirectional_rnn/fw/fw/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
╨
)biLSTM_layers/bidirectional_rnn/fw/fw/MinMin1biLSTM_layers/bidirectional_rnn/fw/fw/CheckSeqLen-biLSTM_layers/bidirectional_rnn/fw/fw/Const_2*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
w
-biLSTM_layers/bidirectional_rnn/fw/fw/Const_3Const*
dtype0*
_output_shapes
:*
valueB: 
╨
)biLSTM_layers/bidirectional_rnn/fw/fw/MaxMax1biLSTM_layers/bidirectional_rnn/fw/fw/CheckSeqLen-biLSTM_layers/bidirectional_rnn/fw/fw/Const_3*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
l
*biLSTM_layers/bidirectional_rnn/fw/fw/timeConst*
value	B : *
dtype0*
_output_shapes
: 
ъ
1biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayTensorArrayV35biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_1*Q
tensor_array_name<:biLSTM_layers/bidirectional_rnn/fw/fw/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *%
element_shape:         А*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
ы
3biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray_1TensorArrayV35biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_1*%
element_shape:         А*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*P
tensor_array_name;9biLSTM_layers/bidirectional_rnn/fw/fw/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: 
н
>biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/ShapeShape/biLSTM_layers/bidirectional_rnn/fw/fw/transpose*
T0*
out_type0*
_output_shapes
:
Ц
LbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
Ш
NbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ш
NbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Ц
FbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_sliceStridedSlice>biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/ShapeLbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stackNbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1NbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
Ж
DbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
Ж
DbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/deltaConst*
_output_shapes
: *
value	B :*
dtype0
╠
>biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/rangeRangeDbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/startFbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_sliceDbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/delta*#
_output_shapes
:         *

Tidx0
║
`biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV33biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray_1>biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/biLSTM_layers/bidirectional_rnn/fw/fw/transpose5biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray_1:1*
_output_shapes
: *
T0*B
_class8
64loc:@biLSTM_layers/bidirectional_rnn/fw/fw/transpose
q
/biLSTM_layers/bidirectional_rnn/fw/fw/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
╡
-biLSTM_layers/bidirectional_rnn/fw/fw/MaximumMaximum/biLSTM_layers/bidirectional_rnn/fw/fw/Maximum/x)biLSTM_layers/bidirectional_rnn/fw/fw/Max*
T0*
_output_shapes
: 
┐
-biLSTM_layers/bidirectional_rnn/fw/fw/MinimumMinimum5biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_1-biLSTM_layers/bidirectional_rnn/fw/fw/Maximum*
T0*
_output_shapes
: 

=biLSTM_layers/bidirectional_rnn/fw/fw/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
У
1biLSTM_layers/bidirectional_rnn/fw/fw/while/EnterEnter=biLSTM_layers/bidirectional_rnn/fw/fw/while/iteration_counter*
is_constant( *
parallel_iterations *
_output_shapes
: *I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0
В
3biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_1Enter*biLSTM_layers/bidirectional_rnn/fw/fw/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
Л
3biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_2Enter3biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray:1*
is_constant( *
parallel_iterations *
_output_shapes
: *I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0
з
3biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_3Enter=biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:         А*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
й
3biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_4Enter?biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:         А*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
╘
1biLSTM_layers/bidirectional_rnn/fw/fw/while/MergeMerge1biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter9biLSTM_layers/bidirectional_rnn/fw/fw/while/NextIteration*
_output_shapes
: : *
T0*
N
┌
3biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_1Merge3biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_1;biLSTM_layers/bidirectional_rnn/fw/fw/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
┌
3biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_2Merge3biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_2;biLSTM_layers/bidirectional_rnn/fw/fw/while/NextIteration_2*
N*
_output_shapes
: : *
T0
ь
3biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_3Merge3biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_3;biLSTM_layers/bidirectional_rnn/fw/fw/while/NextIteration_3*
T0*
N**
_output_shapes
:         А: 
ь
3biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_4Merge3biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_4;biLSTM_layers/bidirectional_rnn/fw/fw/while/NextIteration_4*
T0*
N**
_output_shapes
:         А: 
─
0biLSTM_layers/bidirectional_rnn/fw/fw/while/LessLess1biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge6biLSTM_layers/bidirectional_rnn/fw/fw/while/Less/Enter*
_output_shapes
: *
T0
Р
6biLSTM_layers/bidirectional_rnn/fw/fw/while/Less/EnterEnter5biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_1*
parallel_iterations *
_output_shapes
: *I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(
╩
2biLSTM_layers/bidirectional_rnn/fw/fw/while/Less_1Less3biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_18biLSTM_layers/bidirectional_rnn/fw/fw/while/Less_1/Enter*
T0*
_output_shapes
: 
К
8biLSTM_layers/bidirectional_rnn/fw/fw/while/Less_1/EnterEnter-biLSTM_layers/bidirectional_rnn/fw/fw/Minimum*
parallel_iterations *
_output_shapes
: *I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(
┬
6biLSTM_layers/bidirectional_rnn/fw/fw/while/LogicalAnd
LogicalAnd0biLSTM_layers/bidirectional_rnn/fw/fw/while/Less2biLSTM_layers/bidirectional_rnn/fw/fw/while/Less_1*
_output_shapes
: 
Р
4biLSTM_layers/bidirectional_rnn/fw/fw/while/LoopCondLoopCond6biLSTM_layers/bidirectional_rnn/fw/fw/while/LogicalAnd*
_output_shapes
: 
О
2biLSTM_layers/bidirectional_rnn/fw/fw/while/SwitchSwitch1biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge4biLSTM_layers/bidirectional_rnn/fw/fw/while/LoopCond*
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge*
_output_shapes
: : 
Ф
4biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_1Switch3biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_14biLSTM_layers/bidirectional_rnn/fw/fw/while/LoopCond*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_1*
_output_shapes
: : 
Ф
4biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_2Switch3biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_24biLSTM_layers/bidirectional_rnn/fw/fw/while/LoopCond*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_2*
_output_shapes
: : 
╕
4biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_3Switch3biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_34biLSTM_layers/bidirectional_rnn/fw/fw/while/LoopCond*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_3*<
_output_shapes*
(:         А:         А*
T0
╕
4biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_4Switch3biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_44biLSTM_layers/bidirectional_rnn/fw/fw/while/LoopCond*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_4*<
_output_shapes*
(:         А:         А
Ч
4biLSTM_layers/bidirectional_rnn/fw/fw/while/IdentityIdentity4biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch:1*
T0*
_output_shapes
: 
Ы
6biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_1Identity6biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_1:1*
T0*
_output_shapes
: 
Ы
6biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_2Identity6biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_2:1*
T0*
_output_shapes
: 
н
6biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_3Identity6biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_3:1*(
_output_shapes
:         А*
T0
н
6biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_4Identity6biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_4:1*
T0*(
_output_shapes
:         А
к
1biLSTM_layers/bidirectional_rnn/fw/fw/while/add/yConst5^biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
└
/biLSTM_layers/bidirectional_rnn/fw/fw/while/addAdd4biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity1biLSTM_layers/bidirectional_rnn/fw/fw/while/add/y*
T0*
_output_shapes
: 
═
=biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3TensorArrayReadV3CbiLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter6biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_1EbiLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1*
dtype0*(
_output_shapes
:         А
Я
CbiLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/EnterEnter3biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
╩
EbiLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1Enter`biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
parallel_iterations *
_output_shapes
: *I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0
ю
8biLSTM_layers/bidirectional_rnn/fw/fw/while/GreaterEqualGreaterEqual6biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_1>biLSTM_layers/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter*
T0*#
_output_shapes
:         
б
>biLSTM_layers/bidirectional_rnn/fw/fw/while/GreaterEqual/EnterEnter1biLSTM_layers/bidirectional_rnn/fw/fw/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *#
_output_shapes
:         *I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
э
TbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"      *F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel*
dtype0*
_output_shapes
:
▀
RbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *є╡╜*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel*
dtype0*
_output_shapes
: 
▀
RbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *є╡=*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel*
dtype0
▄
\biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformTbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
АА*

seed *
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel*
seed2 
ъ
RbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/subSubRbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/maxRbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel*
_output_shapes
: 
■
RbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/mulMul\biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
АА*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel
Ё
NbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniformAddRbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/mulRbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform/min*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel* 
_output_shapes
:
АА*
T0
є
3biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel
VariableV2*
shared_name *F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА
х
:biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/AssignAssign3biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernelNbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА
д
8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/readIdentity3biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel*
T0* 
_output_shapes
:
АА
ф
SbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zeros/shape_as_tensorConst*
valueB:А*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
dtype0*
_output_shapes
:
╘
IbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zeros/ConstConst*
valueB
 *    *D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
dtype0*
_output_shapes
: 
щ
CbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zerosFillSbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zeros/shape_as_tensorIbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zeros/Const*
T0*

index_type0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
_output_shapes	
:А
х
1biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
	container *
shape:А
╧
8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/AssignAssign1biLSTM_layers/bidirectional_rnn/fw/lstm_cell/biasCbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zeros*
_output_shapes	
:А*
use_locking(*
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
validate_shape(
Ы
6biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/readIdentity1biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
T0*
_output_shapes	
:А
║
AbiLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat/axisConst5^biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
╩
<biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concatConcatV2=biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV36biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_4AbiLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat/axis*
T0*
N*(
_output_shapes
:         А*

Tidx0
б
<biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMulMatMul<biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concatBbiLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter*
T0*(
_output_shapes
:         А*
transpose_a( *
transpose_b( 
й
BbiLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/EnterEnter8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/read* 
_output_shapes
:
АА*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(*
parallel_iterations 
Х
=biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAddBiasAdd<biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMulCbiLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter*
data_formatNHWC*(
_output_shapes
:         А*
T0
г
CbiLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/EnterEnter6biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:А*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
┤
;biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/ConstConst5^biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
╛
EbiLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/split/split_dimConst5^biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
╩
;biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/splitSplitEbiLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/split/split_dim=biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split*
T0
╖
;biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add/yConst5^biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity*
valueB
 *  А?*
dtype0*
_output_shapes
: 
я
9biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/addAdd=biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/split:2;biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add/y*(
_output_shapes
:         А*
T0
╢
=biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/SigmoidSigmoid9biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add*
T0*(
_output_shapes
:         А
ъ
9biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mulMul=biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid6biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_3*
T0*(
_output_shapes
:         А
║
?biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1Sigmoid;biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/split*(
_output_shapes
:         А*
T0
┤
:biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/TanhTanh=biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/split:1*(
_output_shapes
:         А*
T0
Є
;biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1Mul?biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1:biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh*
T0*(
_output_shapes
:         А
э
;biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1Add9biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul;biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1*(
_output_shapes
:         А*
T0
╝
?biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2Sigmoid=biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/split:3*(
_output_shapes
:         А*
T0
┤
<biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1Tanh;biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1*
T0*(
_output_shapes
:         А
Ї
;biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2Mul?biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2<biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1*
T0*(
_output_shapes
:         А
Ё
2biLSTM_layers/bidirectional_rnn/fw/fw/while/SelectSelect8biLSTM_layers/bidirectional_rnn/fw/fw/while/GreaterEqual8biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter;biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*(
_output_shapes
:         А*
T0*N
_classD
B@loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2
ъ
8biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/EnterEnter+biLSTM_layers/bidirectional_rnn/fw/fw/zeros*
T0*N
_classD
B@loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
parallel_iterations *
is_constant(*(
_output_shapes
:         А*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
Ё
4biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1Select8biLSTM_layers/bidirectional_rnn/fw/fw/while/GreaterEqual6biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_3;biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1*(
_output_shapes
:         А*
T0*N
_classD
B@loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1
Ё
4biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2Select8biLSTM_layers/bidirectional_rnn/fw/fw/while/GreaterEqual6biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_4;biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
T0*N
_classD
B@loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*(
_output_shapes
:         А
╤
ObiLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3UbiLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter6biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_12biLSTM_layers/bidirectional_rnn/fw/fw/while/Select6biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_2*
T0*N
_classD
B@loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
_output_shapes
: 
 
UbiLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter1biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray*
is_constant(*
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*N
_classD
B@loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
parallel_iterations 
м
3biLSTM_layers/bidirectional_rnn/fw/fw/while/add_1/yConst5^biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
╞
1biLSTM_layers/bidirectional_rnn/fw/fw/while/add_1Add6biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_13biLSTM_layers/bidirectional_rnn/fw/fw/while/add_1/y*
T0*
_output_shapes
: 
Ь
9biLSTM_layers/bidirectional_rnn/fw/fw/while/NextIterationNextIteration/biLSTM_layers/bidirectional_rnn/fw/fw/while/add*
T0*
_output_shapes
: 
а
;biLSTM_layers/bidirectional_rnn/fw/fw/while/NextIteration_1NextIteration1biLSTM_layers/bidirectional_rnn/fw/fw/while/add_1*
T0*
_output_shapes
: 
╛
;biLSTM_layers/bidirectional_rnn/fw/fw/while/NextIteration_2NextIterationObiLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
╡
;biLSTM_layers/bidirectional_rnn/fw/fw/while/NextIteration_3NextIteration4biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1*
T0*(
_output_shapes
:         А
╡
;biLSTM_layers/bidirectional_rnn/fw/fw/while/NextIteration_4NextIteration4biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2*
T0*(
_output_shapes
:         А
Н
0biLSTM_layers/bidirectional_rnn/fw/fw/while/ExitExit2biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch*
T0*
_output_shapes
: 
С
2biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_1Exit4biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_1*
T0*
_output_shapes
: 
С
2biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_2Exit4biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_2*
T0*
_output_shapes
: 
г
2biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_3Exit4biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_3*
T0*(
_output_shapes
:         А
г
2biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_4Exit4biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_4*
T0*(
_output_shapes
:         А
в
HbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV31biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray2biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_2*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray*
_output_shapes
: 
╩
BbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayStack/range/startConst*
value	B : *D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray*
dtype0*
_output_shapes
: 
╩
BbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayStack/range/deltaConst*
value	B :*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray*
dtype0*
_output_shapes
: 
О
<biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayStack/rangeRangeBbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayStack/range/startHbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV3BbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayStack/range/delta*

Tidx0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray*#
_output_shapes
:         
╖
JbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV31biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray<biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayStack/range2biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_2*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray*
dtype0*5
_output_shapes#
!:                  А*%
element_shape:         А
x
-biLSTM_layers/bidirectional_rnn/fw/fw/Const_4Const*
valueB:А*
dtype0*
_output_shapes
:
n
,biLSTM_layers/bidirectional_rnn/fw/fw/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
u
3biLSTM_layers/bidirectional_rnn/fw/fw/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
u
3biLSTM_layers/bidirectional_rnn/fw/fw/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ў
-biLSTM_layers/bidirectional_rnn/fw/fw/range_1Range3biLSTM_layers/bidirectional_rnn/fw/fw/range_1/start,biLSTM_layers/bidirectional_rnn/fw/fw/Rank_13biLSTM_layers/bidirectional_rnn/fw/fw/range_1/delta*

Tidx0*
_output_shapes
:
И
7biLSTM_layers/bidirectional_rnn/fw/fw/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
u
3biLSTM_layers/bidirectional_rnn/fw/fw/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
С
.biLSTM_layers/bidirectional_rnn/fw/fw/concat_2ConcatV27biLSTM_layers/bidirectional_rnn/fw/fw/concat_2/values_0-biLSTM_layers/bidirectional_rnn/fw/fw/range_13biLSTM_layers/bidirectional_rnn/fw/fw/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0
З
1biLSTM_layers/bidirectional_rnn/fw/fw/transpose_1	TransposeJbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3.biLSTM_layers/bidirectional_rnn/fw/fw/concat_2*
T0*5
_output_shapes#
!:                  А*
Tperm0
╪
2biLSTM_layers/bidirectional_rnn/bw/ReverseSequenceReverseSequence embedding_layer/embedding_lookupSum*
T0*
seq_dim*5
_output_shapes#
!:                  А*

Tlen0*
	batch_dim 
l
*biLSTM_layers/bidirectional_rnn/bw/bw/RankConst*
value	B :*
dtype0*
_output_shapes
: 
s
1biLSTM_layers/bidirectional_rnn/bw/bw/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
s
1biLSTM_layers/bidirectional_rnn/bw/bw/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ю
+biLSTM_layers/bidirectional_rnn/bw/bw/rangeRange1biLSTM_layers/bidirectional_rnn/bw/bw/range/start*biLSTM_layers/bidirectional_rnn/bw/bw/Rank1biLSTM_layers/bidirectional_rnn/bw/bw/range/delta*
_output_shapes
:*

Tidx0
Ж
5biLSTM_layers/bidirectional_rnn/bw/bw/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
s
1biLSTM_layers/bidirectional_rnn/bw/bw/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Й
,biLSTM_layers/bidirectional_rnn/bw/bw/concatConcatV25biLSTM_layers/bidirectional_rnn/bw/bw/concat/values_0+biLSTM_layers/bidirectional_rnn/bw/bw/range1biLSTM_layers/bidirectional_rnn/bw/bw/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
ы
/biLSTM_layers/bidirectional_rnn/bw/bw/transpose	Transpose2biLSTM_layers/bidirectional_rnn/bw/ReverseSequence,biLSTM_layers/bidirectional_rnn/bw/bw/concat*
Tperm0*
T0*5
_output_shapes#
!:                  А
t
5biLSTM_layers/bidirectional_rnn/bw/bw/sequence_lengthIdentitySum*
T0*#
_output_shapes
:         
Ъ
+biLSTM_layers/bidirectional_rnn/bw/bw/ShapeShape/biLSTM_layers/bidirectional_rnn/bw/bw/transpose*
T0*
out_type0*
_output_shapes
:
Г
9biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
Е
;biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Е
;biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╖
3biLSTM_layers/bidirectional_rnn/bw/bw/strided_sliceStridedSlice+biLSTM_layers/bidirectional_rnn/bw/bw/Shape9biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice/stack;biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice/stack_1;biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
И
FbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
■
BbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/ExpandDims
ExpandDims3biLSTM_layers/bidirectional_rnn/bw/bw/strided_sliceFbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
И
=biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/ConstConst*
valueB:А*
dtype0*
_output_shapes
:
Е
CbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╠
>biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/concatConcatV2BbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/ExpandDims=biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/ConstCbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
И
CbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
П
=biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/zerosFill>biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/concatCbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/zeros/Const*
T0*

index_type0*(
_output_shapes
:         А
К
HbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
В
DbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/ExpandDims_1
ExpandDims3biLSTM_layers/bidirectional_rnn/bw/bw/strided_sliceHbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
К
?biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/Const_1Const*
dtype0*
_output_shapes
:*
valueB:А
К
HbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
В
DbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/ExpandDims_2
ExpandDims3biLSTM_layers/bidirectional_rnn/bw/bw/strided_sliceHbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/ExpandDims_2/dim*

Tdim0*
T0*
_output_shapes
:
К
?biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/Const_2Const*
valueB:А*
dtype0*
_output_shapes
:
З
EbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╘
@biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/concat_1ConcatV2DbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/ExpandDims_2?biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/Const_2EbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
К
EbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Х
?biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/zeros_1Fill@biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/concat_1EbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/zeros_1/Const*
T0*

index_type0*(
_output_shapes
:         А
К
HbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/ExpandDims_3/dimConst*
dtype0*
_output_shapes
: *
value	B : 
В
DbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/ExpandDims_3
ExpandDims3biLSTM_layers/bidirectional_rnn/bw/bw/strided_sliceHbiLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/ExpandDims_3/dim*

Tdim0*
T0*
_output_shapes
:
К
?biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/Const_3Const*
_output_shapes
:*
valueB:А*
dtype0
в
-biLSTM_layers/bidirectional_rnn/bw/bw/Shape_1Shape5biLSTM_layers/bidirectional_rnn/bw/bw/sequence_length*
T0*
out_type0*
_output_shapes
:
в
+biLSTM_layers/bidirectional_rnn/bw/bw/stackPack3biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice*
T0*

axis *
N*
_output_shapes
:
╡
+biLSTM_layers/bidirectional_rnn/bw/bw/EqualEqual-biLSTM_layers/bidirectional_rnn/bw/bw/Shape_1+biLSTM_layers/bidirectional_rnn/bw/bw/stack*
_output_shapes
:*
T0
u
+biLSTM_layers/bidirectional_rnn/bw/bw/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
┐
)biLSTM_layers/bidirectional_rnn/bw/bw/AllAll+biLSTM_layers/bidirectional_rnn/bw/bw/Equal+biLSTM_layers/bidirectional_rnn/bw/bw/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
╚
2biLSTM_layers/bidirectional_rnn/bw/bw/Assert/ConstConst*f
value]B[ BUExpected shape for Tensor biLSTM_layers/bidirectional_rnn/bw/bw/sequence_length:0 is *
dtype0*
_output_shapes
: 
Е
4biLSTM_layers/bidirectional_rnn/bw/bw/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
╨
:biLSTM_layers/bidirectional_rnn/bw/bw/Assert/Assert/data_0Const*
_output_shapes
: *f
value]B[ BUExpected shape for Tensor biLSTM_layers/bidirectional_rnn/bw/bw/sequence_length:0 is *
dtype0
Л
:biLSTM_layers/bidirectional_rnn/bw/bw/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
▄
3biLSTM_layers/bidirectional_rnn/bw/bw/Assert/AssertAssert)biLSTM_layers/bidirectional_rnn/bw/bw/All:biLSTM_layers/bidirectional_rnn/bw/bw/Assert/Assert/data_0+biLSTM_layers/bidirectional_rnn/bw/bw/stack:biLSTM_layers/bidirectional_rnn/bw/bw/Assert/Assert/data_2-biLSTM_layers/bidirectional_rnn/bw/bw/Shape_1*
T
2*
	summarize
╪
1biLSTM_layers/bidirectional_rnn/bw/bw/CheckSeqLenIdentity5biLSTM_layers/bidirectional_rnn/bw/bw/sequence_length4^biLSTM_layers/bidirectional_rnn/bw/bw/Assert/Assert*
T0*#
_output_shapes
:         
Ь
-biLSTM_layers/bidirectional_rnn/bw/bw/Shape_2Shape/biLSTM_layers/bidirectional_rnn/bw/bw/transpose*
T0*
out_type0*
_output_shapes
:
Е
;biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
З
=biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
З
=biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
┴
5biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_1StridedSlice-biLSTM_layers/bidirectional_rnn/bw/bw/Shape_2;biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_1/stack=biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_1/stack_1=biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
Ь
-biLSTM_layers/bidirectional_rnn/bw/bw/Shape_3Shape/biLSTM_layers/bidirectional_rnn/bw/bw/transpose*
T0*
out_type0*
_output_shapes
:
Е
;biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
З
=biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
З
=biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_2/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
┴
5biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_2StridedSlice-biLSTM_layers/bidirectional_rnn/bw/bw/Shape_3;biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_2/stack=biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_2/stack_1=biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
v
4biLSTM_layers/bidirectional_rnn/bw/bw/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
▄
0biLSTM_layers/bidirectional_rnn/bw/bw/ExpandDims
ExpandDims5biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_24biLSTM_layers/bidirectional_rnn/bw/bw/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
x
-biLSTM_layers/bidirectional_rnn/bw/bw/Const_1Const*
valueB:А*
dtype0*
_output_shapes
:
u
3biLSTM_layers/bidirectional_rnn/bw/bw/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
К
.biLSTM_layers/bidirectional_rnn/bw/bw/concat_1ConcatV20biLSTM_layers/bidirectional_rnn/bw/bw/ExpandDims-biLSTM_layers/bidirectional_rnn/bw/bw/Const_13biLSTM_layers/bidirectional_rnn/bw/bw/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
v
1biLSTM_layers/bidirectional_rnn/bw/bw/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
█
+biLSTM_layers/bidirectional_rnn/bw/bw/zerosFill.biLSTM_layers/bidirectional_rnn/bw/bw/concat_11biLSTM_layers/bidirectional_rnn/bw/bw/zeros/Const*(
_output_shapes
:         А*
T0*

index_type0
w
-biLSTM_layers/bidirectional_rnn/bw/bw/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
╨
)biLSTM_layers/bidirectional_rnn/bw/bw/MinMin1biLSTM_layers/bidirectional_rnn/bw/bw/CheckSeqLen-biLSTM_layers/bidirectional_rnn/bw/bw/Const_2*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
w
-biLSTM_layers/bidirectional_rnn/bw/bw/Const_3Const*
valueB: *
dtype0*
_output_shapes
:
╨
)biLSTM_layers/bidirectional_rnn/bw/bw/MaxMax1biLSTM_layers/bidirectional_rnn/bw/bw/CheckSeqLen-biLSTM_layers/bidirectional_rnn/bw/bw/Const_3*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
l
*biLSTM_layers/bidirectional_rnn/bw/bw/timeConst*
value	B : *
dtype0*
_output_shapes
: 
ъ
1biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayTensorArrayV35biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_1*Q
tensor_array_name<:biLSTM_layers/bidirectional_rnn/bw/bw/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *%
element_shape:         А*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
ы
3biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray_1TensorArrayV35biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_1*P
tensor_array_name;9biLSTM_layers/bidirectional_rnn/bw/bw/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *%
element_shape:         А*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
н
>biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/ShapeShape/biLSTM_layers/bidirectional_rnn/bw/bw/transpose*
_output_shapes
:*
T0*
out_type0
Ц
LbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
Ш
NbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ш
NbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Ц
FbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_sliceStridedSlice>biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/ShapeLbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stackNbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1NbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
Ж
DbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
Ж
DbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
╠
>biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/rangeRangeDbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/startFbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_sliceDbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/delta*#
_output_shapes
:         *

Tidx0
║
`biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV33biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray_1>biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/biLSTM_layers/bidirectional_rnn/bw/bw/transpose5biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray_1:1*
T0*B
_class8
64loc:@biLSTM_layers/bidirectional_rnn/bw/bw/transpose*
_output_shapes
: 
q
/biLSTM_layers/bidirectional_rnn/bw/bw/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
╡
-biLSTM_layers/bidirectional_rnn/bw/bw/MaximumMaximum/biLSTM_layers/bidirectional_rnn/bw/bw/Maximum/x)biLSTM_layers/bidirectional_rnn/bw/bw/Max*
T0*
_output_shapes
: 
┐
-biLSTM_layers/bidirectional_rnn/bw/bw/MinimumMinimum5biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_1-biLSTM_layers/bidirectional_rnn/bw/bw/Maximum*
T0*
_output_shapes
: 

=biLSTM_layers/bidirectional_rnn/bw/bw/while/iteration_counterConst*
dtype0*
_output_shapes
: *
value	B : 
У
1biLSTM_layers/bidirectional_rnn/bw/bw/while/EnterEnter=biLSTM_layers/bidirectional_rnn/bw/bw/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
В
3biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_1Enter*biLSTM_layers/bidirectional_rnn/bw/bw/time*
parallel_iterations *
_output_shapes
: *I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant( 
Л
3biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_2Enter3biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray:1*
parallel_iterations *
_output_shapes
: *I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant( 
з
3biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_3Enter=biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:         А*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
й
3biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_4Enter?biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/zeros_1*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:         А*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
╘
1biLSTM_layers/bidirectional_rnn/bw/bw/while/MergeMerge1biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter9biLSTM_layers/bidirectional_rnn/bw/bw/while/NextIteration*
_output_shapes
: : *
T0*
N
┌
3biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_1Merge3biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_1;biLSTM_layers/bidirectional_rnn/bw/bw/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
┌
3biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_2Merge3biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_2;biLSTM_layers/bidirectional_rnn/bw/bw/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
ь
3biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_3Merge3biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_3;biLSTM_layers/bidirectional_rnn/bw/bw/while/NextIteration_3*
T0*
N**
_output_shapes
:         А: 
ь
3biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_4Merge3biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_4;biLSTM_layers/bidirectional_rnn/bw/bw/while/NextIteration_4*
T0*
N**
_output_shapes
:         А: 
─
0biLSTM_layers/bidirectional_rnn/bw/bw/while/LessLess1biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge6biLSTM_layers/bidirectional_rnn/bw/bw/while/Less/Enter*
T0*
_output_shapes
: 
Р
6biLSTM_layers/bidirectional_rnn/bw/bw/while/Less/EnterEnter5biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
╩
2biLSTM_layers/bidirectional_rnn/bw/bw/while/Less_1Less3biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_18biLSTM_layers/bidirectional_rnn/bw/bw/while/Less_1/Enter*
T0*
_output_shapes
: 
К
8biLSTM_layers/bidirectional_rnn/bw/bw/while/Less_1/EnterEnter-biLSTM_layers/bidirectional_rnn/bw/bw/Minimum*
is_constant(*
parallel_iterations *
_output_shapes
: *I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0
┬
6biLSTM_layers/bidirectional_rnn/bw/bw/while/LogicalAnd
LogicalAnd0biLSTM_layers/bidirectional_rnn/bw/bw/while/Less2biLSTM_layers/bidirectional_rnn/bw/bw/while/Less_1*
_output_shapes
: 
Р
4biLSTM_layers/bidirectional_rnn/bw/bw/while/LoopCondLoopCond6biLSTM_layers/bidirectional_rnn/bw/bw/while/LogicalAnd*
_output_shapes
: 
О
2biLSTM_layers/bidirectional_rnn/bw/bw/while/SwitchSwitch1biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge4biLSTM_layers/bidirectional_rnn/bw/bw/while/LoopCond*
_output_shapes
: : *
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge
Ф
4biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_1Switch3biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_14biLSTM_layers/bidirectional_rnn/bw/bw/while/LoopCond*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_1*
_output_shapes
: : 
Ф
4biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_2Switch3biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_24biLSTM_layers/bidirectional_rnn/bw/bw/while/LoopCond*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_2*
_output_shapes
: : 
╕
4biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_3Switch3biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_34biLSTM_layers/bidirectional_rnn/bw/bw/while/LoopCond*<
_output_shapes*
(:         А:         А*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_3
╕
4biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_4Switch3biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_44biLSTM_layers/bidirectional_rnn/bw/bw/while/LoopCond*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_4*<
_output_shapes*
(:         А:         А*
T0
Ч
4biLSTM_layers/bidirectional_rnn/bw/bw/while/IdentityIdentity4biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch:1*
T0*
_output_shapes
: 
Ы
6biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_1Identity6biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_1:1*
_output_shapes
: *
T0
Ы
6biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_2Identity6biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_2:1*
T0*
_output_shapes
: 
н
6biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_3Identity6biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_3:1*
T0*(
_output_shapes
:         А
н
6biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_4Identity6biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_4:1*
T0*(
_output_shapes
:         А
к
1biLSTM_layers/bidirectional_rnn/bw/bw/while/add/yConst5^biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
└
/biLSTM_layers/bidirectional_rnn/bw/bw/while/addAdd4biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity1biLSTM_layers/bidirectional_rnn/bw/bw/while/add/y*
T0*
_output_shapes
: 
═
=biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3TensorArrayReadV3CbiLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter6biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_1EbiLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1*
dtype0*(
_output_shapes
:         А
Я
CbiLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/EnterEnter3biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
╩
EbiLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1Enter`biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
ю
8biLSTM_layers/bidirectional_rnn/bw/bw/while/GreaterEqualGreaterEqual6biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_1>biLSTM_layers/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter*#
_output_shapes
:         *
T0
б
>biLSTM_layers/bidirectional_rnn/bw/bw/while/GreaterEqual/EnterEnter1biLSTM_layers/bidirectional_rnn/bw/bw/CheckSeqLen*
parallel_iterations *#
_output_shapes
:         *I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant(
э
TbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel
▀
RbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *є╡╜*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel*
dtype0*
_output_shapes
: 
▀
RbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *є╡=*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel*
dtype0*
_output_shapes
: 
▄
\biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformTbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
АА*

seed *
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel*
seed2 
ъ
RbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/subSubRbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/maxRbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/min*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel*
_output_shapes
: 
■
RbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/mulMul\biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/sub*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel* 
_output_shapes
:
АА
Ё
NbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniformAddRbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/mulRbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform/min* 
_output_shapes
:
АА*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel
є
3biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel
VariableV2* 
_output_shapes
:
АА*
shared_name *F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel*
	container *
shape:
АА*
dtype0
х
:biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/AssignAssign3biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernelNbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
д
8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/readIdentity3biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel* 
_output_shapes
:
АА*
T0
ф
SbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zeros/shape_as_tensorConst*
valueB:А*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
dtype0*
_output_shapes
:
╘
IbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zeros/ConstConst*
valueB
 *    *D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
dtype0*
_output_shapes
: 
щ
CbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zerosFillSbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zeros/shape_as_tensorIbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zeros/Const*

index_type0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
_output_shapes	
:А*
T0
х
1biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
	container *
shape:А
╧
8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/AssignAssign1biLSTM_layers/bidirectional_rnn/bw/lstm_cell/biasCbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zeros*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
Ы
6biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/readIdentity1biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
T0*
_output_shapes	
:А
║
AbiLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat/axisConst5^biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
╩
<biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concatConcatV2=biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV36biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_4AbiLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat/axis*
N*(
_output_shapes
:         А*

Tidx0*
T0
б
<biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMulMatMul<biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concatBbiLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter*
T0*(
_output_shapes
:         А*
transpose_a( *
transpose_b( 
й
BbiLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/EnterEnter8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
АА*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
Х
=biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAddBiasAdd<biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMulCbiLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter*
data_formatNHWC*(
_output_shapes
:         А*
T0
г
CbiLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/EnterEnter6biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:А*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
┤
;biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/ConstConst5^biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
╛
EbiLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/split/split_dimConst5^biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
╩
;biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/splitSplitEbiLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/split/split_dim=biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd*d
_output_shapesR
P:         А:         А:         А:         А*
	num_split*
T0
╖
;biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add/yConst5^biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity*
valueB
 *  А?*
dtype0*
_output_shapes
: 
я
9biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/addAdd=biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/split:2;biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add/y*
T0*(
_output_shapes
:         А
╢
=biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/SigmoidSigmoid9biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add*(
_output_shapes
:         А*
T0
ъ
9biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mulMul=biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid6biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_3*(
_output_shapes
:         А*
T0
║
?biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1Sigmoid;biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/split*(
_output_shapes
:         А*
T0
┤
:biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/TanhTanh=biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/split:1*(
_output_shapes
:         А*
T0
Є
;biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1Mul?biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1:biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh*(
_output_shapes
:         А*
T0
э
;biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1Add9biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul;biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1*
T0*(
_output_shapes
:         А
╝
?biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2Sigmoid=biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/split:3*(
_output_shapes
:         А*
T0
┤
<biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1Tanh;biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1*(
_output_shapes
:         А*
T0
Ї
;biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2Mul?biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2<biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1*
T0*(
_output_shapes
:         А
Ё
2biLSTM_layers/bidirectional_rnn/bw/bw/while/SelectSelect8biLSTM_layers/bidirectional_rnn/bw/bw/while/GreaterEqual8biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter;biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
T0*N
_classD
B@loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*(
_output_shapes
:         А
ъ
8biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/EnterEnter+biLSTM_layers/bidirectional_rnn/bw/bw/zeros*
T0*N
_classD
B@loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
parallel_iterations *
is_constant(*(
_output_shapes
:         А*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
Ё
4biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1Select8biLSTM_layers/bidirectional_rnn/bw/bw/while/GreaterEqual6biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_3;biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1*(
_output_shapes
:         А*
T0*N
_classD
B@loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1
Ё
4biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2Select8biLSTM_layers/bidirectional_rnn/bw/bw/while/GreaterEqual6biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_4;biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*(
_output_shapes
:         А*
T0*N
_classD
B@loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2
╤
ObiLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3UbiLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter6biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_12biLSTM_layers/bidirectional_rnn/bw/bw/while/Select6biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_2*
_output_shapes
: *
T0*N
_classD
B@loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2
 
UbiLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter1biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray*
T0*N
_classD
B@loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
м
3biLSTM_layers/bidirectional_rnn/bw/bw/while/add_1/yConst5^biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity*
_output_shapes
: *
value	B :*
dtype0
╞
1biLSTM_layers/bidirectional_rnn/bw/bw/while/add_1Add6biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_13biLSTM_layers/bidirectional_rnn/bw/bw/while/add_1/y*
_output_shapes
: *
T0
Ь
9biLSTM_layers/bidirectional_rnn/bw/bw/while/NextIterationNextIteration/biLSTM_layers/bidirectional_rnn/bw/bw/while/add*
T0*
_output_shapes
: 
а
;biLSTM_layers/bidirectional_rnn/bw/bw/while/NextIteration_1NextIteration1biLSTM_layers/bidirectional_rnn/bw/bw/while/add_1*
T0*
_output_shapes
: 
╛
;biLSTM_layers/bidirectional_rnn/bw/bw/while/NextIteration_2NextIterationObiLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
╡
;biLSTM_layers/bidirectional_rnn/bw/bw/while/NextIteration_3NextIteration4biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1*
T0*(
_output_shapes
:         А
╡
;biLSTM_layers/bidirectional_rnn/bw/bw/while/NextIteration_4NextIteration4biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2*(
_output_shapes
:         А*
T0
Н
0biLSTM_layers/bidirectional_rnn/bw/bw/while/ExitExit2biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch*
_output_shapes
: *
T0
С
2biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_1Exit4biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_1*
T0*
_output_shapes
: 
С
2biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_2Exit4biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_2*
T0*
_output_shapes
: 
г
2biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_3Exit4biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_3*
T0*(
_output_shapes
:         А
г
2biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_4Exit4biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_4*
T0*(
_output_shapes
:         А
в
HbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV31biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray2biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_2*
_output_shapes
: *D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray
╩
BbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayStack/range/startConst*
value	B : *D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray*
dtype0*
_output_shapes
: 
╩
BbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayStack/range/deltaConst*
value	B :*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray*
dtype0*
_output_shapes
: 
О
<biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayStack/rangeRangeBbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayStack/range/startHbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV3BbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayStack/range/delta*

Tidx0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray*#
_output_shapes
:         
╖
JbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV31biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray<biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayStack/range2biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_2*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray*
dtype0*5
_output_shapes#
!:                  А*%
element_shape:         А
x
-biLSTM_layers/bidirectional_rnn/bw/bw/Const_4Const*
valueB:А*
dtype0*
_output_shapes
:
n
,biLSTM_layers/bidirectional_rnn/bw/bw/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
u
3biLSTM_layers/bidirectional_rnn/bw/bw/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
u
3biLSTM_layers/bidirectional_rnn/bw/bw/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ў
-biLSTM_layers/bidirectional_rnn/bw/bw/range_1Range3biLSTM_layers/bidirectional_rnn/bw/bw/range_1/start,biLSTM_layers/bidirectional_rnn/bw/bw/Rank_13biLSTM_layers/bidirectional_rnn/bw/bw/range_1/delta*

Tidx0*
_output_shapes
:
И
7biLSTM_layers/bidirectional_rnn/bw/bw/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
u
3biLSTM_layers/bidirectional_rnn/bw/bw/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
С
.biLSTM_layers/bidirectional_rnn/bw/bw/concat_2ConcatV27biLSTM_layers/bidirectional_rnn/bw/bw/concat_2/values_0-biLSTM_layers/bidirectional_rnn/bw/bw/range_13biLSTM_layers/bidirectional_rnn/bw/bw/concat_2/axis*
N*
_output_shapes
:*

Tidx0*
T0
З
1biLSTM_layers/bidirectional_rnn/bw/bw/transpose_1	TransposeJbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3.biLSTM_layers/bidirectional_rnn/bw/bw/concat_2*
T0*5
_output_shapes#
!:                  А*
Tperm0
╘
biLSTM_layers/ReverseSequenceReverseSequence1biLSTM_layers/bidirectional_rnn/bw/bw/transpose_1Sum*
seq_dim*5
_output_shapes#
!:                  А*

Tlen0*
	batch_dim *
T0
[
biLSTM_layers/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
т
biLSTM_layers/concatConcatV21biLSTM_layers/bidirectional_rnn/fw/fw/transpose_1biLSTM_layers/ReverseSequencebiLSTM_layers/concat/axis*
T0*
N*5
_output_shapes#
!:                  А*

Tidx0
o
biLSTM_layers/dropout/ShapeShapebiLSTM_layers/concat*
T0*
out_type0*
_output_shapes
:
m
(biLSTM_layers/dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0
m
(biLSTM_layers/dropout/random_uniform/maxConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╞
2biLSTM_layers/dropout/random_uniform/RandomUniformRandomUniformbiLSTM_layers/dropout/Shape*
T0*
dtype0*5
_output_shapes#
!:                  А*
seed2 *

seed 
д
(biLSTM_layers/dropout/random_uniform/subSub(biLSTM_layers/dropout/random_uniform/max(biLSTM_layers/dropout/random_uniform/min*
T0*
_output_shapes
: 
═
(biLSTM_layers/dropout/random_uniform/mulMul2biLSTM_layers/dropout/random_uniform/RandomUniform(biLSTM_layers/dropout/random_uniform/sub*
T0*5
_output_shapes#
!:                  А
┐
$biLSTM_layers/dropout/random_uniformAdd(biLSTM_layers/dropout/random_uniform/mul(biLSTM_layers/dropout/random_uniform/min*5
_output_shapes#
!:                  А*
T0
t
biLSTM_layers/dropout/addAdd	keep_prob$biLSTM_layers/dropout/random_uniform*
T0*
_output_shapes
:
b
biLSTM_layers/dropout/FloorFloorbiLSTM_layers/dropout/add*
T0*
_output_shapes
:
h
biLSTM_layers/dropout/divRealDivbiLSTM_layers/concat	keep_prob*
T0*
_output_shapes
:
Ш
biLSTM_layers/dropout/mulMulbiLSTM_layers/dropout/divbiLSTM_layers/dropout/Floor*5
_output_shapes#
!:                  А*
T0
l
biLSTM_layers/ShapeShapebiLSTM_layers/dropout/mul*
_output_shapes
:*
T0*
out_type0
l
biLSTM_layers/Reshape/shapeConst*
_output_shapes
:*
valueB"       *
dtype0
Щ
biLSTM_layers/ReshapeReshapebiLSTM_layers/dropout/mulbiLSTM_layers/Reshape/shape*(
_output_shapes
:         А*
T0*
Tshape0
п
6biLSTM_layers/W_out/Initializer/truncated_normal/shapeConst*
valueB"      *&
_class
loc:@biLSTM_layers/W_out*
dtype0*
_output_shapes
:
в
5biLSTM_layers/W_out/Initializer/truncated_normal/meanConst*
valueB
 *    *&
_class
loc:@biLSTM_layers/W_out*
dtype0*
_output_shapes
: 
д
7biLSTM_layers/W_out/Initializer/truncated_normal/stddevConst*
valueB
 *
╫#<*&
_class
loc:@biLSTM_layers/W_out*
dtype0*
_output_shapes
: 
Г
@biLSTM_layers/W_out/Initializer/truncated_normal/TruncatedNormalTruncatedNormal6biLSTM_layers/W_out/Initializer/truncated_normal/shape*
T0*&
_class
loc:@biLSTM_layers/W_out*
seed2 *
dtype0*
_output_shapes
:	А*

seed 
И
4biLSTM_layers/W_out/Initializer/truncated_normal/mulMul@biLSTM_layers/W_out/Initializer/truncated_normal/TruncatedNormal7biLSTM_layers/W_out/Initializer/truncated_normal/stddev*
T0*&
_class
loc:@biLSTM_layers/W_out*
_output_shapes
:	А
Ў
0biLSTM_layers/W_out/Initializer/truncated_normalAdd4biLSTM_layers/W_out/Initializer/truncated_normal/mul5biLSTM_layers/W_out/Initializer/truncated_normal/mean*
_output_shapes
:	А*
T0*&
_class
loc:@biLSTM_layers/W_out
▒
biLSTM_layers/W_out
VariableV2*
dtype0*
_output_shapes
:	А*
shared_name *&
_class
loc:@biLSTM_layers/W_out*
	container *
shape:	А
ц
biLSTM_layers/W_out/AssignAssignbiLSTM_layers/W_out0biLSTM_layers/W_out/Initializer/truncated_normal*
use_locking(*
T0*&
_class
loc:@biLSTM_layers/W_out*
validate_shape(*
_output_shapes
:	А
Л
biLSTM_layers/W_out/readIdentitybiLSTM_layers/W_out*
T0*&
_class
loc:@biLSTM_layers/W_out*
_output_shapes
:	А
а
2biLSTM_layers/b/Initializer/truncated_normal/shapeConst*
valueB:*"
_class
loc:@biLSTM_layers/b*
dtype0*
_output_shapes
:
Ъ
1biLSTM_layers/b/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *"
_class
loc:@biLSTM_layers/b
Ь
3biLSTM_layers/b/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *
╫#<*"
_class
loc:@biLSTM_layers/b
Є
<biLSTM_layers/b/Initializer/truncated_normal/TruncatedNormalTruncatedNormal2biLSTM_layers/b/Initializer/truncated_normal/shape*
dtype0*
_output_shapes
:*

seed *
T0*"
_class
loc:@biLSTM_layers/b*
seed2 
є
0biLSTM_layers/b/Initializer/truncated_normal/mulMul<biLSTM_layers/b/Initializer/truncated_normal/TruncatedNormal3biLSTM_layers/b/Initializer/truncated_normal/stddev*
T0*"
_class
loc:@biLSTM_layers/b*
_output_shapes
:
с
,biLSTM_layers/b/Initializer/truncated_normalAdd0biLSTM_layers/b/Initializer/truncated_normal/mul1biLSTM_layers/b/Initializer/truncated_normal/mean*
T0*"
_class
loc:@biLSTM_layers/b*
_output_shapes
:
Я
biLSTM_layers/b
VariableV2*
_output_shapes
:*
shared_name *"
_class
loc:@biLSTM_layers/b*
	container *
shape:*
dtype0
╤
biLSTM_layers/b/AssignAssignbiLSTM_layers/b,biLSTM_layers/b/Initializer/truncated_normal*
use_locking(*
T0*"
_class
loc:@biLSTM_layers/b*
validate_shape(*
_output_shapes
:
z
biLSTM_layers/b/readIdentitybiLSTM_layers/b*
T0*"
_class
loc:@biLSTM_layers/b*
_output_shapes
:
з
biLSTM_layers/MatMulMatMulbiLSTM_layers/ReshapebiLSTM_layers/W_out/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
v
biLSTM_layers/addAddbiLSTM_layers/MatMulbiLSTM_layers/b/read*
T0*'
_output_shapes
:         
k
!biLSTM_layers/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
m
#biLSTM_layers/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
m
#biLSTM_layers/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┐
biLSTM_layers/strided_sliceStridedSlicebiLSTM_layers/Shape!biLSTM_layers/strided_slice/stack#biLSTM_layers/strided_slice/stack_1#biLSTM_layers/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
j
biLSTM_layers/Reshape_1/shape/0Const*
valueB :
         *
dtype0*
_output_shapes
: 
a
biLSTM_layers/Reshape_1/shape/2Const*
_output_shapes
: *
value	B :*
dtype0
╛
biLSTM_layers/Reshape_1/shapePackbiLSTM_layers/Reshape_1/shape/0biLSTM_layers/strided_slicebiLSTM_layers/Reshape_1/shape/2*
T0*

axis *
N*
_output_shapes
:
б
biLSTM_layers/Reshape_1ReshapebiLSTM_layers/addbiLSTM_layers/Reshape_1/shape*4
_output_shapes"
 :                  *
T0*
Tshape0
Э
,transitions/Initializer/random_uniform/shapeConst*
valueB"      *
_class
loc:@transitions*
dtype0*
_output_shapes
:
П
*transitions/Initializer/random_uniform/minConst*
valueB
 *bЧ'┐*
_class
loc:@transitions*
dtype0*
_output_shapes
: 
П
*transitions/Initializer/random_uniform/maxConst*
valueB
 *bЧ'?*
_class
loc:@transitions*
dtype0*
_output_shapes
: 
т
4transitions/Initializer/random_uniform/RandomUniformRandomUniform,transitions/Initializer/random_uniform/shape*
T0*
_class
loc:@transitions*
seed2 *
dtype0*
_output_shapes

:*

seed 
╩
*transitions/Initializer/random_uniform/subSub*transitions/Initializer/random_uniform/max*transitions/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@transitions
▄
*transitions/Initializer/random_uniform/mulMul4transitions/Initializer/random_uniform/RandomUniform*transitions/Initializer/random_uniform/sub*
T0*
_class
loc:@transitions*
_output_shapes

:
╬
&transitions/Initializer/random_uniformAdd*transitions/Initializer/random_uniform/mul*transitions/Initializer/random_uniform/min*
T0*
_class
loc:@transitions*
_output_shapes

:
Я
transitions
VariableV2*
dtype0*
_output_shapes

:*
shared_name *
_class
loc:@transitions*
	container *
shape
:
├
transitions/AssignAssigntransitions&transitions/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@transitions*
validate_shape(*
_output_shapes

:
r
transitions/readIdentitytransitions*
T0*
_class
loc:@transitions*
_output_shapes

:
\
ShapeShapebiLSTM_layers/Reshape_1*
T0*
out_type0*
_output_shapes
:
]
strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
∙
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
I
Equal/yConst*
value	B :*
dtype0*
_output_shapes
: 
G
EqualEqualstrided_sliceEqual/y*
_output_shapes
: *
T0
F
cond/SwitchSwitchEqualEqual*
_output_shapes
: : *
T0

I
cond/switch_tIdentitycond/Switch:1*
T0
*
_output_shapes
: 
G
cond/switch_fIdentitycond/Switch*
T0
*
_output_shapes
: 
@
cond/pred_idIdentityEqual*
T0
*
_output_shapes
: 
]

cond/ShapeShapecond/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
═
cond/Shape/SwitchSwitchbiLSTM_layers/Reshape_1cond/pred_id*
T0**
_class 
loc:@biLSTM_layers/Reshape_1*T
_output_shapesB
@:                  :                  
r
cond/strided_slice/stackConst^cond/switch_t*
valueB: *
dtype0*
_output_shapes
:
t
cond/strided_slice/stack_1Const^cond/switch_t*
valueB:*
dtype0*
_output_shapes
:
t
cond/strided_slice/stack_2Const^cond/switch_t*
dtype0*
_output_shapes
:*
valueB:
Т
cond/strided_sliceStridedSlice
cond/Shapecond/strided_slice/stackcond/strided_slice/stack_1cond/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
b
cond/range/startConst^cond/switch_t*
value	B : *
dtype0*
_output_shapes
: 
b
cond/range/deltaConst^cond/switch_t*
value	B :*
dtype0*
_output_shapes
: 
|

cond/rangeRangecond/range/startcond/strided_slicecond/range/delta*#
_output_shapes
:         *

Tidx0
s
cond/Reshape/shapeConst^cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
w
cond/ReshapeReshape
cond/rangecond/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
u
cond/SqueezeSqueezecond/Shape/Switch:1*
squeeze_dims
*
T0*'
_output_shapes
:         
b
cond/concat/axisConst^cond/switch_t*
value	B :*
dtype0*
_output_shapes
: 
Э
cond/concatConcatV2cond/Reshapecond/concat/Switch:1cond/concat/axis*
N*0
_output_shapes
:                  *

Tidx0*
T0
ж
cond/concat/SwitchSwitchtargetscond/pred_id*L
_output_shapes:
8:                  :                  *
T0*
_class
loc:@targets
m
cond/GatherNdGatherNdcond/Squeezecond/concat*
_output_shapes
:*
Tindices0*
Tparams0
_
cond/Shape_1Shapecond/Shape_1/Switch*
T0*
out_type0*
_output_shapes
:
╧
cond/Shape_1/SwitchSwitchbiLSTM_layers/Reshape_1cond/pred_id*
T0**
_class 
loc:@biLSTM_layers/Reshape_1*T
_output_shapesB
@:                  :                  
t
cond/strided_slice_1/stackConst^cond/switch_f*
valueB: *
dtype0*
_output_shapes
:
v
cond/strided_slice_1/stack_1Const^cond/switch_f*
dtype0*
_output_shapes
:*
valueB:
v
cond/strided_slice_1/stack_2Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
Ь
cond/strided_slice_1StridedSlicecond/Shape_1cond/strided_slice_1/stackcond/strided_slice_1/stack_1cond/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
_
cond/Shape_2Shapecond/Shape_1/Switch*
T0*
out_type0*
_output_shapes
:
t
cond/strided_slice_2/stackConst^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
v
cond/strided_slice_2/stack_1Const^cond/switch_f*
_output_shapes
:*
valueB:*
dtype0
v
cond/strided_slice_2/stack_2Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
Ь
cond/strided_slice_2StridedSlicecond/Shape_2cond/strided_slice_2/stackcond/strided_slice_2/stack_1cond/strided_slice_2/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
_
cond/Shape_3Shapecond/Shape_1/Switch*
out_type0*
_output_shapes
:*
T0
t
cond/strided_slice_3/stackConst^cond/switch_f*
_output_shapes
:*
valueB:*
dtype0
v
cond/strided_slice_3/stack_1Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
v
cond/strided_slice_3/stack_2Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
Ь
cond/strided_slice_3StridedSlicecond/Shape_3cond/strided_slice_3/stackcond/strided_slice_3/stack_1cond/strided_slice_3/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
w
cond/Reshape_1/shapeConst^cond/switch_f*
valueB:
         *
dtype0*
_output_shapes
:
А
cond/Reshape_1Reshapecond/Shape_1/Switchcond/Reshape_1/shape*
T0*
Tshape0*#
_output_shapes
:         
d
cond/range_1/startConst^cond/switch_f*
dtype0*
_output_shapes
: *
value	B : 
d
cond/range_1/deltaConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
Д
cond/range_1Rangecond/range_1/startcond/strided_slice_1cond/range_1/delta*#
_output_shapes
:         *

Tidx0
a
cond/mulMulcond/range_1cond/strided_slice_2*#
_output_shapes
:         *
T0
_

cond/mul_1Mulcond/mulcond/strided_slice_3*
T0*#
_output_shapes
:         
e
cond/ExpandDims/dimConst^cond/switch_f*
dtype0*
_output_shapes
: *
value	B :
|
cond/ExpandDims
ExpandDims
cond/mul_1cond/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
d
cond/range_2/startConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
d
cond/range_2/deltaConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
Д
cond/range_2Rangecond/range_2/startcond/strided_slice_2cond/range_2/delta*#
_output_shapes
:         *

Tidx0
c

cond/mul_2Mulcond/range_2cond/strided_slice_3*
T0*#
_output_shapes
:         
g
cond/ExpandDims_1/dimConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
А
cond/ExpandDims_1
ExpandDims
cond/mul_2cond/ExpandDims_1/dim*
T0*'
_output_shapes
:         *

Tdim0
n
cond/addAddcond/ExpandDimscond/ExpandDims_1*
T0*0
_output_shapes
:                  
i

cond/add_1Addcond/addcond/add_1/Switch*0
_output_shapes
:                  *
T0
е
cond/add_1/SwitchSwitchtargetscond/pred_id*
T0*
_class
loc:@targets*L
_output_shapes:
8:                  :                  
w
cond/Reshape_2/shapeConst^cond/switch_f*
_output_shapes
:*
valueB:
         *
dtype0
w
cond/Reshape_2Reshape
cond/add_1cond/Reshape_2/shape*
T0*
Tshape0*#
_output_shapes
:         
С
cond/GatherGathercond/Reshape_1cond/Reshape_2*
Tindices0*
Tparams0*
validate_indices(*#
_output_shapes
:         
В
cond/Reshape_3/shapePackcond/strided_slice_1cond/strided_slice_2*
T0*

axis *
N*
_output_shapes
:
Е
cond/Reshape_3Reshapecond/Gathercond/Reshape_3/shape*
T0*
Tshape0*0
_output_shapes
:                  
]
cond/Shape_4Shapecond/add_1/Switch*
T0*
out_type0*
_output_shapes
:
t
cond/strided_slice_4/stackConst^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
v
cond/strided_slice_4/stack_1Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
v
cond/strided_slice_4/stack_2Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
Ь
cond/strided_slice_4StridedSlicecond/Shape_4cond/strided_slice_4/stackcond/strided_slice_4/stack_1cond/strided_slice_4/stack_2*
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask 
i
cond/SequenceMask/ConstConst^cond/switch_f*
_output_shapes
: *
value	B : *
dtype0
k
cond/SequenceMask/Const_1Const^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
Ы
cond/SequenceMask/RangeRangecond/SequenceMask/Constcond/strided_slice_4cond/SequenceMask/Const_1*#
_output_shapes
:         *

Tidx0
{
 cond/SequenceMask/ExpandDims/dimConst^cond/switch_f*
valueB :
         *
dtype0*
_output_shapes
: 
п
cond/SequenceMask/ExpandDims
ExpandDims#cond/SequenceMask/ExpandDims/Switch cond/SequenceMask/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
Х
#cond/SequenceMask/ExpandDims/SwitchSwitchSumcond/pred_id*
T0*
_class

loc:@Sum*2
_output_shapes 
:         :         
}
cond/SequenceMask/CastCastcond/SequenceMask/ExpandDims*'
_output_shapes
:         *

DstT0*

SrcT0
К
cond/SequenceMask/LessLesscond/SequenceMask/Rangecond/SequenceMask/Cast*0
_output_shapes
:                  *
T0
В
cond/SequenceMask/Cast_1Castcond/SequenceMask/Less*

SrcT0
*0
_output_shapes
:                  *

DstT0
v

cond/mul_3Mulcond/Reshape_3cond/SequenceMask/Cast_1*
T0*0
_output_shapes
:                  
l
cond/Sum/reduction_indicesConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
В
cond/SumSum
cond/mul_3cond/Sum/reduction_indices*
T0*#
_output_shapes
:         *

Tidx0*
	keep_dims( 
]
cond/Shape_5Shapecond/add_1/Switch*
T0*
out_type0*
_output_shapes
:
t
cond/strided_slice_5/stackConst^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
v
cond/strided_slice_5/stack_1Const^cond/switch_f*
_output_shapes
:*
valueB:*
dtype0
v
cond/strided_slice_5/stack_2Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
Ь
cond/strided_slice_5StridedSlicecond/Shape_5cond/strided_slice_5/stackcond/strided_slice_5/stack_1cond/strided_slice_5/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
\

cond/sub/yConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
R
cond/subSubcond/strided_slice_5
cond/sub/y*
T0*
_output_shapes
: 
q
cond/Slice/beginConst^cond/switch_f*
valueB"        *
dtype0*
_output_shapes
:
l
cond/Slice/size/0Const^cond/switch_f*
_output_shapes
: *
valueB :
         *
dtype0
n
cond/Slice/sizePackcond/Slice/size/0cond/sub*

axis *
N*
_output_shapes
:*
T0
С

cond/SliceSlicecond/add_1/Switchcond/Slice/begincond/Slice/size*
T0*
Index0*0
_output_shapes
:                  
s
cond/Slice_1/beginConst^cond/switch_f*
_output_shapes
:*
valueB"       *
dtype0
n
cond/Slice_1/size/0Const^cond/switch_f*
valueB :
         *
dtype0*
_output_shapes
: 
r
cond/Slice_1/sizePackcond/Slice_1/size/0cond/sub*
T0*

axis *
N*
_output_shapes
:
Ч
cond/Slice_1Slicecond/add_1/Switchcond/Slice_1/begincond/Slice_1/size*
T0*
Index0*0
_output_shapes
:                  
^
cond/mul_4/yConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
f

cond/mul_4Mul
cond/Slicecond/mul_4/y*
T0*0
_output_shapes
:                  
f

cond/add_2Add
cond/mul_4cond/Slice_1*
T0*0
_output_shapes
:                  
w
cond/Reshape_4/shapeConst^cond/switch_f*
valueB:
         *
dtype0*
_output_shapes
:
y
cond/Reshape_4Reshapecond/Reshape_4/Switchcond/Reshape_4/shape*
T0*
Tshape0*
_output_shapes
:1
Т
cond/Reshape_4/SwitchSwitchtransitions/readcond/pred_id*
T0*
_class
loc:@transitions*(
_output_shapes
::
Ь
cond/Gather_1Gathercond/Reshape_4
cond/add_2*
Tparams0*
validate_indices(*0
_output_shapes
:                  *
Tindices0
]
cond/Shape_6Shapecond/add_1/Switch*
_output_shapes
:*
T0*
out_type0
t
cond/strided_slice_6/stackConst^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
v
cond/strided_slice_6/stack_1Const^cond/switch_f*
valueB:*
dtype0*
_output_shapes
:
v
cond/strided_slice_6/stack_2Const^cond/switch_f*
dtype0*
_output_shapes
:*
valueB:
Ь
cond/strided_slice_6StridedSlicecond/Shape_6cond/strided_slice_6/stackcond/strided_slice_6/stack_1cond/strided_slice_6/stack_2*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
k
cond/SequenceMask_1/ConstConst^cond/switch_f*
value	B : *
dtype0*
_output_shapes
: 
m
cond/SequenceMask_1/Const_1Const^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
б
cond/SequenceMask_1/RangeRangecond/SequenceMask_1/Constcond/strided_slice_6cond/SequenceMask_1/Const_1*

Tidx0*#
_output_shapes
:         
}
"cond/SequenceMask_1/ExpandDims/dimConst^cond/switch_f*
valueB :
         *
dtype0*
_output_shapes
: 
│
cond/SequenceMask_1/ExpandDims
ExpandDims#cond/SequenceMask/ExpandDims/Switch"cond/SequenceMask_1/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
Б
cond/SequenceMask_1/CastCastcond/SequenceMask_1/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
Р
cond/SequenceMask_1/LessLesscond/SequenceMask_1/Rangecond/SequenceMask_1/Cast*
T0*0
_output_shapes
:                  
Ж
cond/SequenceMask_1/Cast_1Castcond/SequenceMask_1/Less*

SrcT0
*0
_output_shapes
:                  *

DstT0
s
cond/Slice_2/beginConst^cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
r
cond/Slice_2/sizeConst^cond/switch_f*
_output_shapes
:*
valueB"        *
dtype0
а
cond/Slice_2Slicecond/SequenceMask_1/Cast_1cond/Slice_2/begincond/Slice_2/size*
T0*
Index0*0
_output_shapes
:                  
i

cond/mul_5Mulcond/Gather_1cond/Slice_2*
T0*0
_output_shapes
:                  
n
cond/Sum_1/reduction_indicesConst^cond/switch_f*
value	B :*
dtype0*
_output_shapes
: 
Ж

cond/Sum_1Sum
cond/mul_5cond/Sum_1/reduction_indices*#
_output_shapes
:         *

Tidx0*
	keep_dims( *
T0
U

cond/add_3Addcond/Sum
cond/Sum_1*
T0*#
_output_shapes
:         
\

cond/MergeMerge
cond/add_3cond/GatherNd*
T0*
N*
_output_shapes
:: 
`
Slice/beginConst*!
valueB"            *
dtype0*
_output_shapes
:
_

Slice/sizeConst*
dtype0*
_output_shapes
:*!
valueB"           
Г
SliceSlicebiLSTM_layers/Reshape_1Slice/begin
Slice/size*
T0*
Index0*+
_output_shapes
:         
b
SqueezeSqueezeSlice*
T0*'
_output_shapes
:         *
squeeze_dims

^
Shape_1ShapebiLSTM_layers/Reshape_1*
_output_shapes
:*
T0*
out_type0
_
strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
a
strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
strided_slice_1StridedSliceShape_1strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
K
	Equal_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
M
Equal_1Equalstrided_slice_1	Equal_1/y*
T0*
_output_shapes
: 
L
cond_1/SwitchSwitchEqual_1Equal_1*
T0
*
_output_shapes
: : 
M
cond_1/switch_tIdentitycond_1/Switch:1*
T0
*
_output_shapes
: 
K
cond_1/switch_fIdentitycond_1/Switch*
_output_shapes
: *
T0

D
cond_1/pred_idIdentityEqual_1*
_output_shapes
: *
T0

И
,cond_1/ReduceLogSumExp/Max/reduction_indicesConst^cond_1/switch_t*
valueB:*
dtype0*
_output_shapes
:
├
cond_1/ReduceLogSumExp/MaxMax#cond_1/ReduceLogSumExp/Max/Switch:1,cond_1/ReduceLogSumExp/Max/reduction_indices*'
_output_shapes
:         *

Tidx0*
	keep_dims(*
T0
е
!cond_1/ReduceLogSumExp/Max/SwitchSwitchSqueezecond_1/pred_id*:
_output_shapes(
&:         :         *
T0*
_class
loc:@Squeeze
y
cond_1/ReduceLogSumExp/IsFiniteIsFinitecond_1/ReduceLogSumExp/Max*
T0*'
_output_shapes
:         
|
!cond_1/ReduceLogSumExp/zeros_like	ZerosLikecond_1/ReduceLogSumExp/Max*'
_output_shapes
:         *
T0
╣
cond_1/ReduceLogSumExp/SelectSelectcond_1/ReduceLogSumExp/IsFinitecond_1/ReduceLogSumExp/Max!cond_1/ReduceLogSumExp/zeros_like*
T0*'
_output_shapes
:         
Д
#cond_1/ReduceLogSumExp/StopGradientStopGradientcond_1/ReduceLogSumExp/Select*
T0*'
_output_shapes
:         
Э
cond_1/ReduceLogSumExp/subSub#cond_1/ReduceLogSumExp/Max/Switch:1#cond_1/ReduceLogSumExp/StopGradient*
T0*'
_output_shapes
:         
o
cond_1/ReduceLogSumExp/ExpExpcond_1/ReduceLogSumExp/sub*'
_output_shapes
:         *
T0
И
,cond_1/ReduceLogSumExp/Sum/reduction_indicesConst^cond_1/switch_t*
valueB:*
dtype0*
_output_shapes
:
╢
cond_1/ReduceLogSumExp/SumSumcond_1/ReduceLogSumExp/Exp,cond_1/ReduceLogSumExp/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:         
k
cond_1/ReduceLogSumExp/LogLogcond_1/ReduceLogSumExp/Sum*#
_output_shapes
:         *
T0
v
cond_1/ReduceLogSumExp/ShapeShapecond_1/ReduceLogSumExp/Log*
_output_shapes
:*
T0*
out_type0
и
cond_1/ReduceLogSumExp/ReshapeReshape#cond_1/ReduceLogSumExp/StopGradientcond_1/ReduceLogSumExp/Shape*
T0*
Tshape0*#
_output_shapes
:         
Л
cond_1/ReduceLogSumExp/addAddcond_1/ReduceLogSumExp/Logcond_1/ReduceLogSumExp/Reshape*
T0*#
_output_shapes
:         
y
cond_1/Slice/beginConst^cond_1/switch_f*!
valueB"           *
dtype0*
_output_shapes
:
x
cond_1/Slice/sizeConst^cond_1/switch_f*!
valueB"            *
dtype0*
_output_shapes
:
Э
cond_1/SliceSlicecond_1/Slice/Switchcond_1/Slice/begincond_1/Slice/size*
T0*
Index0*4
_output_shapes"
 :                  
╤
cond_1/Slice/SwitchSwitchbiLSTM_layers/Reshape_1cond_1/pred_id*
T0**
_class 
loc:@biLSTM_layers/Reshape_1*T
_output_shapesB
@:                  :                  
i
cond_1/ExpandDims/dimConst^cond_1/switch_f*
value	B : *
dtype0*
_output_shapes
: 
Й
cond_1/ExpandDims
ExpandDimscond_1/ExpandDims/Switchcond_1/ExpandDims/dim*"
_output_shapes
:*

Tdim0*
T0
Ч
cond_1/ExpandDims/SwitchSwitchtransitions/readcond_1/pred_id*(
_output_shapes
::*
T0*
_class
loc:@transitions
`
cond_1/sub/yConst^cond_1/switch_f*
value	B :*
dtype0*
_output_shapes
: 
`

cond_1/subSubcond_1/sub/Switchcond_1/sub/y*
T0*#
_output_shapes
:         
Е
cond_1/sub/SwitchSwitchSumcond_1/pred_id*
_class

loc:@Sum*2
_output_shapes 
:         :         *
T0
c
cond_1/rnn/RankConst^cond_1/switch_f*
value	B :*
dtype0*
_output_shapes
: 
j
cond_1/rnn/range/startConst^cond_1/switch_f*
value	B :*
dtype0*
_output_shapes
: 
j
cond_1/rnn/range/deltaConst^cond_1/switch_f*
dtype0*
_output_shapes
: *
value	B :
В
cond_1/rnn/rangeRangecond_1/rnn/range/startcond_1/rnn/Rankcond_1/rnn/range/delta*

Tidx0*
_output_shapes
:
}
cond_1/rnn/concat/values_0Const^cond_1/switch_f*
dtype0*
_output_shapes
:*
valueB"       
j
cond_1/rnn/concat/axisConst^cond_1/switch_f*
value	B : *
dtype0*
_output_shapes
: 
Э
cond_1/rnn/concatConcatV2cond_1/rnn/concat/values_0cond_1/rnn/rangecond_1/rnn/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
О
cond_1/rnn/transpose	Transposecond_1/Slicecond_1/rnn/concat*
T0*4
_output_shapes"
 :                  *
Tperm0
`
cond_1/rnn/sequence_lengthIdentity
cond_1/sub*
T0*#
_output_shapes
:         
d
cond_1/rnn/ShapeShapecond_1/rnn/transpose*
T0*
out_type0*
_output_shapes
:
z
cond_1/rnn/strided_slice/stackConst^cond_1/switch_f*
valueB:*
dtype0*
_output_shapes
:
|
 cond_1/rnn/strided_slice/stack_1Const^cond_1/switch_f*
valueB:*
dtype0*
_output_shapes
:
|
 cond_1/rnn/strided_slice/stack_2Const^cond_1/switch_f*
valueB:*
dtype0*
_output_shapes
:
░
cond_1/rnn/strided_sliceStridedSlicecond_1/rnn/Shapecond_1/rnn/strided_slice/stack cond_1/rnn/strided_slice/stack_1 cond_1/rnn/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
l
cond_1/rnn/Shape_1Shapecond_1/rnn/sequence_length*
_output_shapes
:*
T0*
out_type0
l
cond_1/rnn/stackPackcond_1/rnn/strided_slice*
T0*

axis *
N*
_output_shapes
:
d
cond_1/rnn/EqualEqualcond_1/rnn/Shape_1cond_1/rnn/stack*
T0*
_output_shapes
:
l
cond_1/rnn/ConstConst^cond_1/switch_f*
dtype0*
_output_shapes
:*
valueB: 
n
cond_1/rnn/AllAllcond_1/rnn/Equalcond_1/rnn/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
д
cond_1/rnn/Assert/ConstConst^cond_1/switch_f*K
valueBB@ B:Expected shape for Tensor cond_1/rnn/sequence_length:0 is *
dtype0*
_output_shapes
: 
|
cond_1/rnn/Assert/Const_1Const^cond_1/switch_f*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
м
cond_1/rnn/Assert/Assert/data_0Const^cond_1/switch_f*
dtype0*
_output_shapes
: *K
valueBB@ B:Expected shape for Tensor cond_1/rnn/sequence_length:0 is 
В
cond_1/rnn/Assert/Assert/data_2Const^cond_1/switch_f*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
║
cond_1/rnn/Assert/AssertAssertcond_1/rnn/Allcond_1/rnn/Assert/Assert/data_0cond_1/rnn/stackcond_1/rnn/Assert/Assert/data_2cond_1/rnn/Shape_1*
T
2*
	summarize
З
cond_1/rnn/CheckSeqLenIdentitycond_1/rnn/sequence_length^cond_1/rnn/Assert/Assert*#
_output_shapes
:         *
T0
f
cond_1/rnn/Shape_2Shapecond_1/rnn/transpose*
_output_shapes
:*
T0*
out_type0
|
 cond_1/rnn/strided_slice_1/stackConst^cond_1/switch_f*
dtype0*
_output_shapes
:*
valueB: 
~
"cond_1/rnn/strided_slice_1/stack_1Const^cond_1/switch_f*
dtype0*
_output_shapes
:*
valueB:
~
"cond_1/rnn/strided_slice_1/stack_2Const^cond_1/switch_f*
valueB:*
dtype0*
_output_shapes
:
║
cond_1/rnn/strided_slice_1StridedSlicecond_1/rnn/Shape_2 cond_1/rnn/strided_slice_1/stack"cond_1/rnn/strided_slice_1/stack_1"cond_1/rnn/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
f
cond_1/rnn/Shape_3Shapecond_1/rnn/transpose*
_output_shapes
:*
T0*
out_type0
|
 cond_1/rnn/strided_slice_2/stackConst^cond_1/switch_f*
valueB:*
dtype0*
_output_shapes
:
~
"cond_1/rnn/strided_slice_2/stack_1Const^cond_1/switch_f*
dtype0*
_output_shapes
:*
valueB:
~
"cond_1/rnn/strided_slice_2/stack_2Const^cond_1/switch_f*
valueB:*
dtype0*
_output_shapes
:
║
cond_1/rnn/strided_slice_2StridedSlicecond_1/rnn/Shape_3 cond_1/rnn/strided_slice_2/stack"cond_1/rnn/strided_slice_2/stack_1"cond_1/rnn/strided_slice_2/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
m
cond_1/rnn/ExpandDims/dimConst^cond_1/switch_f*
dtype0*
_output_shapes
: *
value	B : 
Л
cond_1/rnn/ExpandDims
ExpandDimscond_1/rnn/strided_slice_2cond_1/rnn/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
n
cond_1/rnn/Const_1Const^cond_1/switch_f*
valueB:*
dtype0*
_output_shapes
:
l
cond_1/rnn/concat_1/axisConst^cond_1/switch_f*
dtype0*
_output_shapes
: *
value	B : 
Ю
cond_1/rnn/concat_1ConcatV2cond_1/rnn/ExpandDimscond_1/rnn/Const_1cond_1/rnn/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
m
cond_1/rnn/zeros/ConstConst^cond_1/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
Й
cond_1/rnn/zerosFillcond_1/rnn/concat_1cond_1/rnn/zeros/Const*
T0*

index_type0*'
_output_shapes
:         
n
cond_1/rnn/Const_2Const^cond_1/switch_f*
valueB: *
dtype0*
_output_shapes
:

cond_1/rnn/MinMincond_1/rnn/CheckSeqLencond_1/rnn/Const_2*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
n
cond_1/rnn/Const_3Const^cond_1/switch_f*
valueB: *
dtype0*
_output_shapes
:

cond_1/rnn/MaxMaxcond_1/rnn/CheckSeqLencond_1/rnn/Const_3*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
c
cond_1/rnn/timeConst^cond_1/switch_f*
dtype0*
_output_shapes
: *
value	B : 
Ш
cond_1/rnn/TensorArrayTensorArrayV3cond_1/rnn/strided_slice_1*6
tensor_array_name!cond_1/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *$
element_shape:         *
dynamic_size( *
clear_after_read(*
identical_element_shapes(
Щ
cond_1/rnn/TensorArray_1TensorArrayV3cond_1/rnn/strided_slice_1*5
tensor_array_name cond_1/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *$
element_shape:         *
dynamic_size( *
clear_after_read(*
identical_element_shapes(
w
#cond_1/rnn/TensorArrayUnstack/ShapeShapecond_1/rnn/transpose*
T0*
out_type0*
_output_shapes
:
Н
1cond_1/rnn/TensorArrayUnstack/strided_slice/stackConst^cond_1/switch_f*
valueB: *
dtype0*
_output_shapes
:
П
3cond_1/rnn/TensorArrayUnstack/strided_slice/stack_1Const^cond_1/switch_f*
dtype0*
_output_shapes
:*
valueB:
П
3cond_1/rnn/TensorArrayUnstack/strided_slice/stack_2Const^cond_1/switch_f*
valueB:*
dtype0*
_output_shapes
:
П
+cond_1/rnn/TensorArrayUnstack/strided_sliceStridedSlice#cond_1/rnn/TensorArrayUnstack/Shape1cond_1/rnn/TensorArrayUnstack/strided_slice/stack3cond_1/rnn/TensorArrayUnstack/strided_slice/stack_13cond_1/rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
}
)cond_1/rnn/TensorArrayUnstack/range/startConst^cond_1/switch_f*
value	B : *
dtype0*
_output_shapes
: 
}
)cond_1/rnn/TensorArrayUnstack/range/deltaConst^cond_1/switch_f*
value	B :*
dtype0*
_output_shapes
: 
р
#cond_1/rnn/TensorArrayUnstack/rangeRange)cond_1/rnn/TensorArrayUnstack/range/start+cond_1/rnn/TensorArrayUnstack/strided_slice)cond_1/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:         *

Tidx0
Ш
Econd_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3cond_1/rnn/TensorArray_1#cond_1/rnn/TensorArrayUnstack/rangecond_1/rnn/transposecond_1/rnn/TensorArray_1:1*
T0*'
_class
loc:@cond_1/rnn/transpose*
_output_shapes
: 
h
cond_1/rnn/Maximum/xConst^cond_1/switch_f*
_output_shapes
: *
value	B :*
dtype0
d
cond_1/rnn/MaximumMaximumcond_1/rnn/Maximum/xcond_1/rnn/Max*
T0*
_output_shapes
: 
n
cond_1/rnn/MinimumMinimumcond_1/rnn/strided_slice_1cond_1/rnn/Maximum*
T0*
_output_shapes
: 
v
"cond_1/rnn/while/iteration_counterConst^cond_1/switch_f*
value	B : *
dtype0*
_output_shapes
: 
Ы
cond_1/rnn/while/SwitchSwitchSqueezecond_1/pred_id*
T0*
_class
loc:@Squeeze*:
_output_shapes(
&:         :         
┬
cond_1/rnn/while/EnterEnter"cond_1/rnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *.

frame_name cond_1/rnn/while/while_context
▒
cond_1/rnn/while/Enter_1Entercond_1/rnn/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *.

frame_name cond_1/rnn/while/while_context
║
cond_1/rnn/while/Enter_2Entercond_1/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *.

frame_name cond_1/rnn/while/while_context
╩
cond_1/rnn/while/Enter_3Entercond_1/rnn/while/Switch*
T0*
is_constant( *
parallel_iterations *'
_output_shapes
:         *.

frame_name cond_1/rnn/while/while_context
Г
cond_1/rnn/while/MergeMergecond_1/rnn/while/Entercond_1/rnn/while/NextIteration*
N*
_output_shapes
: : *
T0
Й
cond_1/rnn/while/Merge_1Mergecond_1/rnn/while/Enter_1 cond_1/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
Й
cond_1/rnn/while/Merge_2Mergecond_1/rnn/while/Enter_2 cond_1/rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
Ъ
cond_1/rnn/while/Merge_3Mergecond_1/rnn/while/Enter_3 cond_1/rnn/while/NextIteration_3*
T0*
N*)
_output_shapes
:         : 
s
cond_1/rnn/while/LessLesscond_1/rnn/while/Mergecond_1/rnn/while/Less/Enter*
T0*
_output_shapes
: 
┐
cond_1/rnn/while/Less/EnterEntercond_1/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name cond_1/rnn/while/while_context
y
cond_1/rnn/while/Less_1Lesscond_1/rnn/while/Merge_1cond_1/rnn/while/Less_1/Enter*
_output_shapes
: *
T0
╣
cond_1/rnn/while/Less_1/EnterEntercond_1/rnn/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name cond_1/rnn/while/while_context
q
cond_1/rnn/while/LogicalAnd
LogicalAndcond_1/rnn/while/Lesscond_1/rnn/while/Less_1*
_output_shapes
: 
Z
cond_1/rnn/while/LoopCondLoopCondcond_1/rnn/while/LogicalAnd*
_output_shapes
: 
д
cond_1/rnn/while/Switch_1Switchcond_1/rnn/while/Mergecond_1/rnn/while/LoopCond*
T0*)
_class
loc:@cond_1/rnn/while/Merge*
_output_shapes
: : 
и
cond_1/rnn/while/Switch_2Switchcond_1/rnn/while/Merge_1cond_1/rnn/while/LoopCond*
T0*+
_class!
loc:@cond_1/rnn/while/Merge_1*
_output_shapes
: : 
и
cond_1/rnn/while/Switch_3Switchcond_1/rnn/while/Merge_2cond_1/rnn/while/LoopCond*
T0*+
_class!
loc:@cond_1/rnn/while/Merge_2*
_output_shapes
: : 
╩
cond_1/rnn/while/Switch_4Switchcond_1/rnn/while/Merge_3cond_1/rnn/while/LoopCond*
T0*+
_class!
loc:@cond_1/rnn/while/Merge_3*:
_output_shapes(
&:         :         
c
cond_1/rnn/while/IdentityIdentitycond_1/rnn/while/Switch_1:1*
_output_shapes
: *
T0
e
cond_1/rnn/while/Identity_1Identitycond_1/rnn/while/Switch_2:1*
T0*
_output_shapes
: 
e
cond_1/rnn/while/Identity_2Identitycond_1/rnn/while/Switch_3:1*
_output_shapes
: *
T0
v
cond_1/rnn/while/Identity_3Identitycond_1/rnn/while/Switch_4:1*
T0*'
_output_shapes
:         
t
cond_1/rnn/while/add/yConst^cond_1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
o
cond_1/rnn/while/addAddcond_1/rnn/while/Identitycond_1/rnn/while/add/y*
T0*
_output_shapes
: 
р
"cond_1/rnn/while/TensorArrayReadV3TensorArrayReadV3(cond_1/rnn/while/TensorArrayReadV3/Entercond_1/rnn/while/Identity_1*cond_1/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:         
╬
(cond_1/rnn/while/TensorArrayReadV3/EnterEntercond_1/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name cond_1/rnn/while/while_context
∙
*cond_1/rnn/while/TensorArrayReadV3/Enter_1EnterEcond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name cond_1/rnn/while/while_context
Э
cond_1/rnn/while/GreaterEqualGreaterEqualcond_1/rnn/while/Identity_1#cond_1/rnn/while/GreaterEqual/Enter*#
_output_shapes
:         *
T0
╨
#cond_1/rnn/while/GreaterEqual/EnterEntercond_1/rnn/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *#
_output_shapes
:         *.

frame_name cond_1/rnn/while/while_context
}
cond_1/rnn/while/ExpandDims/dimConst^cond_1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
й
cond_1/rnn/while/ExpandDims
ExpandDimscond_1/rnn/while/Identity_3cond_1/rnn/while/ExpandDims/dim*+
_output_shapes
:         *

Tdim0*
T0
О
cond_1/rnn/while/add_1Addcond_1/rnn/while/ExpandDimscond_1/rnn/while/add_1/Enter*
T0*+
_output_shapes
:         
├
cond_1/rnn/while/add_1/EnterEntercond_1/ExpandDims*"
_output_shapes
:*.

frame_name cond_1/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
Ь
6cond_1/rnn/while/ReduceLogSumExp/Max/reduction_indicesConst^cond_1/rnn/while/Identity*
valueB:*
dtype0*
_output_shapes
:
╬
$cond_1/rnn/while/ReduceLogSumExp/MaxMaxcond_1/rnn/while/add_16cond_1/rnn/while/ReduceLogSumExp/Max/reduction_indices*+
_output_shapes
:         *

Tidx0*
	keep_dims(*
T0
С
)cond_1/rnn/while/ReduceLogSumExp/IsFiniteIsFinite$cond_1/rnn/while/ReduceLogSumExp/Max*
T0*+
_output_shapes
:         
Ф
+cond_1/rnn/while/ReduceLogSumExp/zeros_like	ZerosLike$cond_1/rnn/while/ReduceLogSumExp/Max*
T0*+
_output_shapes
:         
х
'cond_1/rnn/while/ReduceLogSumExp/SelectSelect)cond_1/rnn/while/ReduceLogSumExp/IsFinite$cond_1/rnn/while/ReduceLogSumExp/Max+cond_1/rnn/while/ReduceLogSumExp/zeros_like*
T0*+
_output_shapes
:         
Ь
-cond_1/rnn/while/ReduceLogSumExp/StopGradientStopGradient'cond_1/rnn/while/ReduceLogSumExp/Select*
T0*+
_output_shapes
:         
и
$cond_1/rnn/while/ReduceLogSumExp/subSubcond_1/rnn/while/add_1-cond_1/rnn/while/ReduceLogSumExp/StopGradient*+
_output_shapes
:         *
T0
З
$cond_1/rnn/while/ReduceLogSumExp/ExpExp$cond_1/rnn/while/ReduceLogSumExp/sub*+
_output_shapes
:         *
T0
Ь
6cond_1/rnn/while/ReduceLogSumExp/Sum/reduction_indicesConst^cond_1/rnn/while/Identity*
dtype0*
_output_shapes
:*
valueB:
╪
$cond_1/rnn/while/ReduceLogSumExp/SumSum$cond_1/rnn/while/ReduceLogSumExp/Exp6cond_1/rnn/while/ReduceLogSumExp/Sum/reduction_indices*'
_output_shapes
:         *

Tidx0*
	keep_dims( *
T0
Г
$cond_1/rnn/while/ReduceLogSumExp/LogLog$cond_1/rnn/while/ReduceLogSumExp/Sum*
T0*'
_output_shapes
:         
К
&cond_1/rnn/while/ReduceLogSumExp/ShapeShape$cond_1/rnn/while/ReduceLogSumExp/Log*
T0*
out_type0*
_output_shapes
:
╩
(cond_1/rnn/while/ReduceLogSumExp/ReshapeReshape-cond_1/rnn/while/ReduceLogSumExp/StopGradient&cond_1/rnn/while/ReduceLogSumExp/Shape*
T0*
Tshape0*'
_output_shapes
:         
н
$cond_1/rnn/while/ReduceLogSumExp/addAdd$cond_1/rnn/while/ReduceLogSumExp/Log(cond_1/rnn/while/ReduceLogSumExp/Reshape*'
_output_shapes
:         *
T0
Щ
cond_1/rnn/while/add_2Add"cond_1/rnn/while/TensorArrayReadV3$cond_1/rnn/while/ReduceLogSumExp/add*
T0*'
_output_shapes
:         
╘
cond_1/rnn/while/SelectSelectcond_1/rnn/while/GreaterEqualcond_1/rnn/while/Select/Entercond_1/rnn/while/add_2*'
_output_shapes
:         *
T0*)
_class
loc:@cond_1/rnn/while/add_2
є
cond_1/rnn/while/Select/EnterEntercond_1/rnn/zeros*
is_constant(*'
_output_shapes
:         *.

frame_name cond_1/rnn/while/while_context*
T0*)
_class
loc:@cond_1/rnn/while/add_2*
parallel_iterations 
╘
cond_1/rnn/while/Select_1Selectcond_1/rnn/while/GreaterEqualcond_1/rnn/while/Identity_3cond_1/rnn/while/add_2*
T0*)
_class
loc:@cond_1/rnn/while/add_2*'
_output_shapes
:         
е
4cond_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3:cond_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Entercond_1/rnn/while/Identity_1cond_1/rnn/while/Selectcond_1/rnn/while/Identity_2*
T0*)
_class
loc:@cond_1/rnn/while/add_2*
_output_shapes
: 
Й
:cond_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntercond_1/rnn/TensorArray*
is_constant(*
_output_shapes
:*.

frame_name cond_1/rnn/while/while_context*
T0*)
_class
loc:@cond_1/rnn/while/add_2*
parallel_iterations 
v
cond_1/rnn/while/add_3/yConst^cond_1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
u
cond_1/rnn/while/add_3Addcond_1/rnn/while/Identity_1cond_1/rnn/while/add_3/y*
_output_shapes
: *
T0
f
cond_1/rnn/while/NextIterationNextIterationcond_1/rnn/while/add*
T0*
_output_shapes
: 
j
 cond_1/rnn/while/NextIteration_1NextIterationcond_1/rnn/while/add_3*
T0*
_output_shapes
: 
И
 cond_1/rnn/while/NextIteration_2NextIteration4cond_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
~
 cond_1/rnn/while/NextIteration_3NextIterationcond_1/rnn/while/Select_1*'
_output_shapes
:         *
T0
Y
cond_1/rnn/while/ExitExitcond_1/rnn/while/Switch_1*
T0*
_output_shapes
: 
[
cond_1/rnn/while/Exit_1Exitcond_1/rnn/while/Switch_2*
T0*
_output_shapes
: 
[
cond_1/rnn/while/Exit_2Exitcond_1/rnn/while/Switch_3*
T0*
_output_shapes
: 
l
cond_1/rnn/while/Exit_3Exitcond_1/rnn/while/Switch_4*
T0*'
_output_shapes
:         
╢
-cond_1/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3cond_1/rnn/TensorArraycond_1/rnn/while/Exit_2*
_output_shapes
: *)
_class
loc:@cond_1/rnn/TensorArray
ж
'cond_1/rnn/TensorArrayStack/range/startConst^cond_1/switch_f*
_output_shapes
: *
value	B : *)
_class
loc:@cond_1/rnn/TensorArray*
dtype0
ж
'cond_1/rnn/TensorArrayStack/range/deltaConst^cond_1/switch_f*
value	B :*)
_class
loc:@cond_1/rnn/TensorArray*
dtype0*
_output_shapes
: 
З
!cond_1/rnn/TensorArrayStack/rangeRange'cond_1/rnn/TensorArrayStack/range/start-cond_1/rnn/TensorArrayStack/TensorArraySizeV3'cond_1/rnn/TensorArrayStack/range/delta*#
_output_shapes
:         *

Tidx0*)
_class
loc:@cond_1/rnn/TensorArray
о
/cond_1/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3cond_1/rnn/TensorArray!cond_1/rnn/TensorArrayStack/rangecond_1/rnn/while/Exit_2*$
element_shape:         *)
_class
loc:@cond_1/rnn/TensorArray*
dtype0*4
_output_shapes"
 :                  
n
cond_1/rnn/Const_4Const^cond_1/switch_f*
valueB:*
dtype0*
_output_shapes
:
e
cond_1/rnn/Rank_1Const^cond_1/switch_f*
value	B :*
dtype0*
_output_shapes
: 
l
cond_1/rnn/range_1/startConst^cond_1/switch_f*
value	B :*
dtype0*
_output_shapes
: 
l
cond_1/rnn/range_1/deltaConst^cond_1/switch_f*
value	B :*
dtype0*
_output_shapes
: 
К
cond_1/rnn/range_1Rangecond_1/rnn/range_1/startcond_1/rnn/Rank_1cond_1/rnn/range_1/delta*

Tidx0*
_output_shapes
:

cond_1/rnn/concat_2/values_0Const^cond_1/switch_f*
valueB"       *
dtype0*
_output_shapes
:
l
cond_1/rnn/concat_2/axisConst^cond_1/switch_f*
value	B : *
dtype0*
_output_shapes
: 
е
cond_1/rnn/concat_2ConcatV2cond_1/rnn/concat_2/values_0cond_1/rnn/range_1cond_1/rnn/concat_2/axis*
N*
_output_shapes
:*

Tidx0*
T0
╡
cond_1/rnn/transpose_1	Transpose/cond_1/rnn/TensorArrayStack/TensorArrayGatherV3cond_1/rnn/concat_2*
Tperm0*
T0*4
_output_shapes"
 :                  
К
.cond_1/ReduceLogSumExp_1/Max/reduction_indicesConst^cond_1/switch_f*
valueB:*
dtype0*
_output_shapes
:
╗
cond_1/ReduceLogSumExp_1/MaxMaxcond_1/rnn/while/Exit_3.cond_1/ReduceLogSumExp_1/Max/reduction_indices*'
_output_shapes
:         *

Tidx0*
	keep_dims(*
T0
}
!cond_1/ReduceLogSumExp_1/IsFiniteIsFinitecond_1/ReduceLogSumExp_1/Max*
T0*'
_output_shapes
:         
А
#cond_1/ReduceLogSumExp_1/zeros_like	ZerosLikecond_1/ReduceLogSumExp_1/Max*'
_output_shapes
:         *
T0
┴
cond_1/ReduceLogSumExp_1/SelectSelect!cond_1/ReduceLogSumExp_1/IsFinitecond_1/ReduceLogSumExp_1/Max#cond_1/ReduceLogSumExp_1/zeros_like*'
_output_shapes
:         *
T0
И
%cond_1/ReduceLogSumExp_1/StopGradientStopGradientcond_1/ReduceLogSumExp_1/Select*'
_output_shapes
:         *
T0
Х
cond_1/ReduceLogSumExp_1/subSubcond_1/rnn/while/Exit_3%cond_1/ReduceLogSumExp_1/StopGradient*
T0*'
_output_shapes
:         
s
cond_1/ReduceLogSumExp_1/ExpExpcond_1/ReduceLogSumExp_1/sub*
T0*'
_output_shapes
:         
К
.cond_1/ReduceLogSumExp_1/Sum/reduction_indicesConst^cond_1/switch_f*
valueB:*
dtype0*
_output_shapes
:
╝
cond_1/ReduceLogSumExp_1/SumSumcond_1/ReduceLogSumExp_1/Exp.cond_1/ReduceLogSumExp_1/Sum/reduction_indices*
T0*#
_output_shapes
:         *

Tidx0*
	keep_dims( 
o
cond_1/ReduceLogSumExp_1/LogLogcond_1/ReduceLogSumExp_1/Sum*
T0*#
_output_shapes
:         
z
cond_1/ReduceLogSumExp_1/ShapeShapecond_1/ReduceLogSumExp_1/Log*
_output_shapes
:*
T0*
out_type0
о
 cond_1/ReduceLogSumExp_1/ReshapeReshape%cond_1/ReduceLogSumExp_1/StopGradientcond_1/ReduceLogSumExp_1/Shape*
T0*
Tshape0*#
_output_shapes
:         
С
cond_1/ReduceLogSumExp_1/addAddcond_1/ReduceLogSumExp_1/Log cond_1/ReduceLogSumExp_1/Reshape*#
_output_shapes
:         *
T0
И
cond_1/MergeMergecond_1/ReduceLogSumExp_1/addcond_1/ReduceLogSumExp/add*
T0*
N*%
_output_shapes
:         : 
G
subSub
cond/Mergecond_1/Merge*
T0*
_output_shapes
:
2
NegNegsub*
_output_shapes
:*
T0
2
RankRankNeg*
_output_shapes
: *
T0
M
range/startConst*
value	B : *
dtype0*
_output_shapes
: 
M
range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
_
rangeRangerange/startRankrange/delta*#
_output_shapes
:         *

Tidx0
V
MeanMeanNegrange*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
e
gradients/f_countConst^cond_1/switch_f*
dtype0*
_output_shapes
: *
value	B : 
о
gradients/f_count_1Entergradients/f_count*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *.

frame_name cond_1/rnn/while/while_context
r
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
T0*
N*
_output_shapes
: : 
i
gradients/SwitchSwitchgradients/Mergecond_1/rnn/while/LoopCond*
T0*
_output_shapes
: : 
m
gradients/Add/yConst^cond_1/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Z
gradients/AddAddgradients/Switch:1gradients/Add/y*
T0*
_output_shapes
: 
Є	
gradients/NextIterationNextIterationgradients/Add@^gradients/cond_1/rnn/while/Select_1_grad/zeros_like/StackPushV2<^gradients/cond_1/rnn/while/Select_1_grad/Select/StackPushV2H^gradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/StackPushV2J^gradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/StackPushV2_1b^gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPushV2V^gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/StackPushV2X^gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/StackPushV2_1L^gradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/StackPushV2K^gradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/StackPushV2N^gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/StackPushV2D^gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/StackPushV2V^gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/StackPushV2X^gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/StackPushV2_1H^gradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/StackPushV2?^gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/StackPushV2*
T0*
_output_shapes
: 
N
gradients/f_count_2Exitgradients/Switch*
_output_shapes
: *
T0
e
gradients/b_countConst^cond_1/switch_f*
value	B :*
dtype0*
_output_shapes
: 
║
gradients/b_count_1Entergradients/f_count_2*
_output_shapes
: *8

frame_name*(gradients/cond_1/rnn/while/while_context*
T0*
is_constant( *
parallel_iterations 
v
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
_output_shapes
: : *
T0*
N
x
gradients/GreaterEqualGreaterEqualgradients/Merge_1gradients/GreaterEqual/Enter*
_output_shapes
: *
T0
┴
gradients/GreaterEqual/EnterEntergradients/b_count*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *8

frame_name*(gradients/cond_1/rnn/while/while_context
O
gradients/b_count_2LoopCondgradients/GreaterEqual*
_output_shapes
: 
g
gradients/Switch_1Switchgradients/Merge_1gradients/b_count_2*
_output_shapes
: : *
T0
i
gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*
_output_shapes
: *
T0
Ч
gradients/NextIteration_1NextIterationgradients/Sub;^gradients/cond_1/rnn/while/Select_1_grad/zeros_like/b_sync*
T0*
_output_shapes
: 
P
gradients/b_count_3Exitgradients/Switch_1*
T0*
_output_shapes
: 
U
gradients/f_count_3Const*
value	B : *
dtype0*
_output_shapes
: 
╦
gradients/f_count_4Entergradients/f_count_3*
parallel_iterations *
_output_shapes
: *I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant( 
v
gradients/Merge_2Mergegradients/f_count_4gradients/NextIteration_2*
N*
_output_shapes
: : *
T0
И
gradients/Switch_2Switchgradients/Merge_24biLSTM_layers/bidirectional_rnn/fw/fw/while/LoopCond*
T0*
_output_shapes
: : 
К
gradients/Add_1/yConst5^biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
`
gradients/Add_1Addgradients/Switch_2:1gradients/Add_1/y*
T0*
_output_shapes
: 
Ё
gradients/NextIteration_2NextIterationgradients/Add_1}^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2[^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPushV2W^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPushV2[^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPushV2m^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2o^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1[^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPushV2]^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2m^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2o^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1k^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2m^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1[^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPushV2m^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2o^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1[^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPushV2]^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2k^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2a^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2_^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPushV2*
_output_shapes
: *
T0
P
gradients/f_count_5Exitgradients/Switch_2*
T0*
_output_shapes
: 
U
gradients/b_count_4Const*
value	B :*
dtype0*
_output_shapes
: 
╒
gradients/b_count_5Entergradients/f_count_5*
_output_shapes
: *S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant( *
parallel_iterations 
v
gradients/Merge_3Mergegradients/b_count_5gradients/NextIteration_3*
N*
_output_shapes
: : *
T0
|
gradients/GreaterEqual_1GreaterEqualgradients/Merge_3gradients/GreaterEqual_1/Enter*
T0*
_output_shapes
: 
р
gradients/GreaterEqual_1/EnterEntergradients/b_count_4*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
Q
gradients/b_count_6LoopCondgradients/GreaterEqual_1*
_output_shapes
: 
g
gradients/Switch_3Switchgradients/Merge_3gradients/b_count_6*
_output_shapes
: : *
T0
m
gradients/Sub_1Subgradients/Switch_3:1gradients/GreaterEqual_1/Enter*
T0*
_output_shapes
: 
╓
gradients/NextIteration_3NextIterationgradients/Sub_1x^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
_output_shapes
: *
T0
P
gradients/b_count_7Exitgradients/Switch_3*
_output_shapes
: *
T0
U
gradients/f_count_6Const*
value	B : *
dtype0*
_output_shapes
: 
╦
gradients/f_count_7Entergradients/f_count_6*
_output_shapes
: *I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant( *
parallel_iterations 
v
gradients/Merge_4Mergegradients/f_count_7gradients/NextIteration_4*
T0*
N*
_output_shapes
: : 
И
gradients/Switch_4Switchgradients/Merge_44biLSTM_layers/bidirectional_rnn/bw/bw/while/LoopCond*
_output_shapes
: : *
T0
К
gradients/Add_2/yConst5^biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
`
gradients/Add_2Addgradients/Switch_4:1gradients/Add_2/y*
T0*
_output_shapes
: 
Ё
gradients/NextIteration_4NextIterationgradients/Add_2}^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2[^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPushV2W^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPushV2[^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPushV2m^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2o^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1[^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPushV2]^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2m^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2o^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1k^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2m^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1[^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPushV2m^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2o^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1[^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPushV2]^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2k^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2a^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2_^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPushV2*
T0*
_output_shapes
: 
P
gradients/f_count_8Exitgradients/Switch_4*
T0*
_output_shapes
: 
U
gradients/b_count_8Const*
dtype0*
_output_shapes
: *
value	B :
╒
gradients/b_count_9Entergradients/f_count_8*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
v
gradients/Merge_5Mergegradients/b_count_9gradients/NextIteration_5*
T0*
N*
_output_shapes
: : 
|
gradients/GreaterEqual_2GreaterEqualgradients/Merge_5gradients/GreaterEqual_2/Enter*
_output_shapes
: *
T0
р
gradients/GreaterEqual_2/EnterEntergradients/b_count_8*
_output_shapes
: *S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant(*
parallel_iterations 
R
gradients/b_count_10LoopCondgradients/GreaterEqual_2*
_output_shapes
: 
h
gradients/Switch_5Switchgradients/Merge_5gradients/b_count_10*
_output_shapes
: : *
T0
m
gradients/Sub_2Subgradients/Switch_5:1gradients/GreaterEqual_2/Enter*
_output_shapes
: *
T0
╓
gradients/NextIteration_5NextIterationgradients/Sub_2x^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
_output_shapes
: *
T0
Q
gradients/b_count_11Exitgradients/Switch_5*
T0*
_output_shapes
: 
e
gradients/Mean_grad/ShapeShapeNeg*#
_output_shapes
:         *
T0*
out_type0
Ъ
gradients/Mean_grad/SizeSizegradients/Mean_grad/Shape*
T0*
out_type0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: 
Ы
gradients/Mean_grad/addAddrangegradients/Mean_grad/Size*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape*#
_output_shapes
:         
▓
gradients/Mean_grad/modFloorModgradients/Mean_grad/addgradients/Mean_grad/Size*#
_output_shapes
:         *
T0*,
_class"
 loc:@gradients/Mean_grad/Shape
а
gradients/Mean_grad/Shape_1Shapegradients/Mean_grad/mod*
T0*
out_type0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
:
П
gradients/Mean_grad/range/startConst*
_output_shapes
: *
value	B : *,
_class"
 loc:@gradients/Mean_grad/Shape*
dtype0
П
gradients/Mean_grad/range/deltaConst*
value	B :*,
_class"
 loc:@gradients/Mean_grad/Shape*
dtype0*
_output_shapes
: 
▌
gradients/Mean_grad/rangeRangegradients/Mean_grad/range/startgradients/Mean_grad/Sizegradients/Mean_grad/range/delta*#
_output_shapes
:         *

Tidx0*,
_class"
 loc:@gradients/Mean_grad/Shape
О
gradients/Mean_grad/Fill/valueConst*
_output_shapes
: *
value	B :*,
_class"
 loc:@gradients/Mean_grad/Shape*
dtype0
╦
gradients/Mean_grad/FillFillgradients/Mean_grad/Shape_1gradients/Mean_grad/Fill/value*#
_output_shapes
:         *
T0*

index_type0*,
_class"
 loc:@gradients/Mean_grad/Shape
А
!gradients/Mean_grad/DynamicStitchDynamicStitchgradients/Mean_grad/rangegradients/Mean_grad/modgradients/Mean_grad/Shapegradients/Mean_grad/Fill*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape*
N*#
_output_shapes
:         
Н
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*,
_class"
 loc:@gradients/Mean_grad/Shape
─
gradients/Mean_grad/MaximumMaximum!gradients/Mean_grad/DynamicStitchgradients/Mean_grad/Maximum/y*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape*#
_output_shapes
:         
╝
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Shapegradients/Mean_grad/Maximum*,
_class"
 loc:@gradients/Mean_grad/Shape*#
_output_shapes
:         *
T0
К
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
Р
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/floordiv*

Tmultiples0*
T0*
_output_shapes
:
g
gradients/Mean_grad/Shape_2ShapeNeg*
T0*
out_type0*#
_output_shapes
:         
^
gradients/Mean_grad/Shape_3Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ц
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_2gradients/Mean_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
e
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
Ъ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_3gradients/Mean_grad/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
a
gradients/Mean_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ж
gradients/Mean_grad/Maximum_1Maximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum_1/y*
T0*
_output_shapes
: 
Д
gradients/Mean_grad/floordiv_1FloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum_1*
_output_shapes
: *
T0
p
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
}
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*
_output_shapes
:
]
gradients/Neg_grad/NegNeggradients/Mean_grad/truediv*
T0*
_output_shapes
:
k
gradients/sub_grad/ShapeShape
cond/Merge*#
_output_shapes
:         *
T0*
out_type0
f
gradients/sub_grad/Shape_1Shapecond_1/Merge*
_output_shapes
:*
T0*
out_type0
┤
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Я
gradients/sub_grad/SumSumgradients/Neg_grad/Neg(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
И
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
г
gradients/sub_grad/Sum_1Sumgradients/Neg_grad/Neg*gradients/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0
Ч
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:         
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
╦
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*-
_class#
!loc:@gradients/sub_grad/Reshape*
_output_shapes
:*
T0
▄
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*#
_output_shapes
:         
╛
#gradients/cond/Merge_grad/cond_gradSwitch+gradients/sub_grad/tuple/control_dependencycond/pred_id*
_output_shapes

::*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
X
*gradients/cond/Merge_grad/tuple/group_depsNoOp$^gradients/cond/Merge_grad/cond_grad
э
2gradients/cond/Merge_grad/tuple/control_dependencyIdentity#gradients/cond/Merge_grad/cond_grad+^gradients/cond/Merge_grad/tuple/group_deps*#
_output_shapes
:         *
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
ц
4gradients/cond/Merge_grad/tuple/control_dependency_1Identity%gradients/cond/Merge_grad/cond_grad:1+^gradients/cond/Merge_grad/tuple/group_deps*-
_class#
!loc:@gradients/sub_grad/Reshape*
_output_shapes
:*
T0
▄
%gradients/cond_1/Merge_grad/cond_gradSwitch-gradients/sub_grad/tuple/control_dependency_1cond_1/pred_id*2
_output_shapes 
:         :         *
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
\
,gradients/cond_1/Merge_grad/tuple/group_depsNoOp&^gradients/cond_1/Merge_grad/cond_grad
ї
4gradients/cond_1/Merge_grad/tuple/control_dependencyIdentity%gradients/cond_1/Merge_grad/cond_grad-^gradients/cond_1/Merge_grad/tuple/group_deps*#
_output_shapes
:         *
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
∙
6gradients/cond_1/Merge_grad/tuple/control_dependency_1Identity'gradients/cond_1/Merge_grad/cond_grad:1-^gradients/cond_1/Merge_grad/tuple/group_deps*#
_output_shapes
:         *
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
g
gradients/cond/add_3_grad/ShapeShapecond/Sum*
T0*
out_type0*
_output_shapes
:
k
!gradients/cond/add_3_grad/Shape_1Shape
cond/Sum_1*
T0*
out_type0*
_output_shapes
:
╔
/gradients/cond/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/cond/add_3_grad/Shape!gradients/cond/add_3_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╔
gradients/cond/add_3_grad/SumSum2gradients/cond/Merge_grad/tuple/control_dependency/gradients/cond/add_3_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
и
!gradients/cond/add_3_grad/ReshapeReshapegradients/cond/add_3_grad/Sumgradients/cond/add_3_grad/Shape*#
_output_shapes
:         *
T0*
Tshape0
═
gradients/cond/add_3_grad/Sum_1Sum2gradients/cond/Merge_grad/tuple/control_dependency1gradients/cond/add_3_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
о
#gradients/cond/add_3_grad/Reshape_1Reshapegradients/cond/add_3_grad/Sum_1!gradients/cond/add_3_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:         
|
*gradients/cond/add_3_grad/tuple/group_depsNoOp"^gradients/cond/add_3_grad/Reshape$^gradients/cond/add_3_grad/Reshape_1
Є
2gradients/cond/add_3_grad/tuple/control_dependencyIdentity!gradients/cond/add_3_grad/Reshape+^gradients/cond/add_3_grad/tuple/group_deps*4
_class*
(&loc:@gradients/cond/add_3_grad/Reshape*#
_output_shapes
:         *
T0
°
4gradients/cond/add_3_grad/tuple/control_dependency_1Identity#gradients/cond/add_3_grad/Reshape_1+^gradients/cond/add_3_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/cond/add_3_grad/Reshape_1*#
_output_shapes
:         
n
"gradients/cond/GatherNd_grad/ShapeShapecond/Squeeze*
T0*
out_type0*
_output_shapes
:
▄
&gradients/cond/GatherNd_grad/ScatterNd	ScatterNdcond/concat4gradients/cond/Merge_grad/tuple/control_dependency_1"gradients/cond/GatherNd_grad/Shape*'
_output_shapes
:         *
Tindices0*
T0
Н
1gradients/cond_1/ReduceLogSumExp_1/add_grad/ShapeShapecond_1/ReduceLogSumExp_1/Log*
_output_shapes
:*
T0*
out_type0
У
3gradients/cond_1/ReduceLogSumExp_1/add_grad/Shape_1Shape cond_1/ReduceLogSumExp_1/Reshape*
T0*
out_type0*
_output_shapes
:
 
Agradients/cond_1/ReduceLogSumExp_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/cond_1/ReduceLogSumExp_1/add_grad/Shape3gradients/cond_1/ReduceLogSumExp_1/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
я
/gradients/cond_1/ReduceLogSumExp_1/add_grad/SumSum4gradients/cond_1/Merge_grad/tuple/control_dependencyAgradients/cond_1/ReduceLogSumExp_1/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
▐
3gradients/cond_1/ReduceLogSumExp_1/add_grad/ReshapeReshape/gradients/cond_1/ReduceLogSumExp_1/add_grad/Sum1gradients/cond_1/ReduceLogSumExp_1/add_grad/Shape*
T0*
Tshape0*#
_output_shapes
:         
є
1gradients/cond_1/ReduceLogSumExp_1/add_grad/Sum_1Sum4gradients/cond_1/Merge_grad/tuple/control_dependencyCgradients/cond_1/ReduceLogSumExp_1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ф
5gradients/cond_1/ReduceLogSumExp_1/add_grad/Reshape_1Reshape1gradients/cond_1/ReduceLogSumExp_1/add_grad/Sum_13gradients/cond_1/ReduceLogSumExp_1/add_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:         
▓
<gradients/cond_1/ReduceLogSumExp_1/add_grad/tuple/group_depsNoOp4^gradients/cond_1/ReduceLogSumExp_1/add_grad/Reshape6^gradients/cond_1/ReduceLogSumExp_1/add_grad/Reshape_1
║
Dgradients/cond_1/ReduceLogSumExp_1/add_grad/tuple/control_dependencyIdentity3gradients/cond_1/ReduceLogSumExp_1/add_grad/Reshape=^gradients/cond_1/ReduceLogSumExp_1/add_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/cond_1/ReduceLogSumExp_1/add_grad/Reshape*#
_output_shapes
:         
└
Fgradients/cond_1/ReduceLogSumExp_1/add_grad/tuple/control_dependency_1Identity5gradients/cond_1/ReduceLogSumExp_1/add_grad/Reshape_1=^gradients/cond_1/ReduceLogSumExp_1/add_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/cond_1/ReduceLogSumExp_1/add_grad/Reshape_1*#
_output_shapes
:         
Й
/gradients/cond_1/ReduceLogSumExp/add_grad/ShapeShapecond_1/ReduceLogSumExp/Log*
_output_shapes
:*
T0*
out_type0
П
1gradients/cond_1/ReduceLogSumExp/add_grad/Shape_1Shapecond_1/ReduceLogSumExp/Reshape*
_output_shapes
:*
T0*
out_type0
∙
?gradients/cond_1/ReduceLogSumExp/add_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/cond_1/ReduceLogSumExp/add_grad/Shape1gradients/cond_1/ReduceLogSumExp/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
э
-gradients/cond_1/ReduceLogSumExp/add_grad/SumSum6gradients/cond_1/Merge_grad/tuple/control_dependency_1?gradients/cond_1/ReduceLogSumExp/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
╪
1gradients/cond_1/ReduceLogSumExp/add_grad/ReshapeReshape-gradients/cond_1/ReduceLogSumExp/add_grad/Sum/gradients/cond_1/ReduceLogSumExp/add_grad/Shape*
T0*
Tshape0*#
_output_shapes
:         
ё
/gradients/cond_1/ReduceLogSumExp/add_grad/Sum_1Sum6gradients/cond_1/Merge_grad/tuple/control_dependency_1Agradients/cond_1/ReduceLogSumExp/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
▐
3gradients/cond_1/ReduceLogSumExp/add_grad/Reshape_1Reshape/gradients/cond_1/ReduceLogSumExp/add_grad/Sum_11gradients/cond_1/ReduceLogSumExp/add_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:         
м
:gradients/cond_1/ReduceLogSumExp/add_grad/tuple/group_depsNoOp2^gradients/cond_1/ReduceLogSumExp/add_grad/Reshape4^gradients/cond_1/ReduceLogSumExp/add_grad/Reshape_1
▓
Bgradients/cond_1/ReduceLogSumExp/add_grad/tuple/control_dependencyIdentity1gradients/cond_1/ReduceLogSumExp/add_grad/Reshape;^gradients/cond_1/ReduceLogSumExp/add_grad/tuple/group_deps*#
_output_shapes
:         *
T0*D
_class:
86loc:@gradients/cond_1/ReduceLogSumExp/add_grad/Reshape
╕
Dgradients/cond_1/ReduceLogSumExp/add_grad/tuple/control_dependency_1Identity3gradients/cond_1/ReduceLogSumExp/add_grad/Reshape_1;^gradients/cond_1/ReduceLogSumExp/add_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/cond_1/ReduceLogSumExp/add_grad/Reshape_1*#
_output_shapes
:         
g
gradients/cond/Sum_grad/ShapeShape
cond/mul_3*
_output_shapes
:*
T0*
out_type0
Р
gradients/cond/Sum_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :*0
_class&
$"loc:@gradients/cond/Sum_grad/Shape
п
gradients/cond/Sum_grad/addAddcond/Sum/reduction_indicesgradients/cond/Sum_grad/Size*0
_class&
$"loc:@gradients/cond/Sum_grad/Shape*
_output_shapes
: *
T0
╡
gradients/cond/Sum_grad/modFloorModgradients/cond/Sum_grad/addgradients/cond/Sum_grad/Size*
_output_shapes
: *
T0*0
_class&
$"loc:@gradients/cond/Sum_grad/Shape
Ф
gradients/cond/Sum_grad/Shape_1Const*
valueB *0
_class&
$"loc:@gradients/cond/Sum_grad/Shape*
dtype0*
_output_shapes
: 
Ч
#gradients/cond/Sum_grad/range/startConst*
value	B : *0
_class&
$"loc:@gradients/cond/Sum_grad/Shape*
dtype0*
_output_shapes
: 
Ч
#gradients/cond/Sum_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*0
_class&
$"loc:@gradients/cond/Sum_grad/Shape
ш
gradients/cond/Sum_grad/rangeRange#gradients/cond/Sum_grad/range/startgradients/cond/Sum_grad/Size#gradients/cond/Sum_grad/range/delta*0
_class&
$"loc:@gradients/cond/Sum_grad/Shape*
_output_shapes
:*

Tidx0
Ц
"gradients/cond/Sum_grad/Fill/valueConst*
value	B :*0
_class&
$"loc:@gradients/cond/Sum_grad/Shape*
dtype0*
_output_shapes
: 
╬
gradients/cond/Sum_grad/FillFillgradients/cond/Sum_grad/Shape_1"gradients/cond/Sum_grad/Fill/value*
T0*

index_type0*0
_class&
$"loc:@gradients/cond/Sum_grad/Shape*
_output_shapes
: 
Ш
%gradients/cond/Sum_grad/DynamicStitchDynamicStitchgradients/cond/Sum_grad/rangegradients/cond/Sum_grad/modgradients/cond/Sum_grad/Shapegradients/cond/Sum_grad/Fill*
T0*0
_class&
$"loc:@gradients/cond/Sum_grad/Shape*
N*#
_output_shapes
:         
Х
!gradients/cond/Sum_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :*0
_class&
$"loc:@gradients/cond/Sum_grad/Shape
╘
gradients/cond/Sum_grad/MaximumMaximum%gradients/cond/Sum_grad/DynamicStitch!gradients/cond/Sum_grad/Maximum/y*0
_class&
$"loc:@gradients/cond/Sum_grad/Shape*#
_output_shapes
:         *
T0
├
 gradients/cond/Sum_grad/floordivFloorDivgradients/cond/Sum_grad/Shapegradients/cond/Sum_grad/Maximum*
T0*0
_class&
$"loc:@gradients/cond/Sum_grad/Shape*
_output_shapes
:
╢
gradients/cond/Sum_grad/ReshapeReshape2gradients/cond/add_3_grad/tuple/control_dependency%gradients/cond/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
┤
gradients/cond/Sum_grad/TileTilegradients/cond/Sum_grad/Reshape gradients/cond/Sum_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:                  
i
gradients/cond/Sum_1_grad/ShapeShape
cond/mul_5*
T0*
out_type0*
_output_shapes
:
Ф
gradients/cond/Sum_1_grad/SizeConst*
value	B :*2
_class(
&$loc:@gradients/cond/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
╖
gradients/cond/Sum_1_grad/addAddcond/Sum_1/reduction_indicesgradients/cond/Sum_1_grad/Size*
T0*2
_class(
&$loc:@gradients/cond/Sum_1_grad/Shape*
_output_shapes
: 
╜
gradients/cond/Sum_1_grad/modFloorModgradients/cond/Sum_1_grad/addgradients/cond/Sum_1_grad/Size*
T0*2
_class(
&$loc:@gradients/cond/Sum_1_grad/Shape*
_output_shapes
: 
Ш
!gradients/cond/Sum_1_grad/Shape_1Const*
valueB *2
_class(
&$loc:@gradients/cond/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
Ы
%gradients/cond/Sum_1_grad/range/startConst*
_output_shapes
: *
value	B : *2
_class(
&$loc:@gradients/cond/Sum_1_grad/Shape*
dtype0
Ы
%gradients/cond/Sum_1_grad/range/deltaConst*
value	B :*2
_class(
&$loc:@gradients/cond/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
Є
gradients/cond/Sum_1_grad/rangeRange%gradients/cond/Sum_1_grad/range/startgradients/cond/Sum_1_grad/Size%gradients/cond/Sum_1_grad/range/delta*
_output_shapes
:*

Tidx0*2
_class(
&$loc:@gradients/cond/Sum_1_grad/Shape
Ъ
$gradients/cond/Sum_1_grad/Fill/valueConst*
value	B :*2
_class(
&$loc:@gradients/cond/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
╓
gradients/cond/Sum_1_grad/FillFill!gradients/cond/Sum_1_grad/Shape_1$gradients/cond/Sum_1_grad/Fill/value*
T0*

index_type0*2
_class(
&$loc:@gradients/cond/Sum_1_grad/Shape*
_output_shapes
: 
д
'gradients/cond/Sum_1_grad/DynamicStitchDynamicStitchgradients/cond/Sum_1_grad/rangegradients/cond/Sum_1_grad/modgradients/cond/Sum_1_grad/Shapegradients/cond/Sum_1_grad/Fill*
T0*2
_class(
&$loc:@gradients/cond/Sum_1_grad/Shape*
N*#
_output_shapes
:         
Щ
#gradients/cond/Sum_1_grad/Maximum/yConst*
value	B :*2
_class(
&$loc:@gradients/cond/Sum_1_grad/Shape*
dtype0*
_output_shapes
: 
▄
!gradients/cond/Sum_1_grad/MaximumMaximum'gradients/cond/Sum_1_grad/DynamicStitch#gradients/cond/Sum_1_grad/Maximum/y*#
_output_shapes
:         *
T0*2
_class(
&$loc:@gradients/cond/Sum_1_grad/Shape
╦
"gradients/cond/Sum_1_grad/floordivFloorDivgradients/cond/Sum_1_grad/Shape!gradients/cond/Sum_1_grad/Maximum*
T0*2
_class(
&$loc:@gradients/cond/Sum_1_grad/Shape*
_output_shapes
:
╝
!gradients/cond/Sum_1_grad/ReshapeReshape4gradients/cond/add_3_grad/tuple/control_dependency_1'gradients/cond/Sum_1_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
║
gradients/cond/Sum_1_grad/TileTile!gradients/cond/Sum_1_grad/Reshape"gradients/cond/Sum_1_grad/floordiv*

Tmultiples0*
T0*0
_output_shapes
:                  
t
!gradients/cond/Squeeze_grad/ShapeShapecond/Shape/Switch:1*
_output_shapes
:*
T0*
out_type0
╞
#gradients/cond/Squeeze_grad/ReshapeReshape&gradients/cond/GatherNd_grad/ScatterNd!gradients/cond/Squeeze_grad/Shape*4
_output_shapes"
 :                  *
T0*
Tshape0
Ъ
5gradients/cond_1/ReduceLogSumExp_1/Reshape_grad/ShapeShape%cond_1/ReduceLogSumExp_1/StopGradient*
T0*
out_type0*
_output_shapes
:
Б
7gradients/cond_1/ReduceLogSumExp_1/Reshape_grad/ReshapeReshapeFgradients/cond_1/ReduceLogSumExp_1/add_grad/tuple/control_dependency_15gradients/cond_1/ReduceLogSumExp_1/Reshape_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
Ц
3gradients/cond_1/ReduceLogSumExp/Reshape_grad/ShapeShape#cond_1/ReduceLogSumExp/StopGradient*
_output_shapes
:*
T0*
out_type0
√
5gradients/cond_1/ReduceLogSumExp/Reshape_grad/ReshapeReshapeDgradients/cond_1/ReduceLogSumExp/add_grad/tuple/control_dependency_13gradients/cond_1/ReduceLogSumExp/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
m
gradients/cond/mul_3_grad/ShapeShapecond/Reshape_3*
T0*
out_type0*
_output_shapes
:
y
!gradients/cond/mul_3_grad/Shape_1Shapecond/SequenceMask/Cast_1*
T0*
out_type0*
_output_shapes
:
╔
/gradients/cond/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/cond/mul_3_grad/Shape!gradients/cond/mul_3_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ч
gradients/cond/mul_3_grad/MulMulgradients/cond/Sum_grad/Tilecond/SequenceMask/Cast_1*
T0*0
_output_shapes
:                  
┤
gradients/cond/mul_3_grad/SumSumgradients/cond/mul_3_grad/Mul/gradients/cond/mul_3_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
╡
!gradients/cond/mul_3_grad/ReshapeReshapegradients/cond/mul_3_grad/Sumgradients/cond/mul_3_grad/Shape*0
_output_shapes
:                  *
T0*
Tshape0
П
gradients/cond/mul_3_grad/Mul_1Mulcond/Reshape_3gradients/cond/Sum_grad/Tile*
T0*0
_output_shapes
:                  
║
gradients/cond/mul_3_grad/Sum_1Sumgradients/cond/mul_3_grad/Mul_11gradients/cond/mul_3_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
╗
#gradients/cond/mul_3_grad/Reshape_1Reshapegradients/cond/mul_3_grad/Sum_1!gradients/cond/mul_3_grad/Shape_1*
T0*
Tshape0*0
_output_shapes
:                  
|
*gradients/cond/mul_3_grad/tuple/group_depsNoOp"^gradients/cond/mul_3_grad/Reshape$^gradients/cond/mul_3_grad/Reshape_1
 
2gradients/cond/mul_3_grad/tuple/control_dependencyIdentity!gradients/cond/mul_3_grad/Reshape+^gradients/cond/mul_3_grad/tuple/group_deps*0
_output_shapes
:                  *
T0*4
_class*
(&loc:@gradients/cond/mul_3_grad/Reshape
Е
4gradients/cond/mul_3_grad/tuple/control_dependency_1Identity#gradients/cond/mul_3_grad/Reshape_1+^gradients/cond/mul_3_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/cond/mul_3_grad/Reshape_1*0
_output_shapes
:                  
l
gradients/cond/mul_5_grad/ShapeShapecond/Gather_1*
out_type0*
_output_shapes
:*
T0
m
!gradients/cond/mul_5_grad/Shape_1Shapecond/Slice_2*
T0*
out_type0*
_output_shapes
:
╔
/gradients/cond/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/cond/mul_5_grad/Shape!gradients/cond/mul_5_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Н
gradients/cond/mul_5_grad/MulMulgradients/cond/Sum_1_grad/Tilecond/Slice_2*
T0*0
_output_shapes
:                  
┤
gradients/cond/mul_5_grad/SumSumgradients/cond/mul_5_grad/Mul/gradients/cond/mul_5_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
╡
!gradients/cond/mul_5_grad/ReshapeReshapegradients/cond/mul_5_grad/Sumgradients/cond/mul_5_grad/Shape*
T0*
Tshape0*0
_output_shapes
:                  
Р
gradients/cond/mul_5_grad/Mul_1Mulcond/Gather_1gradients/cond/Sum_1_grad/Tile*
T0*0
_output_shapes
:                  
║
gradients/cond/mul_5_grad/Sum_1Sumgradients/cond/mul_5_grad/Mul_11gradients/cond/mul_5_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
╗
#gradients/cond/mul_5_grad/Reshape_1Reshapegradients/cond/mul_5_grad/Sum_1!gradients/cond/mul_5_grad/Shape_1*
T0*
Tshape0*0
_output_shapes
:                  
|
*gradients/cond/mul_5_grad/tuple/group_depsNoOp"^gradients/cond/mul_5_grad/Reshape$^gradients/cond/mul_5_grad/Reshape_1
 
2gradients/cond/mul_5_grad/tuple/control_dependencyIdentity!gradients/cond/mul_5_grad/Reshape+^gradients/cond/mul_5_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/cond/mul_5_grad/Reshape*0
_output_shapes
:                  
Е
4gradients/cond/mul_5_grad/tuple/control_dependency_1Identity#gradients/cond/mul_5_grad/Reshape_1+^gradients/cond/mul_5_grad/tuple/group_deps*0
_output_shapes
:                  *
T0*6
_class,
*(loc:@gradients/cond/mul_5_grad/Reshape_1
n
#gradients/cond/Reshape_3_grad/ShapeShapecond/Gather*
T0*
out_type0*
_output_shapes
:
┼
%gradients/cond/Reshape_3_grad/ReshapeReshape2gradients/cond/mul_3_grad/tuple/control_dependency#gradients/cond/Reshape_3_grad/Shape*#
_output_shapes
:         *
T0*
Tshape0
П
"gradients/cond/Gather_1_grad/ShapeConst*
valueB	R1*!
_class
loc:@cond/Reshape_4*
dtype0	*
_output_shapes
:
з
$gradients/cond/Gather_1_grad/ToInt32Cast"gradients/cond/Gather_1_grad/Shape*

SrcT0	*!
_class
loc:@cond/Reshape_4*
_output_shapes
:*

DstT0
f
!gradients/cond/Gather_1_grad/SizeSize
cond/add_2*
T0*
out_type0*
_output_shapes
: 
m
+gradients/cond/Gather_1_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
╢
'gradients/cond/Gather_1_grad/ExpandDims
ExpandDims!gradients/cond/Gather_1_grad/Size+gradients/cond/Gather_1_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
z
0gradients/cond/Gather_1_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
|
2gradients/cond/Gather_1_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
|
2gradients/cond/Gather_1_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
О
*gradients/cond/Gather_1_grad/strided_sliceStridedSlice$gradients/cond/Gather_1_grad/ToInt320gradients/cond/Gather_1_grad/strided_slice/stack2gradients/cond/Gather_1_grad/strided_slice/stack_12gradients/cond/Gather_1_grad/strided_slice/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes
: *
Index0*
T0
j
(gradients/cond/Gather_1_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ш
#gradients/cond/Gather_1_grad/concatConcatV2'gradients/cond/Gather_1_grad/ExpandDims*gradients/cond/Gather_1_grad/strided_slice(gradients/cond/Gather_1_grad/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
─
$gradients/cond/Gather_1_grad/ReshapeReshape2gradients/cond/mul_5_grad/tuple/control_dependency#gradients/cond/Gather_1_grad/concat*
T0*
Tshape0*#
_output_shapes
:         
в
&gradients/cond/Gather_1_grad/Reshape_1Reshape
cond/add_2'gradients/cond/Gather_1_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:         
b
 gradients/cond/Slice_2_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
m
!gradients/cond/Slice_2_grad/ShapeShapecond/Slice_2*
_output_shapes
:*
T0*
out_type0
e
#gradients/cond/Slice_2_grad/stack/1Const*
value	B :*
dtype0*
_output_shapes
: 
к
!gradients/cond/Slice_2_grad/stackPack gradients/cond/Slice_2_grad/Rank#gradients/cond/Slice_2_grad/stack/1*
T0*

axis *
N*
_output_shapes
:
Ь
#gradients/cond/Slice_2_grad/ReshapeReshapecond/Slice_2/begin!gradients/cond/Slice_2_grad/stack*
T0*
Tshape0*
_output_shapes

:
}
#gradients/cond/Slice_2_grad/Shape_1Shapecond/SequenceMask_1/Cast_1*
_output_shapes
:*
T0*
out_type0
У
gradients/cond/Slice_2_grad/subSub#gradients/cond/Slice_2_grad/Shape_1!gradients/cond/Slice_2_grad/Shape*
T0*
_output_shapes
:
В
!gradients/cond/Slice_2_grad/sub_1Subgradients/cond/Slice_2_grad/subcond/Slice_2/begin*
_output_shapes
:*
T0
н
%gradients/cond/Slice_2_grad/Reshape_1Reshape!gradients/cond/Slice_2_grad/sub_1!gradients/cond/Slice_2_grad/stack*
T0*
Tshape0*
_output_shapes

:
i
'gradients/cond/Slice_2_grad/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
с
"gradients/cond/Slice_2_grad/concatConcatV2#gradients/cond/Slice_2_grad/Reshape%gradients/cond/Slice_2_grad/Reshape_1'gradients/cond/Slice_2_grad/concat/axis*
N*
_output_shapes

:*

Tidx0*
T0
╠
gradients/cond/Slice_2_grad/PadPad4gradients/cond/mul_5_grad/tuple/control_dependency_1"gradients/cond/Slice_2_grad/concat*
T0*
	Tpaddings0*0
_output_shapes
:                  
╫
6gradients/cond_1/ReduceLogSumExp_1/Log_grad/Reciprocal
Reciprocalcond_1/ReduceLogSumExp_1/SumE^gradients/cond_1/ReduceLogSumExp_1/add_grad/tuple/control_dependency*#
_output_shapes
:         *
T0
т
/gradients/cond_1/ReduceLogSumExp_1/Log_grad/mulMulDgradients/cond_1/ReduceLogSumExp_1/add_grad/tuple/control_dependency6gradients/cond_1/ReduceLogSumExp_1/Log_grad/Reciprocal*#
_output_shapes
:         *
T0
╤
4gradients/cond_1/ReduceLogSumExp/Log_grad/Reciprocal
Reciprocalcond_1/ReduceLogSumExp/SumC^gradients/cond_1/ReduceLogSumExp/add_grad/tuple/control_dependency*#
_output_shapes
:         *
T0
▄
-gradients/cond_1/ReduceLogSumExp/Log_grad/mulMulBgradients/cond_1/ReduceLogSumExp/add_grad/tuple/control_dependency4gradients/cond_1/ReduceLogSumExp/Log_grad/Reciprocal*#
_output_shapes
:         *
T0
С
 gradients/cond/Gather_grad/ShapeShapecond/Reshape_1*
_output_shapes
:*
T0*
out_type0	*!
_class
loc:@cond/Reshape_1
г
"gradients/cond/Gather_grad/ToInt32Cast gradients/cond/Gather_grad/Shape*

SrcT0	*!
_class
loc:@cond/Reshape_1*
_output_shapes
:*

DstT0
h
gradients/cond/Gather_grad/SizeSizecond/Reshape_2*
T0*
out_type0*
_output_shapes
: 
k
)gradients/cond/Gather_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
░
%gradients/cond/Gather_grad/ExpandDims
ExpandDimsgradients/cond/Gather_grad/Size)gradients/cond/Gather_grad/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
x
.gradients/cond/Gather_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
z
0gradients/cond/Gather_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
z
0gradients/cond/Gather_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Д
(gradients/cond/Gather_grad/strided_sliceStridedSlice"gradients/cond/Gather_grad/ToInt32.gradients/cond/Gather_grad/strided_slice/stack0gradients/cond/Gather_grad/strided_slice/stack_10gradients/cond/Gather_grad/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: *
Index0*
T0
h
&gradients/cond/Gather_grad/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
р
!gradients/cond/Gather_grad/concatConcatV2%gradients/cond/Gather_grad/ExpandDims(gradients/cond/Gather_grad/strided_slice&gradients/cond/Gather_grad/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
│
"gradients/cond/Gather_grad/ReshapeReshape%gradients/cond/Reshape_3_grad/Reshape!gradients/cond/Gather_grad/concat*#
_output_shapes
:         *
T0*
Tshape0
в
$gradients/cond/Gather_grad/Reshape_1Reshapecond/Reshape_2%gradients/cond/Gather_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:         
t
#gradients/cond/Reshape_4_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
Г
9gradients/cond/Reshape_4_grad/Reshape/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Е
;gradients/cond/Reshape_4_grad/Reshape/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Е
;gradients/cond/Reshape_4_grad/Reshape/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
░
3gradients/cond/Reshape_4_grad/Reshape/strided_sliceStridedSlice$gradients/cond/Gather_1_grad/ToInt329gradients/cond/Reshape_4_grad/Reshape/strided_slice/stack;gradients/cond/Reshape_4_grad/Reshape/strided_slice/stack_1;gradients/cond/Reshape_4_grad/Reshape/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
Ч
,gradients/cond/Reshape_4_grad/Reshape/tensorUnsortedSegmentSum$gradients/cond/Gather_1_grad/Reshape&gradients/cond/Gather_1_grad/Reshape_13gradients/cond/Reshape_4_grad/Reshape/strided_slice*
Tnumsegments0*
Tindices0*
T0*#
_output_shapes
:         
║
%gradients/cond/Reshape_4_grad/ReshapeReshape,gradients/cond/Reshape_4_grad/Reshape/tensor#gradients/cond/Reshape_4_grad/Shape*
T0*
Tshape0*
_output_shapes

:
Н
1gradients/cond_1/ReduceLogSumExp_1/Sum_grad/ShapeShapecond_1/ReduceLogSumExp_1/Exp*
_output_shapes
:*
T0*
out_type0
╕
0gradients/cond_1/ReduceLogSumExp_1/Sum_grad/SizeConst*
value	B :*D
_class:
86loc:@gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Shape*
dtype0*
_output_shapes
: 
Г
/gradients/cond_1/ReduceLogSumExp_1/Sum_grad/addAdd.cond_1/ReduceLogSumExp_1/Sum/reduction_indices0gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Size*
_output_shapes
:*
T0*D
_class:
86loc:@gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Shape
Й
/gradients/cond_1/ReduceLogSumExp_1/Sum_grad/modFloorMod/gradients/cond_1/ReduceLogSumExp_1/Sum_grad/add0gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Size*
_output_shapes
:*
T0*D
_class:
86loc:@gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Shape
├
3gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Shape_1Const*
valueB:*D
_class:
86loc:@gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Shape*
dtype0*
_output_shapes
:
┐
7gradients/cond_1/ReduceLogSumExp_1/Sum_grad/range/startConst*
value	B : *D
_class:
86loc:@gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Shape*
dtype0*
_output_shapes
: 
┐
7gradients/cond_1/ReduceLogSumExp_1/Sum_grad/range/deltaConst*
value	B :*D
_class:
86loc:@gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Shape*
dtype0*
_output_shapes
: 
╠
1gradients/cond_1/ReduceLogSumExp_1/Sum_grad/rangeRange7gradients/cond_1/ReduceLogSumExp_1/Sum_grad/range/start0gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Size7gradients/cond_1/ReduceLogSumExp_1/Sum_grad/range/delta*D
_class:
86loc:@gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Shape*
_output_shapes
:*

Tidx0
╛
6gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*D
_class:
86loc:@gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Shape
в
0gradients/cond_1/ReduceLogSumExp_1/Sum_grad/FillFill3gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Shape_16gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Fill/value*
T0*

index_type0*D
_class:
86loc:@gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Shape*
_output_shapes
:
Р
9gradients/cond_1/ReduceLogSumExp_1/Sum_grad/DynamicStitchDynamicStitch1gradients/cond_1/ReduceLogSumExp_1/Sum_grad/range/gradients/cond_1/ReduceLogSumExp_1/Sum_grad/mod1gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Shape0gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Fill*D
_class:
86loc:@gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Shape*
N*#
_output_shapes
:         *
T0
╜
5gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Maximum/yConst*
value	B :*D
_class:
86loc:@gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Shape*
dtype0*
_output_shapes
: 
д
3gradients/cond_1/ReduceLogSumExp_1/Sum_grad/MaximumMaximum9gradients/cond_1/ReduceLogSumExp_1/Sum_grad/DynamicStitch5gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Maximum/y*
T0*D
_class:
86loc:@gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Shape*#
_output_shapes
:         
У
4gradients/cond_1/ReduceLogSumExp_1/Sum_grad/floordivFloorDiv1gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Shape3gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Maximum*
T0*D
_class:
86loc:@gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Shape*
_output_shapes
:
█
3gradients/cond_1/ReduceLogSumExp_1/Sum_grad/ReshapeReshape/gradients/cond_1/ReduceLogSumExp_1/Log_grad/mul9gradients/cond_1/ReduceLogSumExp_1/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
ч
0gradients/cond_1/ReduceLogSumExp_1/Sum_grad/TileTile3gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Reshape4gradients/cond_1/ReduceLogSumExp_1/Sum_grad/floordiv*
T0*'
_output_shapes
:         *

Tmultiples0
Й
/gradients/cond_1/ReduceLogSumExp/Sum_grad/ShapeShapecond_1/ReduceLogSumExp/Exp*
T0*
out_type0*
_output_shapes
:
┤
.gradients/cond_1/ReduceLogSumExp/Sum_grad/SizeConst*
value	B :*B
_class8
64loc:@gradients/cond_1/ReduceLogSumExp/Sum_grad/Shape*
dtype0*
_output_shapes
: 
√
-gradients/cond_1/ReduceLogSumExp/Sum_grad/addAdd,cond_1/ReduceLogSumExp/Sum/reduction_indices.gradients/cond_1/ReduceLogSumExp/Sum_grad/Size*
_output_shapes
:*
T0*B
_class8
64loc:@gradients/cond_1/ReduceLogSumExp/Sum_grad/Shape
Б
-gradients/cond_1/ReduceLogSumExp/Sum_grad/modFloorMod-gradients/cond_1/ReduceLogSumExp/Sum_grad/add.gradients/cond_1/ReduceLogSumExp/Sum_grad/Size*
T0*B
_class8
64loc:@gradients/cond_1/ReduceLogSumExp/Sum_grad/Shape*
_output_shapes
:
┐
1gradients/cond_1/ReduceLogSumExp/Sum_grad/Shape_1Const*
valueB:*B
_class8
64loc:@gradients/cond_1/ReduceLogSumExp/Sum_grad/Shape*
dtype0*
_output_shapes
:
╗
5gradients/cond_1/ReduceLogSumExp/Sum_grad/range/startConst*
value	B : *B
_class8
64loc:@gradients/cond_1/ReduceLogSumExp/Sum_grad/Shape*
dtype0*
_output_shapes
: 
╗
5gradients/cond_1/ReduceLogSumExp/Sum_grad/range/deltaConst*
value	B :*B
_class8
64loc:@gradients/cond_1/ReduceLogSumExp/Sum_grad/Shape*
dtype0*
_output_shapes
: 
┬
/gradients/cond_1/ReduceLogSumExp/Sum_grad/rangeRange5gradients/cond_1/ReduceLogSumExp/Sum_grad/range/start.gradients/cond_1/ReduceLogSumExp/Sum_grad/Size5gradients/cond_1/ReduceLogSumExp/Sum_grad/range/delta*

Tidx0*B
_class8
64loc:@gradients/cond_1/ReduceLogSumExp/Sum_grad/Shape*
_output_shapes
:
║
4gradients/cond_1/ReduceLogSumExp/Sum_grad/Fill/valueConst*
value	B :*B
_class8
64loc:@gradients/cond_1/ReduceLogSumExp/Sum_grad/Shape*
dtype0*
_output_shapes
: 
Ъ
.gradients/cond_1/ReduceLogSumExp/Sum_grad/FillFill1gradients/cond_1/ReduceLogSumExp/Sum_grad/Shape_14gradients/cond_1/ReduceLogSumExp/Sum_grad/Fill/value*
_output_shapes
:*
T0*

index_type0*B
_class8
64loc:@gradients/cond_1/ReduceLogSumExp/Sum_grad/Shape
Д
7gradients/cond_1/ReduceLogSumExp/Sum_grad/DynamicStitchDynamicStitch/gradients/cond_1/ReduceLogSumExp/Sum_grad/range-gradients/cond_1/ReduceLogSumExp/Sum_grad/mod/gradients/cond_1/ReduceLogSumExp/Sum_grad/Shape.gradients/cond_1/ReduceLogSumExp/Sum_grad/Fill*B
_class8
64loc:@gradients/cond_1/ReduceLogSumExp/Sum_grad/Shape*
N*#
_output_shapes
:         *
T0
╣
3gradients/cond_1/ReduceLogSumExp/Sum_grad/Maximum/yConst*
value	B :*B
_class8
64loc:@gradients/cond_1/ReduceLogSumExp/Sum_grad/Shape*
dtype0*
_output_shapes
: 
Ь
1gradients/cond_1/ReduceLogSumExp/Sum_grad/MaximumMaximum7gradients/cond_1/ReduceLogSumExp/Sum_grad/DynamicStitch3gradients/cond_1/ReduceLogSumExp/Sum_grad/Maximum/y*
T0*B
_class8
64loc:@gradients/cond_1/ReduceLogSumExp/Sum_grad/Shape*#
_output_shapes
:         
Л
2gradients/cond_1/ReduceLogSumExp/Sum_grad/floordivFloorDiv/gradients/cond_1/ReduceLogSumExp/Sum_grad/Shape1gradients/cond_1/ReduceLogSumExp/Sum_grad/Maximum*
T0*B
_class8
64loc:@gradients/cond_1/ReduceLogSumExp/Sum_grad/Shape*
_output_shapes
:
╒
1gradients/cond_1/ReduceLogSumExp/Sum_grad/ReshapeReshape-gradients/cond_1/ReduceLogSumExp/Log_grad/mul7gradients/cond_1/ReduceLogSumExp/Sum_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
с
.gradients/cond_1/ReduceLogSumExp/Sum_grad/TileTile1gradients/cond_1/ReduceLogSumExp/Sum_grad/Reshape2gradients/cond_1/ReduceLogSumExp/Sum_grad/floordiv*'
_output_shapes
:         *

Tmultiples0*
T0
v
#gradients/cond/Reshape_1_grad/ShapeShapecond/Shape_1/Switch*
T0*
out_type0*
_output_shapes
:
Г
9gradients/cond/Reshape_1_grad/Reshape/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Е
;gradients/cond/Reshape_1_grad/Reshape/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Е
;gradients/cond/Reshape_1_grad/Reshape/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
о
3gradients/cond/Reshape_1_grad/Reshape/strided_sliceStridedSlice"gradients/cond/Gather_grad/ToInt329gradients/cond/Reshape_1_grad/Reshape/strided_slice/stack;gradients/cond/Reshape_1_grad/Reshape/strided_slice/stack_1;gradients/cond/Reshape_1_grad/Reshape/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
У
,gradients/cond/Reshape_1_grad/Reshape/tensorUnsortedSegmentSum"gradients/cond/Gather_grad/Reshape$gradients/cond/Gather_grad/Reshape_13gradients/cond/Reshape_1_grad/Reshape/strided_slice*#
_output_shapes
:         *
Tnumsegments0*
Tindices0*
T0
╨
%gradients/cond/Reshape_1_grad/ReshapeReshape,gradients/cond/Reshape_1_grad/Reshape/tensor#gradients/cond/Reshape_1_grad/Shape*
T0*
Tshape0*4
_output_shapes"
 :                  
o
gradients/Switch_6Switchtransitions/readcond/pred_id*(
_output_shapes
::*
T0
e
gradients/Shape_1Shapegradients/Switch_6:1*
T0*
out_type0*
_output_shapes
:
Z
gradients/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
|
gradients/zerosFillgradients/Shape_1gradients/zeros/Const*

index_type0*
_output_shapes

:*
T0
г
.gradients/cond/Reshape_4/Switch_grad/cond_gradMerge%gradients/cond/Reshape_4_grad/Reshapegradients/zeros*
T0*
N* 
_output_shapes
:: 
╕
/gradients/cond_1/ReduceLogSumExp_1/Exp_grad/mulMul0gradients/cond_1/ReduceLogSumExp_1/Sum_grad/Tilecond_1/ReduceLogSumExp_1/Exp*
T0*'
_output_shapes
:         
▓
-gradients/cond_1/ReduceLogSumExp/Exp_grad/mulMul.gradients/cond_1/ReduceLogSumExp/Sum_grad/Tilecond_1/ReduceLogSumExp/Exp*
T0*'
_output_shapes
:         
в
gradients/Switch_7SwitchbiLSTM_layers/Reshape_1cond/pred_id*T
_output_shapesB
@:                  :                  *
T0
c
gradients/Shape_2Shapegradients/Switch_7*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ц
gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*
T0*

index_type0*4
_output_shapes"
 :                  
╡
*gradients/cond/Shape/Switch_grad/cond_gradMergegradients/zeros_1#gradients/cond/Squeeze_grad/Reshape*
T0*
N*6
_output_shapes$
":                  : 
И
1gradients/cond_1/ReduceLogSumExp_1/sub_grad/ShapeShapecond_1/rnn/while/Exit_3*
_output_shapes
:*
T0*
out_type0
Ш
3gradients/cond_1/ReduceLogSumExp_1/sub_grad/Shape_1Shape%cond_1/ReduceLogSumExp_1/StopGradient*
T0*
out_type0*
_output_shapes
:
 
Agradients/cond_1/ReduceLogSumExp_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/cond_1/ReduceLogSumExp_1/sub_grad/Shape3gradients/cond_1/ReduceLogSumExp_1/sub_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ъ
/gradients/cond_1/ReduceLogSumExp_1/sub_grad/SumSum/gradients/cond_1/ReduceLogSumExp_1/Exp_grad/mulAgradients/cond_1/ReduceLogSumExp_1/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
т
3gradients/cond_1/ReduceLogSumExp_1/sub_grad/ReshapeReshape/gradients/cond_1/ReduceLogSumExp_1/sub_grad/Sum1gradients/cond_1/ReduceLogSumExp_1/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
ю
1gradients/cond_1/ReduceLogSumExp_1/sub_grad/Sum_1Sum/gradients/cond_1/ReduceLogSumExp_1/Exp_grad/mulCgradients/cond_1/ReduceLogSumExp_1/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
М
/gradients/cond_1/ReduceLogSumExp_1/sub_grad/NegNeg1gradients/cond_1/ReduceLogSumExp_1/sub_grad/Sum_1*
T0*
_output_shapes
:
ц
5gradients/cond_1/ReduceLogSumExp_1/sub_grad/Reshape_1Reshape/gradients/cond_1/ReduceLogSumExp_1/sub_grad/Neg3gradients/cond_1/ReduceLogSumExp_1/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
▓
<gradients/cond_1/ReduceLogSumExp_1/sub_grad/tuple/group_depsNoOp4^gradients/cond_1/ReduceLogSumExp_1/sub_grad/Reshape6^gradients/cond_1/ReduceLogSumExp_1/sub_grad/Reshape_1
╛
Dgradients/cond_1/ReduceLogSumExp_1/sub_grad/tuple/control_dependencyIdentity3gradients/cond_1/ReduceLogSumExp_1/sub_grad/Reshape=^gradients/cond_1/ReduceLogSumExp_1/sub_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/cond_1/ReduceLogSumExp_1/sub_grad/Reshape*'
_output_shapes
:         
─
Fgradients/cond_1/ReduceLogSumExp_1/sub_grad/tuple/control_dependency_1Identity5gradients/cond_1/ReduceLogSumExp_1/sub_grad/Reshape_1=^gradients/cond_1/ReduceLogSumExp_1/sub_grad/tuple/group_deps*'
_output_shapes
:         *
T0*H
_class>
<:loc:@gradients/cond_1/ReduceLogSumExp_1/sub_grad/Reshape_1
Т
/gradients/cond_1/ReduceLogSumExp/sub_grad/ShapeShape#cond_1/ReduceLogSumExp/Max/Switch:1*
T0*
out_type0*
_output_shapes
:
Ф
1gradients/cond_1/ReduceLogSumExp/sub_grad/Shape_1Shape#cond_1/ReduceLogSumExp/StopGradient*
_output_shapes
:*
T0*
out_type0
∙
?gradients/cond_1/ReduceLogSumExp/sub_grad/BroadcastGradientArgsBroadcastGradientArgs/gradients/cond_1/ReduceLogSumExp/sub_grad/Shape1gradients/cond_1/ReduceLogSumExp/sub_grad/Shape_1*
T0*2
_output_shapes 
:         :         
ф
-gradients/cond_1/ReduceLogSumExp/sub_grad/SumSum-gradients/cond_1/ReduceLogSumExp/Exp_grad/mul?gradients/cond_1/ReduceLogSumExp/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
▄
1gradients/cond_1/ReduceLogSumExp/sub_grad/ReshapeReshape-gradients/cond_1/ReduceLogSumExp/sub_grad/Sum/gradients/cond_1/ReduceLogSumExp/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
ш
/gradients/cond_1/ReduceLogSumExp/sub_grad/Sum_1Sum-gradients/cond_1/ReduceLogSumExp/Exp_grad/mulAgradients/cond_1/ReduceLogSumExp/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
И
-gradients/cond_1/ReduceLogSumExp/sub_grad/NegNeg/gradients/cond_1/ReduceLogSumExp/sub_grad/Sum_1*
_output_shapes
:*
T0
р
3gradients/cond_1/ReduceLogSumExp/sub_grad/Reshape_1Reshape-gradients/cond_1/ReduceLogSumExp/sub_grad/Neg1gradients/cond_1/ReduceLogSumExp/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
м
:gradients/cond_1/ReduceLogSumExp/sub_grad/tuple/group_depsNoOp2^gradients/cond_1/ReduceLogSumExp/sub_grad/Reshape4^gradients/cond_1/ReduceLogSumExp/sub_grad/Reshape_1
╢
Bgradients/cond_1/ReduceLogSumExp/sub_grad/tuple/control_dependencyIdentity1gradients/cond_1/ReduceLogSumExp/sub_grad/Reshape;^gradients/cond_1/ReduceLogSumExp/sub_grad/tuple/group_deps*'
_output_shapes
:         *
T0*D
_class:
86loc:@gradients/cond_1/ReduceLogSumExp/sub_grad/Reshape
╝
Dgradients/cond_1/ReduceLogSumExp/sub_grad/tuple/control_dependency_1Identity3gradients/cond_1/ReduceLogSumExp/sub_grad/Reshape_1;^gradients/cond_1/ReduceLogSumExp/sub_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/cond_1/ReduceLogSumExp/sub_grad/Reshape_1*'
_output_shapes
:         
Ю
gradients/AddNAddN7gradients/cond_1/ReduceLogSumExp_1/Reshape_grad/ReshapeFgradients/cond_1/ReduceLogSumExp_1/sub_grad/tuple/control_dependency_1*
T0*J
_class@
><loc:@gradients/cond_1/ReduceLogSumExp_1/Reshape_grad/Reshape*
N*'
_output_shapes
:         
Ъ
gradients/AddN_1AddN5gradients/cond_1/ReduceLogSumExp/Reshape_grad/ReshapeDgradients/cond_1/ReduceLogSumExp/sub_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@gradients/cond_1/ReduceLogSumExp/Reshape_grad/Reshape*
N*'
_output_shapes
:         
d
!gradients/zeros_2/shape_as_tensorConst*
valueB *
dtype0*
_output_shapes
: 
\
gradients/zeros_2/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
И
gradients/zeros_2Fill!gradients/zeros_2/shape_as_tensorgradients/zeros_2/Const*

index_type0*
_output_shapes
: *
T0
Ц
-gradients/cond_1/rnn/while/Exit_3_grad/b_exitEnterDgradients/cond_1/ReduceLogSumExp_1/sub_grad/tuple/control_dependency*
T0*
is_constant( *
parallel_iterations *'
_output_shapes
:         *8

frame_name*(gradients/cond_1/rnn/while/while_context
╥
-gradients/cond_1/rnn/while/Exit_2_grad/b_exitEntergradients/zeros_2*
parallel_iterations *
_output_shapes
: *8

frame_name*(gradients/cond_1/rnn/while/while_context*
T0*
is_constant( 
z
gradients/Switch_8SwitchSqueezecond_1/pred_id*
T0*:
_output_shapes(
&:         :         
c
gradients/Shape_3Shapegradients/Switch_8*
_output_shapes
:*
T0*
out_type0
\
gradients/zeros_3/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Й
gradients/zeros_3Fillgradients/Shape_3gradients/zeros_3/Const*
T0*

index_type0*'
_output_shapes
:         
╫
:gradients/cond_1/ReduceLogSumExp/Max/Switch_grad/cond_gradMergegradients/zeros_3Bgradients/cond_1/ReduceLogSumExp/sub_grad/tuple/control_dependency*
T0*
N*)
_output_shapes
:         : 
р
1gradients/cond_1/rnn/while/Switch_4_grad/b_switchMerge-gradients/cond_1/rnn/while/Exit_3_grad/b_exit8gradients/cond_1/rnn/while/Switch_4_grad_1/NextIteration*
T0*
N*)
_output_shapes
:         : 
Л
.gradients/cond_1/rnn/while/Merge_3_grad/SwitchSwitch1gradients/cond_1/rnn/while/Switch_4_grad/b_switchgradients/b_count_2*D
_class:
86loc:@gradients/cond_1/rnn/while/Switch_4_grad/b_switch*:
_output_shapes(
&:         :         *
T0
q
8gradients/cond_1/rnn/while/Merge_3_grad/tuple/group_depsNoOp/^gradients/cond_1/rnn/while/Merge_3_grad/Switch
п
@gradients/cond_1/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity.gradients/cond_1/rnn/while/Merge_3_grad/Switch9^gradients/cond_1/rnn/while/Merge_3_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/cond_1/rnn/while/Switch_4_grad/b_switch*'
_output_shapes
:         
│
Bgradients/cond_1/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity0gradients/cond_1/rnn/while/Merge_3_grad/Switch:19^gradients/cond_1/rnn/while/Merge_3_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/cond_1/rnn/while/Switch_4_grad/b_switch*'
_output_shapes
:         
в
gradients/Switch_9SwitchbiLSTM_layers/Reshape_1cond/pred_id*
T0*T
_output_shapesB
@:                  :                  
e
gradients/Shape_4Shapegradients/Switch_9:1*
T0*
out_type0*
_output_shapes
:
\
gradients/zeros_4/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ц
gradients/zeros_4Fillgradients/Shape_4gradients/zeros_4/Const*
T0*

index_type0*4
_output_shapes"
 :                  
╣
,gradients/cond/Shape_1/Switch_grad/cond_gradMerge%gradients/cond/Reshape_1_grad/Reshapegradients/zeros_4*
T0*
N*6
_output_shapes$
":                  : 
и
,gradients/cond_1/rnn/while/Enter_3_grad/ExitExit@gradients/cond_1/rnn/while/Merge_3_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
{
gradients/Switch_10SwitchSqueezecond_1/pred_id*:
_output_shapes(
&:         :         *
T0
f
gradients/Shape_5Shapegradients/Switch_10:1*
_output_shapes
:*
T0*
out_type0
\
gradients/zeros_5/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Й
gradients/zeros_5Fillgradients/Shape_5gradients/zeros_5/Const*
T0*

index_type0*'
_output_shapes
:         
╖
0gradients/cond_1/rnn/while/Switch_grad/cond_gradMerge,gradients/cond_1/rnn/while/Enter_3_grad/Exitgradients/zeros_5*
T0*
N*)
_output_shapes
:         : 
▓
3gradients/cond_1/rnn/while/Select_1_grad/zeros_like	ZerosLike>gradients/cond_1/rnn/while/Select_1_grad/zeros_like/StackPopV2*'
_output_shapes
:         *
T0
┤
9gradients/cond_1/rnn/while/Select_1_grad/zeros_like/ConstConst*
valueB :
         *.
_class$
" loc:@cond_1/rnn/while/Identity_3*
dtype0*
_output_shapes
: 
ю
9gradients/cond_1/rnn/while/Select_1_grad/zeros_like/f_accStackV29gradients/cond_1/rnn/while/Select_1_grad/zeros_like/Const*
_output_shapes
:*
	elem_type0*.
_class$
" loc:@cond_1/rnn/while/Identity_3*

stack_name 
ъ
:gradients/cond_1/rnn/while/Select_1_grad/zeros_like/SwitchSwitch9gradients/cond_1/rnn/while/Select_1_grad/zeros_like/f_acccond_1/pred_id* 
_output_shapes
::*
T0*.
_class$
" loc:@cond_1/rnn/while/Identity_3
Б
9gradients/cond_1/rnn/while/Select_1_grad/zeros_like/EnterEnter:gradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name cond_1/rnn/while/while_context*
T0
√
?gradients/cond_1/rnn/while/Select_1_grad/zeros_like/StackPushV2StackPushV29gradients/cond_1/rnn/while/Select_1_grad/zeros_like/Entercond_1/rnn/while/Identity_3^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
╗
<gradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1SwitchBgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/EnterDgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter_1^gradients/Sub*.
_class$
" loc:@cond_1/rnn/while/Identity_3* 
_output_shapes
::*
T0
─
Bgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/EnterEnter:gradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch*
T0*.
_class$
" loc:@cond_1/rnn/while/Identity_3*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(gradients/cond_1/rnn/while/while_context
Ц
Dgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter_1Entercond_1/pred_id*
T0
*.
_class$
" loc:@cond_1/rnn/while/Identity_3*
parallel_iterations *
is_constant(*
_output_shapes
: *8

frame_name*(gradients/cond_1/rnn/while/while_context
─
>gradients/cond_1/rnn/while/Select_1_grad/zeros_like/StackPopV2
StackPopV2<gradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1*'
_output_shapes
:         *
	elem_type0
╫	
:gradients/cond_1/rnn/while/Select_1_grad/zeros_like/b_syncControlTrigger?^gradients/cond_1/rnn/while/Select_1_grad/zeros_like/StackPopV2;^gradients/cond_1/rnn/while/Select_1_grad/Select/StackPopV2G^gradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/StackPopV2I^gradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/StackPopV2_1a^gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopV2U^gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/StackPopV2W^gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/StackPopV2_1K^gradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/StackPopV2J^gradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/StackPopV2M^gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/StackPopV2C^gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/StackPopV2U^gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/StackPopV2W^gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/StackPopV2_1G^gradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/StackPopV2>^gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/StackPopV2
а
/gradients/cond_1/rnn/while/Select_1_grad/SelectSelect:gradients/cond_1/rnn/while/Select_1_grad/Select/StackPopV2Bgradients/cond_1/rnn/while/Merge_3_grad/tuple/control_dependency_13gradients/cond_1/rnn/while/Select_1_grad/zeros_like*'
_output_shapes
:         *
T0
▓
5gradients/cond_1/rnn/while/Select_1_grad/Select/ConstConst*
valueB :
         *0
_class&
$"loc:@cond_1/rnn/while/GreaterEqual*
dtype0*
_output_shapes
: 
ш
5gradients/cond_1/rnn/while/Select_1_grad/Select/f_accStackV25gradients/cond_1/rnn/while/Select_1_grad/Select/Const*
	elem_type0
*0
_class&
$"loc:@cond_1/rnn/while/GreaterEqual*

stack_name *
_output_shapes
:
ф
6gradients/cond_1/rnn/while/Select_1_grad/Select/SwitchSwitch5gradients/cond_1/rnn/while/Select_1_grad/Select/f_acccond_1/pred_id*
T0*0
_class&
$"loc:@cond_1/rnn/while/GreaterEqual* 
_output_shapes
::
∙
5gradients/cond_1/rnn/while/Select_1_grad/Select/EnterEnter6gradients/cond_1/rnn/while/Select_1_grad/Select/Switch*
_output_shapes
:*.

frame_name cond_1/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
ё
;gradients/cond_1/rnn/while/Select_1_grad/Select/StackPushV2StackPushV25gradients/cond_1/rnn/while/Select_1_grad/Select/Entercond_1/rnn/while/GreaterEqual^gradients/Add*#
_output_shapes
:         *
swap_memory( *
T0

╡
8gradients/cond_1/rnn/while/Select_1_grad/Select/Switch_1Switch>gradients/cond_1/rnn/while/Select_1_grad/Select/Switch_1/EnterDgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter_1^gradients/Sub*
T0*0
_class&
$"loc:@cond_1/rnn/while/GreaterEqual* 
_output_shapes
::
╛
>gradients/cond_1/rnn/while/Select_1_grad/Select/Switch_1/EnterEnter6gradients/cond_1/rnn/while/Select_1_grad/Select/Switch*
is_constant(*
_output_shapes
:*8

frame_name*(gradients/cond_1/rnn/while/while_context*
T0*0
_class&
$"loc:@cond_1/rnn/while/GreaterEqual*
parallel_iterations 
╕
:gradients/cond_1/rnn/while/Select_1_grad/Select/StackPopV2
StackPopV28gradients/cond_1/rnn/while/Select_1_grad/Select/Switch_1*#
_output_shapes
:         *
	elem_type0

в
1gradients/cond_1/rnn/while/Select_1_grad/Select_1Select:gradients/cond_1/rnn/while/Select_1_grad/Select/StackPopV23gradients/cond_1/rnn/while/Select_1_grad/zeros_likeBgradients/cond_1/rnn/while/Merge_3_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         
з
9gradients/cond_1/rnn/while/Select_1_grad/tuple/group_depsNoOp0^gradients/cond_1/rnn/while/Select_1_grad/Select2^gradients/cond_1/rnn/while/Select_1_grad/Select_1
░
Agradients/cond_1/rnn/while/Select_1_grad/tuple/control_dependencyIdentity/gradients/cond_1/rnn/while/Select_1_grad/Select:^gradients/cond_1/rnn/while/Select_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/cond_1/rnn/while/Select_1_grad/Select*'
_output_shapes
:         
╢
Cgradients/cond_1/rnn/while/Select_1_grad/tuple/control_dependency_1Identity1gradients/cond_1/rnn/while/Select_1_grad/Select_1:^gradients/cond_1/rnn/while/Select_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/cond_1/rnn/while/Select_1_grad/Select_1*'
_output_shapes
:         
Р
gradients/AddN_2AddN:gradients/cond_1/ReduceLogSumExp/Max/Switch_grad/cond_grad0gradients/cond_1/rnn/while/Switch_grad/cond_grad*
N*'
_output_shapes
:         *
T0*M
_classC
A?loc:@gradients/cond_1/ReduceLogSumExp/Max/Switch_grad/cond_grad
a
gradients/Squeeze_grad/ShapeShapeSlice*
_output_shapes
:*
T0*
out_type0
Э
gradients/Squeeze_grad/ReshapeReshapegradients/AddN_2gradients/Squeeze_grad/Shape*
T0*
Tshape0*+
_output_shapes
:         
Н
+gradients/cond_1/rnn/while/add_2_grad/ShapeShape"cond_1/rnn/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:
С
-gradients/cond_1/rnn/while/add_2_grad/Shape_1Shape$cond_1/rnn/while/ReduceLogSumExp/add*
T0*
out_type0*
_output_shapes
:
г
;gradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/StackPopV2Hgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
╠
Agradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/ConstConst*
valueB :
         *>
_class4
20loc:@gradients/cond_1/rnn/while/add_2_grad/Shape*
dtype0*
_output_shapes
: 
О
Agradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/f_accStackV2Agradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Const*>
_class4
20loc:@gradients/cond_1/rnn/while/add_2_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
К
Bgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/SwitchSwitchAgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/f_acccond_1/pred_id*
T0*>
_class4
20loc:@gradients/cond_1/rnn/while/add_2_grad/Shape* 
_output_shapes
::
С
Agradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/EnterEnterBgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name cond_1/rnn/while/while_context
О
Ggradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Agradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Enter+gradients/cond_1/rnn/while/add_2_grad/Shape^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
█
Dgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_1SwitchJgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_1/EnterDgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter_1^gradients/Sub* 
_output_shapes
::*
T0*>
_class4
20loc:@gradients/cond_1/rnn/while/add_2_grad/Shape
ф
Jgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_1/EnterEnterBgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch*
T0*>
_class4
20loc:@gradients/cond_1/rnn/while/add_2_grad/Shape*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(gradients/cond_1/rnn/while/while_context
╟
Fgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Dgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_1*
_output_shapes
:*
	elem_type0
╨
Cgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *
valueB :
         *@
_class6
42loc:@gradients/cond_1/rnn/while/add_2_grad/Shape_1
Ф
Cgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/f_acc_1StackV2Cgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Const_1*
_output_shapes
:*
	elem_type0*@
_class6
42loc:@gradients/cond_1/rnn/while/add_2_grad/Shape_1*

stack_name 
Р
Dgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_2SwitchCgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/f_acc_1cond_1/pred_id* 
_output_shapes
::*
T0*@
_class6
42loc:@gradients/cond_1/rnn/while/add_2_grad/Shape_1
Х
Cgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Enter_1EnterDgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_2*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name cond_1/rnn/while/while_context
Ф
Igradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Cgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Enter_1-gradients/cond_1/rnn/while/add_2_grad/Shape_1^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
▌
Dgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_3SwitchJgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_3/EnterDgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter_1^gradients/Sub* 
_output_shapes
::*
T0*@
_class6
42loc:@gradients/cond_1/rnn/while/add_2_grad/Shape_1
ш
Jgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_3/EnterEnterDgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_2*
T0*@
_class6
42loc:@gradients/cond_1/rnn/while/add_2_grad/Shape_1*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(gradients/cond_1/rnn/while/while_context
╔
Hgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Dgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_3*
_output_shapes
:*
	elem_type0
Є
)gradients/cond_1/rnn/while/add_2_grad/SumSumCgradients/cond_1/rnn/while/Select_1_grad/tuple/control_dependency_1;gradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ї
-gradients/cond_1/rnn/while/add_2_grad/ReshapeReshape)gradients/cond_1/rnn/while/add_2_grad/SumFgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*0
_output_shapes
:                  
Ў
+gradients/cond_1/rnn/while/add_2_grad/Sum_1SumCgradients/cond_1/rnn/while/Select_1_grad/tuple/control_dependency_1=gradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
·
/gradients/cond_1/rnn/while/add_2_grad/Reshape_1Reshape+gradients/cond_1/rnn/while/add_2_grad/Sum_1Hgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*0
_output_shapes
:                  
а
6gradients/cond_1/rnn/while/add_2_grad/tuple/group_depsNoOp.^gradients/cond_1/rnn/while/add_2_grad/Reshape0^gradients/cond_1/rnn/while/add_2_grad/Reshape_1
ж
>gradients/cond_1/rnn/while/add_2_grad/tuple/control_dependencyIdentity-gradients/cond_1/rnn/while/add_2_grad/Reshape7^gradients/cond_1/rnn/while/add_2_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/cond_1/rnn/while/add_2_grad/Reshape*'
_output_shapes
:         
м
@gradients/cond_1/rnn/while/add_2_grad/tuple/control_dependency_1Identity/gradients/cond_1/rnn/while/add_2_grad/Reshape_17^gradients/cond_1/rnn/while/add_2_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/cond_1/rnn/while/add_2_grad/Reshape_1*'
_output_shapes
:         
[
gradients/Slice_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
_
gradients/Slice_grad/ShapeShapeSlice*
T0*
out_type0*
_output_shapes
:
^
gradients/Slice_grad/stack/1Const*
value	B :*
dtype0*
_output_shapes
: 
Х
gradients/Slice_grad/stackPackgradients/Slice_grad/Rankgradients/Slice_grad/stack/1*
N*
_output_shapes
:*
T0*

axis 
З
gradients/Slice_grad/ReshapeReshapeSlice/begingradients/Slice_grad/stack*
_output_shapes

:*
T0*
Tshape0
s
gradients/Slice_grad/Shape_1ShapebiLSTM_layers/Reshape_1*
out_type0*
_output_shapes
:*
T0
~
gradients/Slice_grad/subSubgradients/Slice_grad/Shape_1gradients/Slice_grad/Shape*
T0*
_output_shapes
:
m
gradients/Slice_grad/sub_1Subgradients/Slice_grad/subSlice/begin*
T0*
_output_shapes
:
Ш
gradients/Slice_grad/Reshape_1Reshapegradients/Slice_grad/sub_1gradients/Slice_grad/stack*
_output_shapes

:*
T0*
Tshape0
b
 gradients/Slice_grad/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
┼
gradients/Slice_grad/concatConcatV2gradients/Slice_grad/Reshapegradients/Slice_grad/Reshape_1 gradients/Slice_grad/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:
м
gradients/Slice_grad/PadPadgradients/Squeeze_grad/Reshapegradients/Slice_grad/concat*
T0*
	Tpaddings0*4
_output_shapes"
 :                  
в
Sgradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Ygradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter[gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub*
source	gradients*;
_class1
/-loc:@cond_1/rnn/while/TensorArrayReadV3/Enter*
_output_shapes

:: 
╞
Ygradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEntercond_1/rnn/TensorArray_1*
T0*;
_class1
/-loc:@cond_1/rnn/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(gradients/cond_1/rnn/while/while_context
ё
[gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterEcond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *
is_constant(*
_output_shapes
: *8

frame_name*(gradients/cond_1/rnn/while/while_context*
T0*;
_class1
/-loc:@cond_1/rnn/while/TensorArrayReadV3/Enter
ь
Ogradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentity[gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1T^gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*;
_class1
/-loc:@cond_1/rnn/while/TensorArrayReadV3/Enter*
_output_shapes
: 
╘
Ugradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Sgradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3`gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopV2>gradients/cond_1/rnn/while/add_2_grad/tuple/control_dependencyOgradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
╓
[gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/ConstConst*
valueB :
         *.
_class$
" loc:@cond_1/rnn/while/Identity_1*
dtype0*
_output_shapes
: 
▓
[gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_accStackV2[gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Const*.
_class$
" loc:@cond_1/rnn/while/Identity_1*

stack_name *
_output_shapes
:*
	elem_type0
о
\gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/SwitchSwitch[gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acccond_1/pred_id*.
_class$
" loc:@cond_1/rnn/while/Identity_1* 
_output_shapes
::*
T0
┼
[gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/EnterEnter\gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Switch*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name cond_1/rnn/while/while_context
о
agradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPushV2StackPushV2[gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Entercond_1/rnn/while/Identity_1^gradients/Add*
T0*
_output_shapes
: *
swap_memory( 
 
^gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Switch_1Switchdgradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Switch_1/EnterDgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter_1^gradients/Sub*
T0*.
_class$
" loc:@cond_1/rnn/while/Identity_1* 
_output_shapes
::
И
dgradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Switch_1/EnterEnter\gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Switch*
is_constant(*
_output_shapes
:*8

frame_name*(gradients/cond_1/rnn/while/while_context*
T0*.
_class$
" loc:@cond_1/rnn/while/Identity_1*
parallel_iterations 
ў
`gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopV2
StackPopV2^gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Switch_1*
_output_shapes
: *
	elem_type0
Э
9gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/ShapeShape$cond_1/rnn/while/ReduceLogSumExp/Log*
T0*
out_type0*
_output_shapes
:
г
;gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Shape_1Shape(cond_1/rnn/while/ReduceLogSumExp/Reshape*
T0*
out_type0*
_output_shapes
:
═
Igradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgsBroadcastGradientArgsTgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/StackPopV2Vgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
ш
Ogradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/ConstConst*
valueB :
         *L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Shape*
dtype0*
_output_shapes
: 
╕
Ogradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/f_accStackV2Ogradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Const*
_output_shapes
:*
	elem_type0*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Shape*

stack_name 
┤
Pgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/SwitchSwitchOgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/f_acccond_1/pred_id*
T0*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Shape* 
_output_shapes
::
н
Ogradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/EnterEnterPgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name cond_1/rnn/while/while_context
╕
Ugradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ogradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Enter9gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Е
Rgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_1SwitchXgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_1/EnterDgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter_1^gradients/Sub* 
_output_shapes
::*
T0*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Shape
О
Xgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_1/EnterEnterPgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch*
T0*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Shape*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(gradients/cond_1/rnn/while/while_context
у
Tgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Rgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_1*
_output_shapes
:*
	elem_type0
ь
Qgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Const_1Const*
valueB :
         *N
_classD
B@loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Shape_1*
dtype0*
_output_shapes
: 
╛
Qgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/f_acc_1StackV2Qgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Const_1*
	elem_type0*N
_classD
B@loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Shape_1*

stack_name *
_output_shapes
:
║
Rgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_2SwitchQgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/f_acc_1cond_1/pred_id*
T0*N
_classD
B@loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Shape_1* 
_output_shapes
::
▒
Qgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Enter_1EnterRgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_2*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name cond_1/rnn/while/while_context*
T0
╛
Wgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Qgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Enter_1;gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Shape_1^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
З
Rgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_3SwitchXgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_3/EnterDgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter_1^gradients/Sub* 
_output_shapes
::*
T0*N
_classD
B@loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Shape_1
Т
Xgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_3/EnterEnterRgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_2*
T0*N
_classD
B@loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Shape_1*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(gradients/cond_1/rnn/while/while_context
х
Vgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Rgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_3*
_output_shapes
:*
	elem_type0
Л
7gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/SumSum@gradients/cond_1/rnn/while/add_2_grad/tuple/control_dependency_1Igradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ю
;gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/ReshapeReshape7gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/SumTgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*0
_output_shapes
:                  
П
9gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Sum_1Sum@gradients/cond_1/rnn/while/add_2_grad/tuple/control_dependency_1Kgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
д
=gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Reshape_1Reshape9gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Sum_1Vgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/StackPopV2_1*0
_output_shapes
:                  *
T0*
Tshape0
╩
Dgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/tuple/group_depsNoOp<^gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Reshape>^gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Reshape_1
▐
Lgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/tuple/control_dependencyIdentity;gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/ReshapeE^gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/tuple/group_deps*N
_classD
B@loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Reshape*'
_output_shapes
:         *
T0
ф
Ngradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/tuple/control_dependency_1Identity=gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Reshape_1E^gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Reshape_1*'
_output_shapes
:         
Ц
?gradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst^cond_1/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
Ф
Agradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Enter?gradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *8

frame_name*(gradients/cond_1/rnn/while/while_context
В
Agradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeAgradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Ggradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
_output_shapes
: : *
T0*
N
┼
@gradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchAgradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_2*
T0*
_output_shapes
: : 
А
=gradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddBgradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1Ugradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
╕
Ggradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration=gradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
м
Agradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Exit@gradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 
к
=gradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/ShapeShape-cond_1/rnn/while/ReduceLogSumExp/StopGradient*
out_type0*
_output_shapes
:*
T0
к
?gradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/ReshapeReshapeNgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/tuple/control_dependency_1Jgradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/StackPopV2*
T0*
Tshape0*+
_output_shapes
:         
т
Egradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/ConstConst*
valueB :
         *P
_classF
DBloc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Shape*
dtype0*
_output_shapes
: 
и
Egradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/f_accStackV2Egradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/Const*P
_classF
DBloc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
д
Fgradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/SwitchSwitchEgradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/f_acccond_1/pred_id*
T0*P
_classF
DBloc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Shape* 
_output_shapes
::
Щ
Egradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/EnterEnterFgradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/Switch*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name cond_1/rnn/while/while_context
и
Kgradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/StackPushV2StackPushV2Egradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/Enter=gradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Shape^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
ї
Hgradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/Switch_1SwitchNgradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/Switch_1/EnterDgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter_1^gradients/Sub*
T0*P
_classF
DBloc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Shape* 
_output_shapes
::
■
Ngradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/Switch_1/EnterEnterFgradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/Switch*
is_constant(*
_output_shapes
:*8

frame_name*(gradients/cond_1/rnn/while/while_context*
T0*P
_classF
DBloc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Shape*
parallel_iterations 
╧
Jgradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/StackPopV2
StackPopV2Hgradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/Switch_1*
_output_shapes
:*
	elem_type0
╩
vgradients/cond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3cond_1/rnn/TensorArray_1Agradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
source	gradients*+
_class!
loc:@cond_1/rnn/TensorArray_1*
_output_shapes

:: 
И
rgradients/cond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityAgradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3w^gradients/cond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*+
_class!
loc:@cond_1/rnn/TensorArray_1*
_output_shapes
: 
╬
hgradients/cond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3vgradients/cond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3#cond_1/rnn/TensorArrayUnstack/rangergradients/cond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*
_output_shapes
:*
element_shape:
Ь
egradients/cond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpB^gradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3i^gradients/cond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
З
mgradients/cond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentityhgradients/cond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3f^gradients/cond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*{
_classq
omloc:@gradients/cond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3*4
_output_shapes"
 :                  *
T0
Э
ogradients/cond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityAgradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3f^gradients/cond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
_output_shapes
: *
T0*T
_classJ
HFloc:@gradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
Ш
>gradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal
ReciprocalIgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/StackPopV2M^gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/tuple/control_dependency*
T0*'
_output_shapes
:         
╚
Dgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/ConstConst*
valueB :
         *7
_class-
+)loc:@cond_1/rnn/while/ReduceLogSumExp/Sum*
dtype0*
_output_shapes
: 
Н
Dgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/f_accStackV2Dgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/Const*
_output_shapes
:*
	elem_type0*7
_class-
+)loc:@cond_1/rnn/while/ReduceLogSumExp/Sum*

stack_name 
Й
Egradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/SwitchSwitchDgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/f_acccond_1/pred_id*
T0*7
_class-
+)loc:@cond_1/rnn/while/ReduceLogSumExp/Sum* 
_output_shapes
::
Ч
Dgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/EnterEnterEgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/Switch*
_output_shapes
:*.

frame_name cond_1/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
Ъ
Jgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/StackPushV2StackPushV2Dgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/Enter$cond_1/rnn/while/ReduceLogSumExp/Sum^gradients/Add*
T0*'
_output_shapes
:         *
swap_memory( 
┌
Ggradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/Switch_1SwitchMgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/Switch_1/EnterDgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter_1^gradients/Sub*
T0*7
_class-
+)loc:@cond_1/rnn/while/ReduceLogSumExp/Sum* 
_output_shapes
::
у
Mgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/Switch_1/EnterEnterEgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/Switch*
is_constant(*
_output_shapes
:*8

frame_name*(gradients/cond_1/rnn/while/while_context*
T0*7
_class-
+)loc:@cond_1/rnn/while/ReduceLogSumExp/Sum*
parallel_iterations 
┌
Igradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/StackPopV2
StackPopV2Ggradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/Switch_1*'
_output_shapes
:         *
	elem_type0
■
7gradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/mulMulLgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/tuple/control_dependency>gradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal*
T0*'
_output_shapes
:         
Э
9gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/ShapeShape$cond_1/rnn/while/ReduceLogSumExp/Exp*
_output_shapes
:*
T0*
out_type0
╪
8gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/SizeConst^gradients/Sub*
value	B :*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape*
dtype0*
_output_shapes
: 
к
7gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/addAdd=gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/add/Const8gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Size*
_output_shapes
:*
T0*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape
х
=gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/add/ConstConst^gradients/Sub*
valueB:*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape*
dtype0*
_output_shapes
:
й
7gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/modFloorMod7gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/add8gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Size*
T0*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape*
_output_shapes
:
у
;gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape_1Const^gradients/Sub*
valueB:*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape*
dtype0*
_output_shapes
:
▀
?gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/range/startConst^gradients/Sub*
value	B : *L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape*
dtype0*
_output_shapes
: 
▀
?gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/range/deltaConst^gradients/Sub*
value	B :*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape*
dtype0*
_output_shapes
: 
Ї
9gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/rangeRange?gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/range/start8gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Size?gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/range/delta*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape*
_output_shapes
:*

Tidx0
▐
>gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Fill/valueConst^gradients/Sub*
dtype0*
_output_shapes
: *
value	B :*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape
┬
8gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/FillFill;gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape_1>gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Fill/value*
T0*

index_type0*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape*
_output_shapes
:
╙
Agradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitchDynamicStitch9gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/range7gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/modLgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/StackPopV28gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Fill*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape*
N*#
_output_shapes
:         *
T0
р
Ggradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/ConstConst*
valueB :
         *L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape*
dtype0*
_output_shapes
: 
и
Ggradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/f_accStackV2Ggradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/Const*
	elem_type0*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape*

stack_name *
_output_shapes
:
д
Hgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/SwitchSwitchGgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/f_acccond_1/pred_id* 
_output_shapes
::*
T0*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape
ы
Ggradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/EnterEnterHgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/Switch*
T0*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape*
parallel_iterations *
is_constant(*
_output_shapes
:*.

frame_name cond_1/rnn/while/while_context
Ў
Mgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/StackPushV2StackPushV2Ggradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/Enter9gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape^gradients/Add*
_output_shapes
:*
swap_memory( *
T0*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape
ї
Jgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/Switch_1SwitchPgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/Switch_1/EnterDgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter_1^gradients/Sub*
T0*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape* 
_output_shapes
::
■
Pgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/Switch_1/EnterEnterHgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/Switch*
T0*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(gradients/cond_1/rnn/while/while_context
б
Lgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/StackPopV2
StackPopV2Jgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/Switch_1*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape*
_output_shapes
:*
	elem_type0
▌
=gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Maximum/yConst^gradients/Sub*
value	B :*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape*
dtype0*
_output_shapes
: 
─
;gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/MaximumMaximumAgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch=gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Maximum/y*
T0*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape*#
_output_shapes
:         
╞
<gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/floordivFloorDivLgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/StackPopV2;gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Maximum*
_output_shapes
:*
T0*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape
є
;gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/ReshapeReshape7gradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/mulAgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
Г
8gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/TileTile;gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Reshape<gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/floordiv*+
_output_shapes
:         *

Tmultiples0*
T0
Є
7gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mulMul8gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/TileBgradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/StackPopV2*+
_output_shapes
:         *
T0
┴
=gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/ConstConst*
valueB :
         *7
_class-
+)loc:@cond_1/rnn/while/ReduceLogSumExp/Exp*
dtype0*
_output_shapes
: 
 
=gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/f_accStackV2=gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/Const*7
_class-
+)loc:@cond_1/rnn/while/ReduceLogSumExp/Exp*

stack_name *
_output_shapes
:*
	elem_type0
√
>gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/SwitchSwitch=gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/f_acccond_1/pred_id*
T0*7
_class-
+)loc:@cond_1/rnn/while/ReduceLogSumExp/Exp* 
_output_shapes
::
Й
=gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/EnterEnter>gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/Switch*
parallel_iterations *
_output_shapes
:*.

frame_name cond_1/rnn/while/while_context*
T0*
is_constant(
Р
Cgradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/StackPushV2StackPushV2=gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/Enter$cond_1/rnn/while/ReduceLogSumExp/Exp^gradients/Add*
T0*+
_output_shapes
:         *
swap_memory( 
╠
@gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/Switch_1SwitchFgradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/Switch_1/EnterDgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter_1^gradients/Sub*
T0*7
_class-
+)loc:@cond_1/rnn/while/ReduceLogSumExp/Exp* 
_output_shapes
::
╒
Fgradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/Switch_1/EnterEnter>gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/Switch*
T0*7
_class-
+)loc:@cond_1/rnn/while/ReduceLogSumExp/Exp*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(gradients/cond_1/rnn/while/while_context
╨
Bgradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/StackPopV2
StackPopV2@gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/Switch_1*+
_output_shapes
:         *
	elem_type0
В
5gradients/cond_1/rnn/transpose_grad/InvertPermutationInvertPermutationcond_1/rnn/concat*
_output_shapes
:*
T0
м
-gradients/cond_1/rnn/transpose_grad/transpose	Transposemgradients/cond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency5gradients/cond_1/rnn/transpose_grad/InvertPermutation*4
_output_shapes"
 :                  *
Tperm0*
T0
П
9gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/ShapeShapecond_1/rnn/while/add_1*
T0*
out_type0*
_output_shapes
:
и
;gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Shape_1Shape-cond_1/rnn/while/ReduceLogSumExp/StopGradient*
T0*
out_type0*
_output_shapes
:
═
Igradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgsBroadcastGradientArgsTgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/StackPopV2Vgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:         :         *
T0
ш
Ogradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/ConstConst*
valueB :
         *L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Shape*
dtype0*
_output_shapes
: 
╕
Ogradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/f_accStackV2Ogradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Const*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
┤
Pgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/SwitchSwitchOgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/f_acccond_1/pred_id*
T0*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Shape* 
_output_shapes
::
н
Ogradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/EnterEnterPgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name cond_1/rnn/while/while_context
╕
Ugradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ogradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Enter9gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
Е
Rgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_1SwitchXgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_1/EnterDgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter_1^gradients/Sub*
T0*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Shape* 
_output_shapes
::
О
Xgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_1/EnterEnterPgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch*
_output_shapes
:*8

frame_name*(gradients/cond_1/rnn/while/while_context*
T0*L
_classB
@>loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Shape*
parallel_iterations *
is_constant(
у
Tgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Rgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_1*
_output_shapes
:*
	elem_type0
ь
Qgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *
valueB :
         *N
_classD
B@loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Shape_1
╛
Qgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/f_acc_1StackV2Qgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Const_1*N
_classD
B@loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
║
Rgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_2SwitchQgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/f_acc_1cond_1/pred_id* 
_output_shapes
::*
T0*N
_classD
B@loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Shape_1
▒
Qgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Enter_1EnterRgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_2*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name cond_1/rnn/while/while_context
╛
Wgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Qgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Enter_1;gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Shape_1^gradients/Add*
_output_shapes
:*
swap_memory( *
T0
З
Rgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_3SwitchXgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_3/EnterDgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter_1^gradients/Sub*
T0*N
_classD
B@loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Shape_1* 
_output_shapes
::
Т
Xgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_3/EnterEnterRgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_2*
is_constant(*
_output_shapes
:*8

frame_name*(gradients/cond_1/rnn/while/while_context*
T0*N
_classD
B@loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Shape_1*
parallel_iterations 
х
Vgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Rgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_3*
_output_shapes
:*
	elem_type0
В
7gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/SumSum7gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mulIgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
л
;gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/ReshapeReshape7gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/SumTgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/StackPopV2*
Tshape0*=
_output_shapes+
):'                           *
T0
Ж
9gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Sum_1Sum7gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mulKgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ь
7gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/NegNeg9gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Sum_1*
_output_shapes
:*
T0
п
=gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Reshape_1Reshape7gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/NegVgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/StackPopV2_1*=
_output_shapes+
):'                           *
T0*
Tshape0
╩
Dgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/tuple/group_depsNoOp<^gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Reshape>^gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Reshape_1
т
Lgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/tuple/control_dependencyIdentity;gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/ReshapeE^gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Reshape*+
_output_shapes
:         
ш
Ngradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/tuple/control_dependency_1Identity=gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Reshape_1E^gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Reshape_1*+
_output_shapes
:         
b
 gradients/cond_1/Slice_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
m
!gradients/cond_1/Slice_grad/ShapeShapecond_1/Slice*
_output_shapes
:*
T0*
out_type0
e
#gradients/cond_1/Slice_grad/stack/1Const*
value	B :*
dtype0*
_output_shapes
: 
к
!gradients/cond_1/Slice_grad/stackPack gradients/cond_1/Slice_grad/Rank#gradients/cond_1/Slice_grad/stack/1*

axis *
N*
_output_shapes
:*
T0
Ь
#gradients/cond_1/Slice_grad/ReshapeReshapecond_1/Slice/begin!gradients/cond_1/Slice_grad/stack*
T0*
Tshape0*
_output_shapes

:
v
#gradients/cond_1/Slice_grad/Shape_1Shapecond_1/Slice/Switch*
T0*
out_type0*
_output_shapes
:
У
gradients/cond_1/Slice_grad/subSub#gradients/cond_1/Slice_grad/Shape_1!gradients/cond_1/Slice_grad/Shape*
T0*
_output_shapes
:
В
!gradients/cond_1/Slice_grad/sub_1Subgradients/cond_1/Slice_grad/subcond_1/Slice/begin*
T0*
_output_shapes
:
н
%gradients/cond_1/Slice_grad/Reshape_1Reshape!gradients/cond_1/Slice_grad/sub_1!gradients/cond_1/Slice_grad/stack*
T0*
Tshape0*
_output_shapes

:
i
'gradients/cond_1/Slice_grad/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
с
"gradients/cond_1/Slice_grad/concatConcatV2#gradients/cond_1/Slice_grad/Reshape%gradients/cond_1/Slice_grad/Reshape_1'gradients/cond_1/Slice_grad/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:
╔
gradients/cond_1/Slice_grad/PadPad-gradients/cond_1/rnn/transpose_grad/transpose"gradients/cond_1/Slice_grad/concat*
T0*
	Tpaddings0*4
_output_shapes"
 :                  
╝
gradients/AddN_3AddN?gradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/ReshapeNgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/tuple/control_dependency_1*
T0*R
_classH
FDloc:@gradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape*
N*+
_output_shapes
:         
е
gradients/Switch_11SwitchbiLSTM_layers/Reshape_1cond_1/pred_id*
T0*T
_output_shapesB
@:                  :                  
f
gradients/Shape_6Shapegradients/Switch_11:1*
_output_shapes
:*
T0*
out_type0
\
gradients/zeros_6/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ц
gradients/zeros_6Fillgradients/Shape_6gradients/zeros_6/Const*
T0*

index_type0*4
_output_shapes"
 :                  
│
,gradients/cond_1/Slice/Switch_grad/cond_gradMergegradients/cond_1/Slice_grad/Padgradients/zeros_6*
N*6
_output_shapes$
":                  : *
T0
Ж
+gradients/cond_1/rnn/while/add_1_grad/ShapeShapecond_1/rnn/while/ExpandDims*
T0*
out_type0*
_output_shapes
:
Т
-gradients/cond_1/rnn/while/add_1_grad/Shape_1Const^gradients/Sub*
_output_shapes
:*!
valueB"         *
dtype0
И
;gradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/StackPopV2-gradients/cond_1/rnn/while/add_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╠
Agradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/ConstConst*
valueB :
         *>
_class4
20loc:@gradients/cond_1/rnn/while/add_1_grad/Shape*
dtype0*
_output_shapes
: 
О
Agradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/f_accStackV2Agradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/Const*>
_class4
20loc:@gradients/cond_1/rnn/while/add_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
К
Bgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/SwitchSwitchAgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/f_acccond_1/pred_id*
T0*>
_class4
20loc:@gradients/cond_1/rnn/while/add_1_grad/Shape* 
_output_shapes
::
С
Agradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/EnterEnterBgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/Switch*
_output_shapes
:*.

frame_name cond_1/rnn/while/while_context*
T0*
is_constant(*
parallel_iterations 
О
Ggradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Agradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/Enter+gradients/cond_1/rnn/while/add_1_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
█
Dgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/Switch_1SwitchJgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/Switch_1/EnterDgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter_1^gradients/Sub* 
_output_shapes
::*
T0*>
_class4
20loc:@gradients/cond_1/rnn/while/add_1_grad/Shape
ф
Jgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/Switch_1/EnterEnterBgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/Switch*
is_constant(*
_output_shapes
:*8

frame_name*(gradients/cond_1/rnn/while/while_context*
T0*>
_class4
20loc:@gradients/cond_1/rnn/while/add_1_grad/Shape*
parallel_iterations 
╟
Fgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Dgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/Switch_1*
_output_shapes
:*
	elem_type0
√
)gradients/cond_1/rnn/while/add_1_grad/SumSumLgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/tuple/control_dependency;gradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Б
-gradients/cond_1/rnn/while/add_1_grad/ReshapeReshape)gradients/cond_1/rnn/while/add_1_grad/SumFgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*=
_output_shapes+
):'                           
 
+gradients/cond_1/rnn/while/add_1_grad/Sum_1SumLgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/tuple/control_dependency=gradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
╤
/gradients/cond_1/rnn/while/add_1_grad/Reshape_1Reshape+gradients/cond_1/rnn/while/add_1_grad/Sum_1-gradients/cond_1/rnn/while/add_1_grad/Shape_1*
T0*
Tshape0*"
_output_shapes
:
а
6gradients/cond_1/rnn/while/add_1_grad/tuple/group_depsNoOp.^gradients/cond_1/rnn/while/add_1_grad/Reshape0^gradients/cond_1/rnn/while/add_1_grad/Reshape_1
к
>gradients/cond_1/rnn/while/add_1_grad/tuple/control_dependencyIdentity-gradients/cond_1/rnn/while/add_1_grad/Reshape7^gradients/cond_1/rnn/while/add_1_grad/tuple/group_deps*+
_output_shapes
:         *
T0*@
_class6
42loc:@gradients/cond_1/rnn/while/add_1_grad/Reshape
з
@gradients/cond_1/rnn/while/add_1_grad/tuple/control_dependency_1Identity/gradients/cond_1/rnn/while/add_1_grad/Reshape_17^gradients/cond_1/rnn/while/add_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/cond_1/rnn/while/add_1_grad/Reshape_1*"
_output_shapes
:
Л
0gradients/cond_1/rnn/while/ExpandDims_grad/ShapeShapecond_1/rnn/while/Identity_3*
out_type0*
_output_shapes
:*
T0
№
2gradients/cond_1/rnn/while/ExpandDims_grad/ReshapeReshape>gradients/cond_1/rnn/while/add_1_grad/tuple/control_dependency=gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/StackPopV2*
T0*
Tshape0*'
_output_shapes
:         
╚
8gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/ConstConst*
valueB :
         *C
_class9
75loc:@gradients/cond_1/rnn/while/ExpandDims_grad/Shape*
dtype0*
_output_shapes
: 
Б
8gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/f_accStackV28gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/Const*C
_class9
75loc:@gradients/cond_1/rnn/while/ExpandDims_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
¤
9gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/SwitchSwitch8gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/f_acccond_1/pred_id*
T0*C
_class9
75loc:@gradients/cond_1/rnn/while/ExpandDims_grad/Shape* 
_output_shapes
::
 
8gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/EnterEnter9gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/Switch*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name cond_1/rnn/while/while_context
Б
>gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/StackPushV2StackPushV28gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/Enter0gradients/cond_1/rnn/while/ExpandDims_grad/Shape^gradients/Add*
T0*
_output_shapes
:*
swap_memory( 
╬
;gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/Switch_1SwitchAgradients/cond_1/rnn/while/ExpandDims_grad/Reshape/Switch_1/EnterDgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter_1^gradients/Sub*
T0*C
_class9
75loc:@gradients/cond_1/rnn/while/ExpandDims_grad/Shape* 
_output_shapes
::
╫
Agradients/cond_1/rnn/while/ExpandDims_grad/Reshape/Switch_1/EnterEnter9gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/Switch*
T0*C
_class9
75loc:@gradients/cond_1/rnn/while/ExpandDims_grad/Shape*
parallel_iterations *
is_constant(*
_output_shapes
:*8

frame_name*(gradients/cond_1/rnn/while/while_context
╡
=gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/StackPopV2
StackPopV2;gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/Switch_1*
_output_shapes
:*
	elem_type0
а
1gradients/cond_1/rnn/while/add_1/Enter_grad/b_accConst^cond_1/switch_f*"
_output_shapes
:*!
valueB*    *
dtype0
Д
3gradients/cond_1/rnn/while/add_1/Enter_grad/b_acc_1Enter1gradients/cond_1/rnn/while/add_1/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *"
_output_shapes
:*8

frame_name*(gradients/cond_1/rnn/while/while_context
ф
3gradients/cond_1/rnn/while/add_1/Enter_grad/b_acc_2Merge3gradients/cond_1/rnn/while/add_1/Enter_grad/b_acc_19gradients/cond_1/rnn/while/add_1/Enter_grad/NextIteration*
N*$
_output_shapes
:: *
T0
┴
2gradients/cond_1/rnn/while/add_1/Enter_grad/SwitchSwitch3gradients/cond_1/rnn/while/add_1/Enter_grad/b_acc_2gradients/b_count_2*
T0*0
_output_shapes
::
█
/gradients/cond_1/rnn/while/add_1/Enter_grad/AddAdd4gradients/cond_1/rnn/while/add_1/Enter_grad/Switch:1@gradients/cond_1/rnn/while/add_1_grad/tuple/control_dependency_1*
T0*"
_output_shapes
:
и
9gradients/cond_1/rnn/while/add_1/Enter_grad/NextIterationNextIteration/gradients/cond_1/rnn/while/add_1/Enter_grad/Add*
T0*"
_output_shapes
:
Ь
3gradients/cond_1/rnn/while/add_1/Enter_grad/b_acc_3Exit2gradients/cond_1/rnn/while/add_1/Enter_grad/Switch*"
_output_shapes
:*
T0
О
gradients/AddN_4AddNAgradients/cond_1/rnn/while/Select_1_grad/tuple/control_dependency2gradients/cond_1/rnn/while/ExpandDims_grad/Reshape*
T0*B
_class8
64loc:@gradients/cond_1/rnn/while/Select_1_grad/Select*
N*'
_output_shapes
:         
w
&gradients/cond_1/ExpandDims_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
╟
(gradients/cond_1/ExpandDims_grad/ReshapeReshape3gradients/cond_1/rnn/while/add_1/Enter_grad/b_acc_3&gradients/cond_1/ExpandDims_grad/Shape*
_output_shapes

:*
T0*
Tshape0
Н
8gradients/cond_1/rnn/while/Switch_4_grad_1/NextIterationNextIterationgradients/AddN_4*
T0*'
_output_shapes
:         
r
gradients/Switch_12Switchtransitions/readcond_1/pred_id*
T0*(
_output_shapes
::
f
gradients/Shape_7Shapegradients/Switch_12:1*
_output_shapes
:*
T0*
out_type0
\
gradients/zeros_7/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
А
gradients/zeros_7Fillgradients/Shape_7gradients/zeros_7/Const*
T0*

index_type0*
_output_shapes

:
л
1gradients/cond_1/ExpandDims/Switch_grad/cond_gradMerge(gradients/cond_1/ExpandDims_grad/Reshapegradients/zeros_7*
N* 
_output_shapes
:: *
T0
Ё
gradients/AddN_5AddN.gradients/cond/Reshape_4/Switch_grad/cond_grad1gradients/cond_1/ExpandDims/Switch_grad/cond_grad*
T0*A
_class7
53loc:@gradients/cond/Reshape_4/Switch_grad/cond_grad*
N*
_output_shapes

:
┴
gradients/AddN_6AddN*gradients/cond/Shape/Switch_grad/cond_grad,gradients/cond/Shape_1/Switch_grad/cond_gradgradients/Slice_grad/Pad,gradients/cond_1/Slice/Switch_grad/cond_grad*4
_output_shapes"
 :                  *
T0*=
_class3
1/loc:@gradients/cond/Shape/Switch_grad/cond_grad*
N
}
,gradients/biLSTM_layers/Reshape_1_grad/ShapeShapebiLSTM_layers/add*
out_type0*
_output_shapes
:*
T0
╣
.gradients/biLSTM_layers/Reshape_1_grad/ReshapeReshapegradients/AddN_6,gradients/biLSTM_layers/Reshape_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
z
&gradients/biLSTM_layers/add_grad/ShapeShapebiLSTM_layers/MatMul*
_output_shapes
:*
T0*
out_type0
r
(gradients/biLSTM_layers/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
▐
6gradients/biLSTM_layers/add_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/biLSTM_layers/add_grad/Shape(gradients/biLSTM_layers/add_grad/Shape_1*2
_output_shapes 
:         :         *
T0
╙
$gradients/biLSTM_layers/add_grad/SumSum.gradients/biLSTM_layers/Reshape_1_grad/Reshape6gradients/biLSTM_layers/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
┴
(gradients/biLSTM_layers/add_grad/ReshapeReshape$gradients/biLSTM_layers/add_grad/Sum&gradients/biLSTM_layers/add_grad/Shape*'
_output_shapes
:         *
T0*
Tshape0
╫
&gradients/biLSTM_layers/add_grad/Sum_1Sum.gradients/biLSTM_layers/Reshape_1_grad/Reshape8gradients/biLSTM_layers/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
║
*gradients/biLSTM_layers/add_grad/Reshape_1Reshape&gradients/biLSTM_layers/add_grad/Sum_1(gradients/biLSTM_layers/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
С
1gradients/biLSTM_layers/add_grad/tuple/group_depsNoOp)^gradients/biLSTM_layers/add_grad/Reshape+^gradients/biLSTM_layers/add_grad/Reshape_1
Т
9gradients/biLSTM_layers/add_grad/tuple/control_dependencyIdentity(gradients/biLSTM_layers/add_grad/Reshape2^gradients/biLSTM_layers/add_grad/tuple/group_deps*;
_class1
/-loc:@gradients/biLSTM_layers/add_grad/Reshape*'
_output_shapes
:         *
T0
Л
;gradients/biLSTM_layers/add_grad/tuple/control_dependency_1Identity*gradients/biLSTM_layers/add_grad/Reshape_12^gradients/biLSTM_layers/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/biLSTM_layers/add_grad/Reshape_1*
_output_shapes
:
т
*gradients/biLSTM_layers/MatMul_grad/MatMulMatMul9gradients/biLSTM_layers/add_grad/tuple/control_dependencybiLSTM_layers/W_out/read*(
_output_shapes
:         А*
transpose_a( *
transpose_b(*
T0
╪
,gradients/biLSTM_layers/MatMul_grad/MatMul_1MatMulbiLSTM_layers/Reshape9gradients/biLSTM_layers/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	А*
transpose_a(
Ш
4gradients/biLSTM_layers/MatMul_grad/tuple/group_depsNoOp+^gradients/biLSTM_layers/MatMul_grad/MatMul-^gradients/biLSTM_layers/MatMul_grad/MatMul_1
Э
<gradients/biLSTM_layers/MatMul_grad/tuple/control_dependencyIdentity*gradients/biLSTM_layers/MatMul_grad/MatMul5^gradients/biLSTM_layers/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/biLSTM_layers/MatMul_grad/MatMul*(
_output_shapes
:         А
Ъ
>gradients/biLSTM_layers/MatMul_grad/tuple/control_dependency_1Identity,gradients/biLSTM_layers/MatMul_grad/MatMul_15^gradients/biLSTM_layers/MatMul_grad/tuple/group_deps*
_output_shapes
:	А*
T0*?
_class5
31loc:@gradients/biLSTM_layers/MatMul_grad/MatMul_1
Г
*gradients/biLSTM_layers/Reshape_grad/ShapeShapebiLSTM_layers/dropout/mul*
T0*
out_type0*
_output_shapes
:
я
,gradients/biLSTM_layers/Reshape_grad/ReshapeReshape<gradients/biLSTM_layers/MatMul_grad/tuple/control_dependency*gradients/biLSTM_layers/Reshape_grad/Shape*5
_output_shapes#
!:                  А*
T0*
Tshape0
Р
.gradients/biLSTM_layers/dropout/mul_grad/ShapeShapebiLSTM_layers/dropout/div*
out_type0*#
_output_shapes
:         *
T0
Ф
0gradients/biLSTM_layers/dropout/mul_grad/Shape_1ShapebiLSTM_layers/dropout/Floor*
T0*
out_type0*#
_output_shapes
:         
Ў
>gradients/biLSTM_layers/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/biLSTM_layers/dropout/mul_grad/Shape0gradients/biLSTM_layers/dropout/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
б
,gradients/biLSTM_layers/dropout/mul_grad/MulMul,gradients/biLSTM_layers/Reshape_grad/ReshapebiLSTM_layers/dropout/Floor*
T0*
_output_shapes
:
с
,gradients/biLSTM_layers/dropout/mul_grad/SumSum,gradients/biLSTM_layers/dropout/mul_grad/Mul>gradients/biLSTM_layers/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
╩
0gradients/biLSTM_layers/dropout/mul_grad/ReshapeReshape,gradients/biLSTM_layers/dropout/mul_grad/Sum.gradients/biLSTM_layers/dropout/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:
б
.gradients/biLSTM_layers/dropout/mul_grad/Mul_1MulbiLSTM_layers/dropout/div,gradients/biLSTM_layers/Reshape_grad/Reshape*
T0*
_output_shapes
:
ч
.gradients/biLSTM_layers/dropout/mul_grad/Sum_1Sum.gradients/biLSTM_layers/dropout/mul_grad/Mul_1@gradients/biLSTM_layers/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
╨
2gradients/biLSTM_layers/dropout/mul_grad/Reshape_1Reshape.gradients/biLSTM_layers/dropout/mul_grad/Sum_10gradients/biLSTM_layers/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
й
9gradients/biLSTM_layers/dropout/mul_grad/tuple/group_depsNoOp1^gradients/biLSTM_layers/dropout/mul_grad/Reshape3^gradients/biLSTM_layers/dropout/mul_grad/Reshape_1
г
Agradients/biLSTM_layers/dropout/mul_grad/tuple/control_dependencyIdentity0gradients/biLSTM_layers/dropout/mul_grad/Reshape:^gradients/biLSTM_layers/dropout/mul_grad/tuple/group_deps*C
_class9
75loc:@gradients/biLSTM_layers/dropout/mul_grad/Reshape*
_output_shapes
:*
T0
й
Cgradients/biLSTM_layers/dropout/mul_grad/tuple/control_dependency_1Identity2gradients/biLSTM_layers/dropout/mul_grad/Reshape_1:^gradients/biLSTM_layers/dropout/mul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/biLSTM_layers/dropout/mul_grad/Reshape_1*
_output_shapes
:
В
.gradients/biLSTM_layers/dropout/div_grad/ShapeShapebiLSTM_layers/concat*
T0*
out_type0*
_output_shapes
:
В
0gradients/biLSTM_layers/dropout/div_grad/Shape_1Shape	keep_prob*
T0*
out_type0*#
_output_shapes
:         
Ў
>gradients/biLSTM_layers/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/biLSTM_layers/dropout/div_grad/Shape0gradients/biLSTM_layers/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:         :         
м
0gradients/biLSTM_layers/dropout/div_grad/RealDivRealDivAgradients/biLSTM_layers/dropout/mul_grad/tuple/control_dependency	keep_prob*
T0*
_output_shapes
:
х
,gradients/biLSTM_layers/dropout/div_grad/SumSum0gradients/biLSTM_layers/dropout/div_grad/RealDiv>gradients/biLSTM_layers/dropout/div_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ч
0gradients/biLSTM_layers/dropout/div_grad/ReshapeReshape,gradients/biLSTM_layers/dropout/div_grad/Sum.gradients/biLSTM_layers/dropout/div_grad/Shape*
T0*
Tshape0*5
_output_shapes#
!:                  А
Й
,gradients/biLSTM_layers/dropout/div_grad/NegNegbiLSTM_layers/concat*
T0*5
_output_shapes#
!:                  А
Щ
2gradients/biLSTM_layers/dropout/div_grad/RealDiv_1RealDiv,gradients/biLSTM_layers/dropout/div_grad/Neg	keep_prob*
T0*
_output_shapes
:
Я
2gradients/biLSTM_layers/dropout/div_grad/RealDiv_2RealDiv2gradients/biLSTM_layers/dropout/div_grad/RealDiv_1	keep_prob*
_output_shapes
:*
T0
═
,gradients/biLSTM_layers/dropout/div_grad/mulMulAgradients/biLSTM_layers/dropout/mul_grad/tuple/control_dependency2gradients/biLSTM_layers/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
х
.gradients/biLSTM_layers/dropout/div_grad/Sum_1Sum,gradients/biLSTM_layers/dropout/div_grad/mul@gradients/biLSTM_layers/dropout/div_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
╨
2gradients/biLSTM_layers/dropout/div_grad/Reshape_1Reshape.gradients/biLSTM_layers/dropout/div_grad/Sum_10gradients/biLSTM_layers/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
й
9gradients/biLSTM_layers/dropout/div_grad/tuple/group_depsNoOp1^gradients/biLSTM_layers/dropout/div_grad/Reshape3^gradients/biLSTM_layers/dropout/div_grad/Reshape_1
└
Agradients/biLSTM_layers/dropout/div_grad/tuple/control_dependencyIdentity0gradients/biLSTM_layers/dropout/div_grad/Reshape:^gradients/biLSTM_layers/dropout/div_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/biLSTM_layers/dropout/div_grad/Reshape*5
_output_shapes#
!:                  А
й
Cgradients/biLSTM_layers/dropout/div_grad/tuple/control_dependency_1Identity2gradients/biLSTM_layers/dropout/div_grad/Reshape_1:^gradients/biLSTM_layers/dropout/div_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/biLSTM_layers/dropout/div_grad/Reshape_1*
_output_shapes
:
j
(gradients/biLSTM_layers/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Щ
'gradients/biLSTM_layers/concat_grad/modFloorModbiLSTM_layers/concat/axis(gradients/biLSTM_layers/concat_grad/Rank*
T0*
_output_shapes
: 
Ъ
)gradients/biLSTM_layers/concat_grad/ShapeShape1biLSTM_layers/bidirectional_rnn/fw/fw/transpose_1*
_output_shapes
:*
T0*
out_type0
╩
*gradients/biLSTM_layers/concat_grad/ShapeNShapeN1biLSTM_layers/bidirectional_rnn/fw/fw/transpose_1biLSTM_layers/ReverseSequence*
T0*
out_type0*
N* 
_output_shapes
::
ю
0gradients/biLSTM_layers/concat_grad/ConcatOffsetConcatOffset'gradients/biLSTM_layers/concat_grad/mod*gradients/biLSTM_layers/concat_grad/ShapeN,gradients/biLSTM_layers/concat_grad/ShapeN:1*
N* 
_output_shapes
::
и
)gradients/biLSTM_layers/concat_grad/SliceSliceAgradients/biLSTM_layers/dropout/div_grad/tuple/control_dependency0gradients/biLSTM_layers/concat_grad/ConcatOffset*gradients/biLSTM_layers/concat_grad/ShapeN*
T0*
Index0*=
_output_shapes+
):'                           
о
+gradients/biLSTM_layers/concat_grad/Slice_1SliceAgradients/biLSTM_layers/dropout/div_grad/tuple/control_dependency2gradients/biLSTM_layers/concat_grad/ConcatOffset:1,gradients/biLSTM_layers/concat_grad/ShapeN:1*=
_output_shapes+
):'                           *
T0*
Index0
Ц
4gradients/biLSTM_layers/concat_grad/tuple/group_depsNoOp*^gradients/biLSTM_layers/concat_grad/Slice,^gradients/biLSTM_layers/concat_grad/Slice_1
и
<gradients/biLSTM_layers/concat_grad/tuple/control_dependencyIdentity)gradients/biLSTM_layers/concat_grad/Slice5^gradients/biLSTM_layers/concat_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/biLSTM_layers/concat_grad/Slice*5
_output_shapes#
!:                  А
о
>gradients/biLSTM_layers/concat_grad/tuple/control_dependency_1Identity+gradients/biLSTM_layers/concat_grad/Slice_15^gradients/biLSTM_layers/concat_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/biLSTM_layers/concat_grad/Slice_1*5
_output_shapes#
!:                  А
╝
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/transpose_1_grad/InvertPermutationInvertPermutation.biLSTM_layers/bidirectional_rnn/fw/fw/concat_2*
T0*
_output_shapes
:
╢
Jgradients/biLSTM_layers/bidirectional_rnn/fw/fw/transpose_1_grad/transpose	Transpose<gradients/biLSTM_layers/concat_grad/tuple/control_dependencyRgradients/biLSTM_layers/bidirectional_rnn/fw/fw/transpose_1_grad/InvertPermutation*5
_output_shapes#
!:                  А*
Tperm0*
T0
А
<gradients/biLSTM_layers/ReverseSequence_grad/ReverseSequenceReverseSequence>gradients/biLSTM_layers/concat_grad/tuple/control_dependency_1Sum*
	batch_dim *
T0*
seq_dim*5
_output_shapes#
!:                  А*

Tlen0
Є
{gradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV31biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray2biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_2*
source	gradients*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray*
_output_shapes

:: 
Ь
wgradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentity2biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_2|^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray*
_output_shapes
: 
╗
Бgradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3{gradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3<biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayStack/rangeJgradients/biLSTM_layers/bidirectional_rnn/fw/fw/transpose_1_grad/transposewgradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
╝
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/transpose_1_grad/InvertPermutationInvertPermutation.biLSTM_layers/bidirectional_rnn/bw/bw/concat_2*
_output_shapes
:*
T0
╢
Jgradients/biLSTM_layers/bidirectional_rnn/bw/bw/transpose_1_grad/transpose	Transpose<gradients/biLSTM_layers/ReverseSequence_grad/ReverseSequenceRgradients/biLSTM_layers/bidirectional_rnn/bw/bw/transpose_1_grad/InvertPermutation*
T0*5
_output_shapes#
!:                  А*
Tperm0
Є
{gradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV31biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray2biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_2*
source	gradients*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray*
_output_shapes

:: 
Ь
wgradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentity2biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_2|^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray*
_output_shapes
: 
╗
Бgradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3{gradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3<biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayStack/rangeJgradients/biLSTM_layers/bidirectional_rnn/bw/bw/transpose_1_grad/transposewgradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
_output_shapes
: *
T0
И
gradients/zeros_like	ZerosLike2biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_3*
T0*(
_output_shapes
:         А
К
gradients/zeros_like_1	ZerosLike2biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_4*
T0*(
_output_shapes
:         А
∙
Hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_2_grad/b_exitEnterБgradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
is_constant( *
parallel_iterations *
_output_shapes
: *S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0
Э
Hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_3_grad/b_exitEntergradients/zeros_like*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:         А*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
Я
Hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_4_grad/b_exitEntergradients/zeros_like_1*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:         А*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
К
gradients/zeros_like_2	ZerosLike2biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_3*(
_output_shapes
:         А*
T0
К
gradients/zeros_like_3	ZerosLike2biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_4*
T0*(
_output_shapes
:         А
а
Lgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switchMergeHgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_2_grad/b_exitSgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_2_grad_1/NextIteration*
N*
_output_shapes
: : *
T0
▓
Lgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switchMergeHgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_3_grad/b_exitSgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_3_grad_1/NextIteration*
T0*
N**
_output_shapes
:         А: 
▓
Lgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switchMergeHgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_4_grad/b_exitSgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_4_grad_1/NextIteration*
T0*
N**
_output_shapes
:         А: 
∙
Hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_2_grad/b_exitEnterБgradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
Я
Hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_3_grad/b_exitEntergradients/zeros_like_2*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:         А*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
Я
Hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_4_grad/b_exitEntergradients/zeros_like_3*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:         А*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
║
Igradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_2_grad/SwitchSwitchLgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switchgradients/b_count_6*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch*
_output_shapes
: : *
T0
з
Sgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/group_depsNoOpJ^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_2_grad/Switch
К
[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependencyIdentityIgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_2_grad/SwitchT^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch*
_output_shapes
: 
О
]gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1IdentityKgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_2_grad/Switch:1T^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch*
_output_shapes
: 
▐
Igradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_3_grad/SwitchSwitchLgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switchgradients/b_count_6*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switch*<
_output_shapes*
(:         А:         А
з
Sgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/group_depsNoOpJ^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_3_grad/Switch
Ь
[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependencyIdentityIgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_3_grad/SwitchT^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switch*(
_output_shapes
:         А
а
]gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency_1IdentityKgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_3_grad/Switch:1T^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_3_grad/b_switch*(
_output_shapes
:         А
▐
Igradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_4_grad/SwitchSwitchLgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switchgradients/b_count_6*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switch*<
_output_shapes*
(:         А:         А
з
Sgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/group_depsNoOpJ^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_4_grad/Switch
Ь
[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependencyIdentityIgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_4_grad/SwitchT^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switch
а
]gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency_1IdentityKgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_4_grad/Switch:1T^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_4_grad/b_switch
а
Lgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switchMergeHgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_2_grad/b_exitSgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_2_grad_1/NextIteration*
N*
_output_shapes
: : *
T0
▓
Lgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switchMergeHgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_3_grad/b_exitSgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_3_grad_1/NextIteration*
T0*
N**
_output_shapes
:         А: 
▓
Lgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switchMergeHgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_4_grad/b_exitSgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_4_grad_1/NextIteration*
T0*
N**
_output_shapes
:         А: 
═
Ggradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_2_grad/ExitExit[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency*
_output_shapes
: *
T0
▀
Ggradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_3_grad/ExitExit[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
▀
Ggradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_4_grad/ExitExit[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
╗
Igradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_2_grad/SwitchSwitchLgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switchgradients/b_count_10*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch*
_output_shapes
: : 
з
Sgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/group_depsNoOpJ^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_2_grad/Switch
К
[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependencyIdentityIgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_2_grad/SwitchT^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch*
_output_shapes
: 
О
]gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1IdentityKgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_2_grad/Switch:1T^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/group_deps*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch*
_output_shapes
: *
T0
▀
Igradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_3_grad/SwitchSwitchLgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switchgradients/b_count_10*<
_output_shapes*
(:         А:         А*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switch
з
Sgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/group_depsNoOpJ^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_3_grad/Switch
Ь
[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependencyIdentityIgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_3_grad/SwitchT^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switch*(
_output_shapes
:         А
а
]gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency_1IdentityKgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_3_grad/Switch:1T^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_3_grad/b_switch*(
_output_shapes
:         А
▀
Igradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_4_grad/SwitchSwitchLgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switchgradients/b_count_10*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switch*<
_output_shapes*
(:         А:         А*
T0
з
Sgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/group_depsNoOpJ^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_4_grad/Switch
Ь
[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependencyIdentityIgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_4_grad/SwitchT^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switch*(
_output_shapes
:         А
а
]gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency_1IdentityKgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_4_grad/Switch:1T^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_4_grad/b_switch
Г
Аgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Жgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter]gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1*
source	gradients*N
_classD
B@loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
_output_shapes

:: 
╗
Жgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter1biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray*
is_constant(*
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*N
_classD
B@loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2*
parallel_iterations 
▄
|gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity]gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1Б^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*N
_classD
B@loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2
║
pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3Аgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3{gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2|gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*(
_output_shapes
:         А
М
vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*
dtype0*
_output_shapes
: *
valueB :
         *I
_class?
=;loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_1
Г
vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*

stack_name *
_output_shapes
:*
	elem_type0*I
_class?
=;loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_1
Х
vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEntervgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(
Б
|gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter6biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_1^gradients/Add_1*
T0*
_output_shapes
: *
swap_memory( 
╚
{gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2Бgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
: *
	elem_type0
л
Бgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEntervgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
Й
wgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTrigger|^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Z^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2V^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Z^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2l^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2n^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1Z^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2\^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2l^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2n^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1j^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2l^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1Z^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2l^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2n^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1Z^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2\^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2j^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2`^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2^^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2
╩
ogradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOp^^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1q^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
а
wgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentitypgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3p^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*Г
_classy
wuloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*(
_output_shapes
:         А
╪
ygradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1Identity]gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_2_grad/tuple/control_dependency_1p^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_2_grad/b_switch*
_output_shapes
: 
г
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/zeros_grad/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
▓
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/zeros_grad/SumSumGgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_3_grad/ExitRgradients/biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/zeros_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
щ
Ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like	ZerosLikeYgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:         А
ъ
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/ConstConst*
valueB :
         *I
_class?
=;loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_3*
dtype0*
_output_shapes
: 
┐
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_accStackV2Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/Const*I
_class?
=;loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_3*

stack_name *
_output_shapes
:*
	elem_type0
╤
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
╧
Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPushV2StackPushV2Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/Enter6biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_3^gradients/Add_1*
T0*(
_output_shapes
:         А*
swap_memory( 
Х
Ygradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2
StackPopV2_gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*(
_output_shapes
:         А
ц
_gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_acc*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(
Н
Jgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/SelectSelectUgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2]gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency_1Ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like*(
_output_shapes
:         А*
T0
ш
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/ConstConst*
dtype0*
_output_shapes
: *
valueB :
         *K
_classA
?=loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/GreaterEqual
╣
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_accStackV2Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Const*K
_classA
?=loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/GreaterEqual*

stack_name *
_output_shapes
:*
	elem_type0

╔
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/EnterEnterPgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
─
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPushV2StackPushV2Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Enter8biLSTM_layers/bidirectional_rnn/fw/fw/while/GreaterEqual^gradients/Add_1*#
_output_shapes
:         *
swap_memory( *
T0

И
Ugradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2
StackPopV2[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2/Enter^gradients/Sub_1*
	elem_type0
*#
_output_shapes
:         
▐
[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2/EnterEnterPgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0
П
Lgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1SelectUgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like]gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_3_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
°
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/group_depsNoOpK^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/SelectM^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1
Э
\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependencyIdentityJgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/SelectU^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*]
_classS
QOloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select
г
^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependency_1IdentityLgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1U^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1
е
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/zeros_1_grad/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
╢
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/zeros_1_grad/SumSumGgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_4_grad/ExitTgradients/biLSTM_layers/bidirectional_rnn/fw/fw/LSTMCellZeroState/zeros_1_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
щ
Ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like	ZerosLikeYgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2*(
_output_shapes
:         А*
T0
ъ
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/ConstConst*
valueB :
         *I
_class?
=;loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_4*
dtype0*
_output_shapes
: 
┐
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_accStackV2Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/Const*I
_class?
=;loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_4*

stack_name *
_output_shapes
:*
	elem_type0
╤
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_acc*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(
╧
Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPushV2StackPushV2Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/Enter6biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_4^gradients/Add_1*
T0*(
_output_shapes
:         А*
swap_memory( 
Х
Ygradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2
StackPopV2_gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*(
_output_shapes
:         А
ц
_gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0
Н
Jgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/SelectSelectUgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2]gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency_1Ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like*(
_output_shapes
:         А*
T0
П
Lgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1SelectUgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like]gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_4_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
°
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/group_depsNoOpK^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/SelectM^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1
Э
\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependencyIdentityJgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/SelectU^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/group_deps*]
_classS
QOloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/Select*(
_output_shapes
:         А*
T0
г
^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependency_1IdentityLgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1U^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1*(
_output_shapes
:         А
═
Ggradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_2_grad/ExitExit[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency*
T0*
_output_shapes
: 
▀
Ggradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_3_grad/ExitExit[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
▀
Ggradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_4_grad/ExitExit[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
Є
Lgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like	ZerosLikeRgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like/Enter^gradients/Sub_1*
T0*(
_output_shapes
:         А
╛
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like/EnterEnter+biLSTM_layers/bidirectional_rnn/fw/fw/zeros*
T0*
is_constant(*
parallel_iterations *(
_output_shapes
:         А*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
г
Hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/SelectSelectUgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2wgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyLgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/zeros_like*
T0*(
_output_shapes
:         А
е
Jgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/Select_1SelectUgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPopV2Lgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/zeros_likewgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
Є
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/tuple/group_depsNoOpI^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/SelectK^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/Select_1
Х
Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/tuple/control_dependencyIdentityHgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/SelectS^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/Select*(
_output_shapes
:         А
Ы
\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/tuple/control_dependency_1IdentityJgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/Select_1S^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/tuple/group_deps*]
_classS
QOloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/Select_1*(
_output_shapes
:         А*
T0
Г
Аgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Жgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter]gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1*
source	gradients*N
_classD
B@loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
_output_shapes

:: 
╗
Жgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter1biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray*
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*N
_classD
B@loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
parallel_iterations *
is_constant(
▄
|gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity]gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1Б^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*N
_classD
B@loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2*
_output_shapes
: *
T0
║
pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3Аgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3{gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2|gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*(
_output_shapes
:         А
М
vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*
_output_shapes
: *
valueB :
         *I
_class?
=;loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_1*
dtype0
Г
vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*I
_class?
=;loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_1*

stack_name *
_output_shapes
:*
	elem_type0
Х
vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEntervgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0
Б
|gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter6biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_1^gradients/Add_2*
_output_shapes
: *
swap_memory( *
T0
╚
{gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2Бgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub_2*
_output_shapes
: *
	elem_type0
л
Бgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEntervgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0
Й
wgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTrigger|^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Z^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2V^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Z^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2l^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2n^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1Z^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2\^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2l^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2n^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1j^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2l^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1Z^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2l^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2n^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1Z^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2\^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2j^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2`^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2^^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2
╩
ogradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOp^^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1q^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
а
wgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentitypgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3p^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*Г
_classy
wuloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*(
_output_shapes
:         А
╪
ygradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1Identity]gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_2_grad/tuple/control_dependency_1p^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_2_grad/b_switch*
_output_shapes
: 
г
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/zeros_grad/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
▓
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/zeros_grad/SumSumGgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_3_grad/ExitRgradients/biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/zeros_grad/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
щ
Ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like	ZerosLikeYgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2*(
_output_shapes
:         А*
T0
ъ
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/ConstConst*
valueB :
         *I
_class?
=;loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_3*
dtype0*
_output_shapes
: 
┐
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_accStackV2Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/Const*I
_class?
=;loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_3*

stack_name *
_output_shapes
:*
	elem_type0
╤
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
╧
Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPushV2StackPushV2Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/Enter6biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_3^gradients/Add_2*(
_output_shapes
:         А*
swap_memory( *
T0
Х
Ygradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2
StackPopV2_gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2/Enter^gradients/Sub_2*(
_output_shapes
:         А*
	elem_type0
ц
_gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_acc*
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant(*
parallel_iterations 
Н
Jgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/SelectSelectUgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2]gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency_1Ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like*(
_output_shapes
:         А*
T0
ш
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/ConstConst*
dtype0*
_output_shapes
: *
valueB :
         *K
_classA
?=loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/GreaterEqual
╣
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_accStackV2Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Const*K
_classA
?=loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/GreaterEqual*

stack_name *
_output_shapes
:*
	elem_type0

╔
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/EnterEnterPgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0
─
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPushV2StackPushV2Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Enter8biLSTM_layers/bidirectional_rnn/bw/bw/while/GreaterEqual^gradients/Add_2*
T0
*#
_output_shapes
:         *
swap_memory( 
И
Ugradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2
StackPopV2[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2/Enter^gradients/Sub_2*#
_output_shapes
:         *
	elem_type0

▐
[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2/EnterEnterPgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc*
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant(*
parallel_iterations 
П
Lgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1SelectUgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like]gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_3_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
°
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/group_depsNoOpK^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/SelectM^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1
Э
\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependencyIdentityJgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/SelectU^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/group_deps*]
_classS
QOloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select*(
_output_shapes
:         А*
T0
г
^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependency_1IdentityLgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1U^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1*(
_output_shapes
:         А
е
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/zeros_1_grad/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
╢
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/zeros_1_grad/SumSumGgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_4_grad/ExitTgradients/biLSTM_layers/bidirectional_rnn/bw/bw/LSTMCellZeroState/zeros_1_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
щ
Ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like	ZerosLikeYgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2*(
_output_shapes
:         А*
T0
ъ
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/ConstConst*
valueB :
         *I
_class?
=;loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_4*
dtype0*
_output_shapes
: 
┐
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_accStackV2Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/Const*I
_class?
=;loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_4*

stack_name *
_output_shapes
:*
	elem_type0
╤
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
╧
Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPushV2StackPushV2Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/Enter6biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_4^gradients/Add_2*
T0*(
_output_shapes
:         А*
swap_memory( 
Х
Ygradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2
StackPopV2_gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2/Enter^gradients/Sub_2*(
_output_shapes
:         А*
	elem_type0
ц
_gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
Н
Jgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/SelectSelectUgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2]gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency_1Ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like*
T0*(
_output_shapes
:         А
П
Lgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1SelectUgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like]gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_4_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
°
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/group_depsNoOpK^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/SelectM^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1
Э
\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependencyIdentityJgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/SelectU^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*]
_classS
QOloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/Select
г
^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependency_1IdentityLgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1U^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1*(
_output_shapes
:         А
╕
Mgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter_grad/ShapeShape+biLSTM_layers/bidirectional_rnn/fw/fw/zeros*
T0*
out_type0*
_output_shapes
:
Ш
Sgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╛
Mgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter_grad/zerosFillMgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter_grad/ShapeSgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:         А
█
Mgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter_grad/b_accEnterMgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter_grad/zeros*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:         А*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
╝
Ogradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter_grad/b_acc_1MergeMgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter_grad/b_accUgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter_grad/NextIteration*
T0*
N**
_output_shapes
:         А: 
Е
Ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter_grad/SwitchSwitchOgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter_grad/b_acc_1gradients/b_count_6*<
_output_shapes*
(:         А:         А*
T0
│
Kgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter_grad/AddAddPgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter_grad/Switch:1Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
ц
Ugradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter_grad/NextIterationNextIterationKgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter_grad/Add*
T0*(
_output_shapes
:         А
┌
Ogradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter_grad/b_acc_2ExitNgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter_grad/Switch*
T0*(
_output_shapes
:         А
є
gradients/AddN_7AddN^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependency_1\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_grad/tuple/control_dependency_1*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/Select_1*
N*(
_output_shapes
:         А
╧
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/ShapeShape?biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2*
T0*
out_type0*
_output_shapes
:
╬
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1Shape<biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1*
T0*
out_type0*
_output_shapes
:
Т
`gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgskgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2mgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
Ц
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/ConstConst*
valueB :
         *c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape*
dtype0*
_output_shapes
: 
¤
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_accStackV2fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const*c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
ї
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterEnterfgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(*
parallel_iterations 
 
lgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterPgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape^gradients/Add_1*
T0*
_output_shapes
:*
swap_memory( 
л
kgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2qgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
К
qgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterfgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
Ъ
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1Const*
valueB :
         *e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1*
dtype0*
_output_shapes
: 
Г
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1*e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
∙
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Enterhgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
Е
ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1^gradients/Add_1*
_output_shapes
:*
swap_memory( *
T0
п
mgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2sgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
О
sgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterhgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(
ї
Ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/MulMulgradients/AddN_7Ygradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2*
T0*(
_output_shapes
:         А
Ё
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/ConstConst*
valueB :
         *O
_classE
CAloc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1*
dtype0*
_output_shapes
: 
┼
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_accStackV2Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/Const*O
_classE
CAloc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1*

stack_name *
_output_shapes
:*
	elem_type0
╤
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
╒
Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/Enter<biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1^gradients/Add_1*(
_output_shapes
:         А*
swap_memory( *
T0
Х
Ygradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2_gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub_1*(
_output_shapes
:         А*
	elem_type0
ц
_gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_acc*
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(*
parallel_iterations 
╟
Ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/SumSumNgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul`gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
у
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/ReshapeReshapeNgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Sumkgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*0
_output_shapes
:                  
∙
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1Mul[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2gradients/AddN_7*
T0*(
_output_shapes
:         А
ї
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/ConstConst*
valueB :
         *R
_classH
FDloc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2*
dtype0*
_output_shapes
: 
╠
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_accStackV2Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/Const*
	elem_type0*R
_classH
FDloc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:
╒
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/EnterEnterVgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0
▄
\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/Enter?biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2^gradients/Add_1*(
_output_shapes
:         А*
swap_memory( *
T0
Щ
[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2agradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*(
_output_shapes
:         А*
	elem_type0
ъ
agradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterVgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
═
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Sum_1SumPgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1bgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
щ
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Reshape_1ReshapePgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Sum_1mgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*0
_output_shapes
:                  *
T0*
Tshape0
П
[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/group_depsNoOpS^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/ReshapeU^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Reshape_1
╗
cgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/control_dependencyIdentityRgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Reshape\^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/group_deps*e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Reshape*(
_output_shapes
:         А*
T0
┴
egradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/control_dependency_1IdentityTgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Reshape_1\^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Reshape_1*(
_output_shapes
:         А
А
Sgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_2_grad_1/NextIterationNextIterationygradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
Є
Lgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like	ZerosLikeRgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like/Enter^gradients/Sub_2*(
_output_shapes
:         А*
T0
╛
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like/EnterEnter+biLSTM_layers/bidirectional_rnn/bw/bw/zeros*
T0*
is_constant(*
parallel_iterations *(
_output_shapes
:         А*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
г
Hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/SelectSelectUgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2wgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyLgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/zeros_like*
T0*(
_output_shapes
:         А
е
Jgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/Select_1SelectUgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPopV2Lgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/zeros_likewgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
Є
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/tuple/group_depsNoOpI^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/SelectK^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/Select_1
Х
Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/tuple/control_dependencyIdentityHgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/SelectS^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*[
_classQ
OMloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/Select
Ы
\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/tuple/control_dependency_1IdentityJgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/Select_1S^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/Select_1*(
_output_shapes
:         А
С
@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/zeros_grad/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Ц
>gradients/biLSTM_layers/bidirectional_rnn/fw/fw/zeros_grad/SumSumOgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter_grad/b_acc_2@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/zeros_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
▐
Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2cgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
╒
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1_grad/TanhGradTanhGradYgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPopV2egradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
╕
Mgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter_grad/ShapeShape+biLSTM_layers/bidirectional_rnn/bw/bw/zeros*
T0*
out_type0*
_output_shapes
:
Ш
Sgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
╛
Mgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter_grad/zerosFillMgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter_grad/ShapeSgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter_grad/zeros/Const*(
_output_shapes
:         А*
T0*

index_type0
█
Mgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter_grad/b_accEnterMgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter_grad/zeros*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:         А*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
╝
Ogradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter_grad/b_acc_1MergeMgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter_grad/b_accUgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter_grad/NextIteration*
N**
_output_shapes
:         А: *
T0
Ж
Ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter_grad/SwitchSwitchOgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter_grad/b_acc_1gradients/b_count_10*<
_output_shapes*
(:         А:         А*
T0
│
Kgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter_grad/AddAddPgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter_grad/Switch:1Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
ц
Ugradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter_grad/NextIterationNextIterationKgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter_grad/Add*
T0*(
_output_shapes
:         А
┌
Ogradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter_grad/b_acc_2ExitNgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter_grad/Switch*
T0*(
_output_shapes
:         А
є
gradients/AddN_8AddN^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependency_1\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_grad/tuple/control_dependency_1*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/Select_1*
N*(
_output_shapes
:         А
╧
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/ShapeShape?biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2*
_output_shapes
:*
T0*
out_type0
╬
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1Shape<biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1*
T0*
out_type0*
_output_shapes
:
Т
`gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgskgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2mgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:         :         *
T0
Ц
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/ConstConst*
valueB :
         *c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape*
dtype0*
_output_shapes
: 
¤
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_accStackV2fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const*
	elem_type0*c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape*

stack_name *
_output_shapes
:
ї
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterEnterfgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
 
lgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterPgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape^gradients/Add_2*
T0*
_output_shapes
:*
swap_memory( 
л
kgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2qgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_2*
_output_shapes
:*
	elem_type0
К
qgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterfgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant(
Ъ
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1Const*
valueB :
         *e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1*
dtype0*
_output_shapes
: 
Г
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1*e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
∙
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Enterhgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
Е
ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1^gradients/Add_2*
T0*
_output_shapes
:*
swap_memory( 
п
mgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2sgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_2*
_output_shapes
:*
	elem_type0
О
sgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterhgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0
ї
Ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/MulMulgradients/AddN_8Ygradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2*(
_output_shapes
:         А*
T0
Ё
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/ConstConst*
dtype0*
_output_shapes
: *
valueB :
         *O
_classE
CAloc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1
┼
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_accStackV2Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/Const*O
_classE
CAloc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1*

stack_name *
_output_shapes
:*
	elem_type0
╤
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
╒
Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/Enter<biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1^gradients/Add_2*(
_output_shapes
:         А*
swap_memory( *
T0
Х
Ygradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2_gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub_2*(
_output_shapes
:         А*
	elem_type0
ц
_gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
╟
Ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/SumSumNgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul`gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
у
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/ReshapeReshapeNgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Sumkgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2*0
_output_shapes
:                  *
T0*
Tshape0
∙
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1Mul[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2gradients/AddN_8*
T0*(
_output_shapes
:         А
ї
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/ConstConst*
_output_shapes
: *
valueB :
         *R
_classH
FDloc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2*
dtype0
╠
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_accStackV2Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/Const*R
_classH
FDloc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2*

stack_name *
_output_shapes
:*
	elem_type0
╒
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/EnterEnterVgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant(
▄
\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/Enter?biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2^gradients/Add_2*(
_output_shapes
:         А*
swap_memory( *
T0
Щ
[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2agradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub_2*(
_output_shapes
:         А*
	elem_type0
ъ
agradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterVgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
═
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Sum_1SumPgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1bgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
щ
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Reshape_1ReshapePgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Sum_1mgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*0
_output_shapes
:                  *
T0*
Tshape0
П
[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/group_depsNoOpS^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/ReshapeU^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Reshape_1
╗
cgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/control_dependencyIdentityRgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Reshape\^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Reshape*(
_output_shapes
:         А
┴
egradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/control_dependency_1IdentityTgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Reshape_1\^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*g
_class]
[Yloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Reshape_1
А
Sgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_2_grad_1/NextIterationNextIterationygradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
ы
gradients/AddN_9AddN^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependency_1Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1_grad/TanhGrad*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select_1*
N*(
_output_shapes
:         А
╔
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/ShapeShape9biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul*
_output_shapes
:*
T0*
out_type0
═
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1Shape;biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1*
T0*
out_type0*
_output_shapes
:
Т
`gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgskgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2mgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
Ц
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/ConstConst*
valueB :
         *c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape*
dtype0*
_output_shapes
: 
¤
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_accStackV2fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const*c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
ї
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterEnterfgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(
 
lgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterPgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape^gradients/Add_1*
T0*
_output_shapes
:*
swap_memory( 
л
kgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2qgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:
К
qgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterfgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
Ъ
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *
valueB :
         *e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1
Г
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1*e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
∙
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Enterhgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0
Е
ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1^gradients/Add_1*
T0*
_output_shapes
:*
swap_memory( 
п
mgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2sgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
О
sgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterhgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
Й
Ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/SumSumgradients/AddN_9`gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
у
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/ReshapeReshapeNgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Sumkgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*0
_output_shapes
:                  
Н
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_9bgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
щ
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape_1ReshapePgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Sum_1mgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*0
_output_shapes
:                  *
T0*
Tshape0
П
[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/group_depsNoOpS^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/ReshapeU^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape_1
╗
cgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/control_dependencyIdentityRgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape\^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape
┴
egradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/control_dependency_1IdentityTgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape_1\^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Reshape_1*(
_output_shapes
:         А
С
@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/zeros_grad/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Ц
>gradients/biLSTM_layers/bidirectional_rnn/bw/bw/zeros_grad/SumSumOgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter_grad/b_acc_2@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/zeros_grad/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
▐
Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2cgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
╒
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1_grad/TanhGradTanhGradYgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPopV2egradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
╦
Ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/ShapeShape=biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid*
_output_shapes
:*
T0*
out_type0
╞
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1Shape6biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_3*
_output_shapes
:*
T0*
out_type0
М
^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsigradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2kgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
Т
dgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/ConstConst*
valueB :
         *a
_classW
USloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape*
dtype0*
_output_shapes
: 
ў
dgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStackV2dgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const*

stack_name *
_output_shapes
:*
	elem_type0*a
_classW
USloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape
ё
dgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterEnterdgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
∙
jgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2dgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterNgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape^gradients/Add_1*
T0*
_output_shapes
:*
swap_memory( 
з
igradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ogradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
Ж
ogradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterdgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
Ц
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1Const*
valueB :
         *c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1*
dtype0*
_output_shapes
: 
¤
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1StackV2fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1*
	elem_type0*c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1*

stack_name *
_output_shapes
:
ї
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Enterfgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
 
lgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1^gradients/Add_1*
T0*
_output_shapes
:*
swap_memory( 
л
kgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2qgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
К
qgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterfgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0
╞
Lgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/MulMulcgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/control_dependencyYgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:         А
┴
Lgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/SumSumLgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
▌
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/ReshapeReshapeLgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Sumigradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2*0
_output_shapes
:                  *
T0*
Tshape0
╚
Ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1MulYgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2cgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
ё
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *
valueB :
         *P
_classF
DBloc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid
╞
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_accStackV2Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/Const*
_output_shapes
:*
	elem_type0*P
_classF
DBloc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid*

stack_name 
╤
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_acc*
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(*
parallel_iterations 
╓
Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/Enter=biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid^gradients/Add_1*
T0*(
_output_shapes
:         А*
swap_memory( 
Х
Ygradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2_gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*(
_output_shapes
:         А*
	elem_type0
ц
_gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
╟
Ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Sum_1SumNgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1`gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
у
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Reshape_1ReshapeNgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Sum_1kgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*0
_output_shapes
:                  
Й
Ygradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/group_depsNoOpQ^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/ReshapeS^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Reshape_1
│
agradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/control_dependencyIdentityPgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/ReshapeZ^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Reshape*(
_output_shapes
:         А
╣
cgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/control_dependency_1IdentityRgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Reshape_1Z^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Reshape_1
╧
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/ShapeShape?biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1*
T0*
out_type0*
_output_shapes
:
╠
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1Shape:biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh*
_output_shapes
:*
T0*
out_type0
Т
`gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgskgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2mgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:         :         *
T0
Ц
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/ConstConst*
valueB :
         *c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape*
dtype0*
_output_shapes
: 
¤
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_accStackV2fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const*c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
ї
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterEnterfgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
 
lgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterPgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape^gradients/Add_1*
_output_shapes
:*
swap_memory( *
T0
л
kgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2qgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
К
qgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterfgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
Ъ
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1Const*
dtype0*
_output_shapes
: *
valueB :
         *e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1
Г
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1*
_output_shapes
:*
	elem_type0*e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1*

stack_name 
∙
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Enterhgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0
Е
ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1^gradients/Add_1*
_output_shapes
:*
swap_memory( *
T0
п
mgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2sgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_1*
	elem_type0*
_output_shapes
:
О
sgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterhgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(*
parallel_iterations 
╩
Ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/MulMulegradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/control_dependency_1Ygradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2*
T0*(
_output_shapes
:         А
ю
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/ConstConst*
valueB :
         *M
_classC
A?loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh*
dtype0*
_output_shapes
: 
├
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_accStackV2Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/Const*M
_classC
A?loc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh*

stack_name *
_output_shapes
:*
	elem_type0
╤
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_acc*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(
╙
Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/Enter:biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh^gradients/Add_1*
T0*(
_output_shapes
:         А*
swap_memory( 
Х
Ygradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2_gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_1*(
_output_shapes
:         А*
	elem_type0
ц
_gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
╟
Ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/SumSumNgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul`gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
у
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/ReshapeReshapeNgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Sumkgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*0
_output_shapes
:                  
╬
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1Mul[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2egradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
ї
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/ConstConst*
valueB :
         *R
_classH
FDloc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1*
dtype0*
_output_shapes
: 
╠
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_accStackV2Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/Const*
_output_shapes
:*
	elem_type0*R
_classH
FDloc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1*

stack_name 
╒
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/EnterEnterVgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
▄
\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/Enter?biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1^gradients/Add_1*(
_output_shapes
:         А*
swap_memory( *
T0
Щ
[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2agradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_1*
	elem_type0*(
_output_shapes
:         А
ъ
agradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterVgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(*
parallel_iterations 
═
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Sum_1SumPgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1bgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
щ
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Reshape_1ReshapePgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Sum_1mgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*0
_output_shapes
:                  
П
[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/group_depsNoOpS^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/ReshapeU^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Reshape_1
╗
cgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/control_dependencyIdentityRgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Reshape\^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Reshape*(
_output_shapes
:         А
┴
egradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/control_dependency_1IdentityTgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Reshape_1\^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Reshape_1*(
_output_shapes
:         А
ь
gradients/AddN_10AddN^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependency_1Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1_grad/TanhGrad*
T0*_
_classU
SQloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select_1*
N*(
_output_shapes
:         А
╔
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/ShapeShape9biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul*
T0*
out_type0*
_output_shapes
:
═
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1Shape;biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1*
T0*
out_type0*
_output_shapes
:
Т
`gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgskgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2mgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*2
_output_shapes 
:         :         
Ц
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/ConstConst*
valueB :
         *c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape*
dtype0*
_output_shapes
: 
¤
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_accStackV2fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const*
	elem_type0*c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape*

stack_name *
_output_shapes
:
ї
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterEnterfgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
 
lgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterPgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape^gradients/Add_2*
T0*
_output_shapes
:*
swap_memory( 
л
kgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2qgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_2*
_output_shapes
:*
	elem_type0
К
qgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterfgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant(
Ъ
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1Const*
valueB :
         *e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1*
dtype0*
_output_shapes
: 
Г
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1*e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
∙
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Enterhgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0
Е
ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1^gradients/Add_2*
_output_shapes
:*
swap_memory( *
T0
п
mgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2sgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_2*
	elem_type0*
_output_shapes
:
О
sgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterhgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
К
Ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/SumSumgradients/AddN_10`gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
у
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/ReshapeReshapeNgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Sumkgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2*
Tshape0*0
_output_shapes
:                  *
T0
О
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Sum_1Sumgradients/AddN_10bgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
щ
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape_1ReshapePgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Sum_1mgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*0
_output_shapes
:                  *
T0*
Tshape0
П
[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/group_depsNoOpS^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/ReshapeU^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape_1
╗
cgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/control_dependencyIdentityRgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape\^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape*(
_output_shapes
:         А
┴
egradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/control_dependency_1IdentityTgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape_1\^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Reshape_1*(
_output_shapes
:         А
╪
Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradYgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPopV2agradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
ў
gradients/AddN_11AddN\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/tuple/control_dependencycgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/tuple/control_dependency_1*
T0*]
_classS
QOloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select*
N*(
_output_shapes
:         А
▐
Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2cgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
╙
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_grad/TanhGradTanhGradYgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPopV2egradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
╦
Ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/ShapeShape=biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid*
T0*
out_type0*
_output_shapes
:
╞
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1Shape6biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_3*
T0*
out_type0*
_output_shapes
:
М
^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsigradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2kgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:         :         *
T0
Т
dgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/ConstConst*
valueB :
         *a
_classW
USloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape*
dtype0*
_output_shapes
: 
ў
dgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStackV2dgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const*a
_classW
USloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
ё
dgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterEnterdgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant(
∙
jgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2dgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterNgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape^gradients/Add_2*
T0*
_output_shapes
:*
swap_memory( 
з
igradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ogradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_2*
_output_shapes
:*
	elem_type0
Ж
ogradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterdgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant(*
parallel_iterations 
Ц
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1Const*
valueB :
         *c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1*
dtype0*
_output_shapes
: 
¤
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1StackV2fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1*c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
ї
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Enterfgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant(*
parallel_iterations 
 
lgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1^gradients/Add_2*
T0*
_output_shapes
:*
swap_memory( 
л
kgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2qgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_2*
_output_shapes
:*
	elem_type0
К
qgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterfgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant(
╞
Lgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/MulMulcgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/control_dependencyYgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPopV2*
T0*(
_output_shapes
:         А
┴
Lgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/SumSumLgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
▌
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/ReshapeReshapeLgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Sumigradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2*
Tshape0*0
_output_shapes
:                  *
T0
╚
Ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1MulYgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2cgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/control_dependency*(
_output_shapes
:         А*
T0
ё
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/ConstConst*
dtype0*
_output_shapes
: *
valueB :
         *P
_classF
DBloc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid
╞
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_accStackV2Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/Const*

stack_name *
_output_shapes
:*
	elem_type0*P
_classF
DBloc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid
╤
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
╓
Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/Enter=biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid^gradients/Add_2*
T0*(
_output_shapes
:         А*
swap_memory( 
Х
Ygradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2_gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub_2*(
_output_shapes
:         А*
	elem_type0
ц
_gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
╟
Ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Sum_1SumNgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1`gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
у
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Reshape_1ReshapeNgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Sum_1kgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0*0
_output_shapes
:                  
Й
Ygradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/group_depsNoOpQ^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/ReshapeS^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Reshape_1
│
agradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/control_dependencyIdentityPgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/ReshapeZ^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Reshape
╣
cgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/control_dependency_1IdentityRgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Reshape_1Z^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Reshape_1*(
_output_shapes
:         А
╧
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/ShapeShape?biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1*
T0*
out_type0*
_output_shapes
:
╠
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1Shape:biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh*
_output_shapes
:*
T0*
out_type0
Т
`gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgskgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2mgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*2
_output_shapes 
:         :         *
T0
Ц
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/ConstConst*
valueB :
         *c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape*
dtype0*
_output_shapes
: 
¤
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_accStackV2fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const*c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
ї
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterEnterfgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
 
lgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterPgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape^gradients/Add_2*
_output_shapes
:*
swap_memory( *
T0
л
kgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2qgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_2*
_output_shapes
:*
	elem_type0
К
qgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterfgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0
Ъ
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1Const*
_output_shapes
: *
valueB :
         *e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1*
dtype0
Г
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1*e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1*

stack_name *
_output_shapes
:*
	elem_type0
∙
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Enterhgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0
Е
ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1^gradients/Add_2*
T0*
_output_shapes
:*
swap_memory( 
п
mgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2sgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub_2*
_output_shapes
:*
	elem_type0
О
sgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterhgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant(*
parallel_iterations 
╩
Ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/MulMulegradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/control_dependency_1Ygradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2*
T0*(
_output_shapes
:         А
ю
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/ConstConst*
valueB :
         *M
_classC
A?loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh*
dtype0*
_output_shapes
: 
├
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_accStackV2Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/Const*
	elem_type0*M
_classC
A?loc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh*

stack_name *
_output_shapes
:
╤
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0
╙
Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/Enter:biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh^gradients/Add_2*(
_output_shapes
:         А*
swap_memory( *
T0
Х
Ygradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2_gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub_2*(
_output_shapes
:         А*
	elem_type0
ц
_gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnterTgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
╟
Ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/SumSumNgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul`gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
у
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/ReshapeReshapeNgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Sumkgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2*0
_output_shapes
:                  *
T0*
Tshape0
╬
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1Mul[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2egradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/tuple/control_dependency_1*(
_output_shapes
:         А*
T0
ї
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/ConstConst*
valueB :
         *R
_classH
FDloc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1*
dtype0*
_output_shapes
: 
╠
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_accStackV2Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/Const*R
_classH
FDloc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1*

stack_name *
_output_shapes
:*
	elem_type0
╒
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/EnterEnterVgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant(*
parallel_iterations 
▄
\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/Enter?biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1^gradients/Add_2*
T0*(
_output_shapes
:         А*
swap_memory( 
Щ
[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2agradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub_2*(
_output_shapes
:         А*
	elem_type0
ъ
agradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterVgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant(*
parallel_iterations 
═
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Sum_1SumPgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1bgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
щ
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Reshape_1ReshapePgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Sum_1mgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*0
_output_shapes
:                  *
T0*
Tshape0
П
[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/group_depsNoOpS^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/ReshapeU^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Reshape_1
╗
cgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/control_dependencyIdentityRgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Reshape\^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Reshape
┴
egradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/control_dependency_1IdentityTgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Reshape_1\^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Reshape_1*(
_output_shapes
:         А
╦
Ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/ShapeShape=biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/split:2*
T0*
out_type0*
_output_shapes
:
е
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape_1Const^gradients/Sub_1*
valueB *
dtype0*
_output_shapes
: 
ё
^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsigradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Т
dgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/ConstConst*
valueB :
         *a
_classW
USloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape*
dtype0*
_output_shapes
: 
ў
dgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_accStackV2dgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/Const*a
_classW
USloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape*

stack_name *
_output_shapes
:*
	elem_type0
ё
dgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterEnterdgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
∙
jgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2dgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterNgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape^gradients/Add_1*
_output_shapes
:*
swap_memory( *
T0
з
igradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ogradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_1*
_output_shapes
:*
	elem_type0
Ж
ogradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnterdgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
═
Lgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/SumSumXgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_grad/SigmoidGrad^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
▌
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/ReshapeReshapeLgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Sumigradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*0
_output_shapes
:                  
╤
Ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Sum_1SumXgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_grad/SigmoidGrad`gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
о
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Reshape_1ReshapeNgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Sum_1Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Й
Ygradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/tuple/group_depsNoOpQ^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/ReshapeS^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Reshape_1
│
agradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/tuple/control_dependencyIdentityPgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/ReshapeZ^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Reshape*(
_output_shapes
:         А
з
cgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/tuple/control_dependency_1IdentityRgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Reshape_1Z^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Reshape_1*
_output_shapes
: 
к
Sgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_3_grad_1/NextIterationNextIterationgradients/AddN_11*(
_output_shapes
:         А*
T0
╪
Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradYgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPopV2agradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
ў
gradients/AddN_12AddN\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/tuple/control_dependencycgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:         А*
T0*]
_classS
QOloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select
▐
Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2cgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:         А
╙
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_grad/TanhGradTanhGradYgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPopV2egradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:         А
э
Qgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concatConcatV2Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1_grad/SigmoidGradRgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_grad/TanhGradagradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/tuple/control_dependencyZgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2_grad/SigmoidGradWgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concat/Const*
T0*
N*(
_output_shapes
:         А*

Tidx0
л
Wgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concat/ConstConst^gradients/Sub_1*
_output_shapes
: *
value	B :*
dtype0
╦
Ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/ShapeShape=biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/split:2*
_output_shapes
:*
T0*
out_type0
е
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape_1Const^gradients/Sub_2*
valueB *
dtype0*
_output_shapes
: 
ё
^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsigradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Т
dgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/ConstConst*
valueB :
         *a
_classW
USloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape*
dtype0*
_output_shapes
: 
ў
dgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_accStackV2dgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/Const*
	elem_type0*a
_classW
USloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape*

stack_name *
_output_shapes
:
ё
dgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterEnterdgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
∙
jgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2dgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterNgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape^gradients/Add_2*
T0*
_output_shapes
:*
swap_memory( 
з
igradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2ogradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub_2*
_output_shapes
:*
	elem_type0
Ж
ogradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnterdgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
═
Lgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/SumSumXgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_grad/SigmoidGrad^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
▌
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/ReshapeReshapeLgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Sumigradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0*0
_output_shapes
:                  
╤
Ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Sum_1SumXgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_grad/SigmoidGrad`gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
о
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Reshape_1ReshapeNgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Sum_1Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Й
Ygradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/tuple/group_depsNoOpQ^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/ReshapeS^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Reshape_1
│
agradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/tuple/control_dependencyIdentityPgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/ReshapeZ^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Reshape*(
_output_shapes
:         А
з
cgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/tuple/control_dependency_1IdentityRgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Reshape_1Z^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Reshape_1*
_output_shapes
: 
к
Sgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_3_grad_1/NextIterationNextIterationgradients/AddN_12*
T0*(
_output_shapes
:         А
ў
Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradQgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:А
Ф
]gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/group_depsNoOpR^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concatY^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/BiasAddGrad
╜
egradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityQgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concat^^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/group_deps*d
_classZ
XVloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/split_grad/concat*(
_output_shapes
:         А*
T0
└
ggradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityXgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/BiasAddGrad^^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
э
Qgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concatConcatV2Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1_grad/SigmoidGradRgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_grad/TanhGradagradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/tuple/control_dependencyZgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2_grad/SigmoidGradWgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concat/Const*
T0*
N*(
_output_shapes
:         А*

Tidx0
л
Wgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concat/ConstConst^gradients/Sub_2*
value	B :*
dtype0*
_output_shapes
: 
Ў
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMulMatMulegradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyXgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul/Enter*(
_output_shapes
:         А*
transpose_a( *
transpose_b(*
T0
╔
Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul/EnterEnter8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
АА*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
ў
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1MatMul_gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2egradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( 
Ў
Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/ConstConst*
valueB :
         *O
_classE
CAloc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat*
dtype0*
_output_shapes
: 
╤
Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/Const*
_output_shapes
:*
	elem_type0*O
_classE
CAloc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat*

stack_name 
▌
Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/EnterEnterZgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(*
parallel_iterations 
с
`gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/Enter<biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat^gradients/Add_1*
T0*(
_output_shapes
:         А*
swap_memory( 
б
_gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2egradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub_1*(
_output_shapes
:         А*
	elem_type0
Є
egradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterZgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
Р
\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/group_depsNoOpS^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMulU^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1
╜
dgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityRgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul]^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul
╗
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityTgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1]^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
АА
з
Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_accConst*
dtype0*
_output_shapes	
:А*
valueBА*    
ц
Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterXgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes	
:А*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
╥
Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeZgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1`gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/NextIteration*
N*
_output_shapes
	:А: *
T0
Б
Ygradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchZgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_6*"
_output_shapes
:А:А*
T0
╔
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/AddAdd[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/Switch:1ggradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:А
я
`gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationVgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/Add*
_output_shapes	
:А*
T0
у
Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitYgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/Switch*
_output_shapes	
:А*
T0
ў
Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradQgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:А
Ф
]gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/group_depsNoOpR^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concatY^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/BiasAddGrad
╜
egradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityQgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concat^^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/group_deps*d
_classZ
XVloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/split_grad/concat*(
_output_shapes
:         А*
T0
└
ggradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityXgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/BiasAddGrad^^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
е
Qgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConstConst^gradients/Sub_1*
dtype0*
_output_shapes
: *
value	B :
д
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/RankConst^gradients/Sub_1*
value	B :*
dtype0*
_output_shapes
: 
б
Ogradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/modFloorModQgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConstPgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
╬
Qgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeShape=biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3*
_output_shapes
:*
T0*
out_type0
┌
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeNShapeN]gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2Ygradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPopV2*
out_type0*
N* 
_output_shapes
::*
T0
ї
Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/ConstConst*
dtype0*
_output_shapes
: *
valueB :
         *P
_classF
DBloc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3
╬
Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_accStackV2Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/Const*

stack_name *
_output_shapes
:*
	elem_type0*P
_classF
DBloc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3
┘
Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/EnterEnterXgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
▐
^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/Enter=biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3^gradients/Add_1*
T0*(
_output_shapes
:         А*
swap_memory( 
Э
]gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2cgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub_1*(
_output_shapes
:         А*
	elem_type0
ю
cgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterXgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
О
Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConcatOffsetConcatOffsetOgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/modRgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeNTgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN:1* 
_output_shapes
::*
N
╢
Qgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/SliceSlicedgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/control_dependencyXgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConcatOffsetRgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN*
T0*
Index0*0
_output_shapes
:                  
╝
Sgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Slice_1Slicedgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/control_dependencyZgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ConcatOffset:1Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN:1*0
_output_shapes
:                  *
T0*
Index0
О
\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/group_depsNoOpR^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/SliceT^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Slice_1
╗
dgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/control_dependencyIdentityQgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Slice]^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/group_deps*d
_classZ
XVloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Slice*(
_output_shapes
:         А*
T0
┴
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/control_dependency_1IdentitySgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Slice_1]^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Slice_1*(
_output_shapes
:         А
░
Wgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_accConst* 
_output_shapes
:
АА*
valueB
АА*    *
dtype0
щ
Ygradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_1EnterWgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations * 
_output_shapes
:
АА*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
╘
Ygradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_2MergeYgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_1_gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N*"
_output_shapes
:
АА: 
Й
Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/SwitchSwitchYgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_6*
T0*,
_output_shapes
:
АА:
АА
╦
Ugradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/AddAddZgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/Switch:1fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
АА*
T0
Є
_gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationUgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:
АА
ц
Ygradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_3ExitXgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:
АА
Ў
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMulMatMulegradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyXgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul/Enter*(
_output_shapes
:         А*
transpose_a( *
transpose_b(*
T0
╔
Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul/EnterEnter8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
АА*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
ў
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1MatMul_gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2egradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
АА*
transpose_a(
Ў
Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/ConstConst*
valueB :
         *O
_classE
CAloc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat*
dtype0*
_output_shapes
: 
╤
Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/Const*O
_classE
CAloc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat*

stack_name *
_output_shapes
:*
	elem_type0
▌
Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/EnterEnterZgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
с
`gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/Enter<biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat^gradients/Add_2*
T0*(
_output_shapes
:         А*
swap_memory( 
б
_gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2egradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub_2*(
_output_shapes
:         А*
	elem_type0
Є
egradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterZgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
Р
\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/group_depsNoOpS^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMulU^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1
╜
dgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityRgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul]^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul*(
_output_shapes
:         А
╗
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityTgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1]^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
АА
з
Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_accConst*
dtype0*
_output_shapes	
:А*
valueBА*    
ц
Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterXgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes	
:А*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
╥
Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeZgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1`gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:А: 
В
Ygradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchZgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_10*
T0*"
_output_shapes
:А:А
╔
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/AddAdd[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/Switch:1ggradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:А*
T0
я
`gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationVgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:А
у
Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitYgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:А
Р
ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Entervgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub_1*
source	gradients*V
_classL
JHloc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter*
_output_shapes

:: 
▓
tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter3biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray_1*
T0*V
_classL
JHloc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
▌
vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1Enter`biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
_output_shapes
: *S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context*
T0*V
_classL
JHloc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter*
parallel_iterations 
╪
jgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityvgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1o^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*V
_classL
JHloc:@biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter*
_output_shapes
: 
ц
pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3{gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2dgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/control_dependencyjgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
·
gradients/AddN_13AddN\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/tuple/control_dependencyfgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/tuple/control_dependency_1*
T0*]
_classS
QOloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/Select*
N*(
_output_shapes
:         А
е
Qgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConstConst^gradients/Sub_2*
value	B :*
dtype0*
_output_shapes
: 
д
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/RankConst^gradients/Sub_2*
value	B :*
dtype0*
_output_shapes
: 
б
Ogradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/modFloorModQgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConstPgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Rank*
_output_shapes
: *
T0
╬
Qgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeShape=biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3*
T0*
out_type0*
_output_shapes
:
┌
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeNShapeN]gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2Ygradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPopV2*
T0*
out_type0*
N* 
_output_shapes
::
ї
Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/ConstConst*
valueB :
         *P
_classF
DBloc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3*
dtype0*
_output_shapes
: 
╬
Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_accStackV2Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/Const*P
_classF
DBloc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3*

stack_name *
_output_shapes
:*
	elem_type0
┘
Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/EnterEnterXgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*I

frame_name;9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
▐
^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/Enter=biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3^gradients/Add_2*
T0*(
_output_shapes
:         А*
swap_memory( 
Э
]gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2cgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub_2*(
_output_shapes
:         А*
	elem_type0
ю
cgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterXgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_acc*
is_constant(*
parallel_iterations *
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0
О
Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConcatOffsetConcatOffsetOgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/modRgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeNTgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN:1*
N* 
_output_shapes
::
╢
Qgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/SliceSlicedgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/control_dependencyXgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConcatOffsetRgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN*0
_output_shapes
:                  *
T0*
Index0
╝
Sgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Slice_1Slicedgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/control_dependencyZgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ConcatOffset:1Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN:1*
T0*
Index0*0
_output_shapes
:                  
О
\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/group_depsNoOpR^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/SliceT^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Slice_1
╗
dgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/control_dependencyIdentityQgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Slice]^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*d
_classZ
XVloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Slice
┴
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/control_dependency_1IdentitySgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Slice_1]^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/group_deps*(
_output_shapes
:         А*
T0*f
_class\
ZXloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Slice_1
░
Wgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
АА*    *
dtype0* 
_output_shapes
:
АА
щ
Ygradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_1EnterWgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc*
parallel_iterations * 
_output_shapes
:
АА*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant( 
╘
Ygradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_2MergeYgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_1_gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/NextIteration*
N*"
_output_shapes
:
АА: *
T0
К
Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/SwitchSwitchYgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_10*,
_output_shapes
:
АА:
АА*
T0
╦
Ugradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/AddAddZgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/Switch:1fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
АА
Є
_gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationUgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:
АА
ц
Ygradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_3ExitXgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:
АА
Я
Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0*
_output_shapes
: 
х
\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterZgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context
╙
\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2Merge\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1bgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
√
[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitch\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_6*
T0*
_output_shapes
: : 
╤
Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/AddAdd]gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/Switch:1pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
ю
bgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationXgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
т
\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Exit[gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0*
_output_shapes
: 
к
Sgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_4_grad_1/NextIterationNextIterationgradients/AddN_13*
T0*(
_output_shapes
:         А
Р
ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Entervgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub_2*
source	gradients*V
_classL
JHloc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter*
_output_shapes

:: 
▓
tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter3biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray_1*
T0*V
_classL
JHloc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(*
_output_shapes
:*S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
▌
vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1Enter`biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
_output_shapes
: *S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context*
T0*V
_classL
JHloc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter*
parallel_iterations *
is_constant(
╪
jgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityvgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1o^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*V
_classL
JHloc:@biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter
ц
pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3{gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2dgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/control_dependencyjgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
·
gradients/AddN_14AddN\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/tuple/control_dependencyfgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/tuple/control_dependency_1*
N*(
_output_shapes
:         А*
T0*]
_classS
QOloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/Select
╖
Сgradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV33biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray_1\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
source	gradients*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray_1*
_output_shapes

:: 
Ў
Нgradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentity\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Т^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray_1
╜
Гgradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3Сgradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3>biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/rangeНgradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*
_output_shapes
:*
element_shape:
я
Аgradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOp]^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Д^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
√
Иgradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentityГgradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3Б^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*5
_output_shapes#
!:                  А*
T0*Щ
_classО
ЛИloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
Л
Кgradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1Identity\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Б^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*o
_classe
caloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes
: 
Я
Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0*
_output_shapes
: 
х
\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1EnterZgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *S

frame_nameECgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context
╙
\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2Merge\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_1bgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N*
_output_shapes
: : 
№
[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitch\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_10*
_output_shapes
: : *
T0
╤
Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/AddAdd]gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/Switch:1pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
ю
bgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIterationXgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/Add*
T0*
_output_shapes
: 
т
\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Exit[gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/Switch*
_output_shapes
: *
T0
к
Sgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_4_grad_1/NextIterationNextIterationgradients/AddN_14*(
_output_shapes
:         А*
T0
╖
Сgradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV33biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray_1\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
source	gradients*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray_1*
_output_shapes

:: 
Ў
Нgradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentity\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Т^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
_output_shapes
: *
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray_1
╜
Гgradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3Сgradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3>biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/rangeНgradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*
_output_shapes
:*
element_shape:
я
Аgradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOp]^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Д^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
√
Иgradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentityГgradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3Б^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*Щ
_classО
ЛИloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3*5
_output_shapes#
!:                  А*
T0
Л
Кgradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1Identity\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Б^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*o
_classe
caloc:@gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
_output_shapes
: 
╕
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/transpose_grad/InvertPermutationInvertPermutation,biLSTM_layers/bidirectional_rnn/fw/fw/concat*
_output_shapes
:*
T0
 
Hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/transpose_grad/transpose	TransposeИgradients/biLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyPgradients/biLSTM_layers/bidirectional_rnn/fw/fw/transpose_grad/InvertPermutation*5
_output_shapes#
!:                  А*
Tperm0*
T0
╕
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/transpose_grad/InvertPermutationInvertPermutation,biLSTM_layers/bidirectional_rnn/bw/bw/concat*
T0*
_output_shapes
:
 
Hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/transpose_grad/transpose	TransposeИgradients/biLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyPgradients/biLSTM_layers/bidirectional_rnn/bw/bw/transpose_grad/InvertPermutation*
T0*5
_output_shapes#
!:                  А*
Tperm0
Я
Qgradients/biLSTM_layers/bidirectional_rnn/bw/ReverseSequence_grad/ReverseSequenceReverseSequenceHgradients/biLSTM_layers/bidirectional_rnn/bw/bw/transpose_grad/transposeSum*5
_output_shapes#
!:                  А*

Tlen0*
	batch_dim *
T0*
seq_dim
▄
gradients/AddN_15AddNHgradients/biLSTM_layers/bidirectional_rnn/fw/fw/transpose_grad/transposeQgradients/biLSTM_layers/bidirectional_rnn/bw/ReverseSequence_grad/ReverseSequence*[
_classQ
OMloc:@gradients/biLSTM_layers/bidirectional_rnn/fw/fw/transpose_grad/transpose*
N*5
_output_shapes#
!:                  А*
T0
├
5gradients/embedding_layer/embedding_lookup_grad/ShapeConst*
dtype0	*
_output_shapes
:*%
valueB	"Д      А       *3
_class)
'%loc:@embedding_layer/embedding_matrix
▀
7gradients/embedding_layer/embedding_lookup_grad/ToInt32Cast5gradients/embedding_layer/embedding_lookup_grad/Shape*

SrcT0	*3
_class)
'%loc:@embedding_layer/embedding_matrix*
_output_shapes
:*

DstT0
u
4gradients/embedding_layer/embedding_lookup_grad/SizeSizeinputs*
_output_shapes
: *
T0*
out_type0
А
>gradients/embedding_layer/embedding_lookup_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
я
:gradients/embedding_layer/embedding_lookup_grad/ExpandDims
ExpandDims4gradients/embedding_layer/embedding_lookup_grad/Size>gradients/embedding_layer/embedding_lookup_grad/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
Н
Cgradients/embedding_layer/embedding_lookup_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
П
Egradients/embedding_layer/embedding_lookup_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
П
Egradients/embedding_layer/embedding_lookup_grad/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
я
=gradients/embedding_layer/embedding_lookup_grad/strided_sliceStridedSlice7gradients/embedding_layer/embedding_lookup_grad/ToInt32Cgradients/embedding_layer/embedding_lookup_grad/strided_slice/stackEgradients/embedding_layer/embedding_lookup_grad/strided_slice/stack_1Egradients/embedding_layer/embedding_lookup_grad/strided_slice/stack_2*
_output_shapes
:*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask
}
;gradients/embedding_layer/embedding_lookup_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
┤
6gradients/embedding_layer/embedding_lookup_grad/concatConcatV2:gradients/embedding_layer/embedding_lookup_grad/ExpandDims=gradients/embedding_layer/embedding_lookup_grad/strided_slice;gradients/embedding_layer/embedding_lookup_grad/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
╬
7gradients/embedding_layer/embedding_lookup_grad/ReshapeReshapegradients/AddN_156gradients/embedding_layer/embedding_lookup_grad/concat*
Tshape0*(
_output_shapes
:         А*
T0
─
9gradients/embedding_layer/embedding_lookup_grad/Reshape_1Reshapeinputs:gradients/embedding_layer/embedding_lookup_grad/ExpandDims*
T0*
Tshape0*#
_output_shapes
:         
Ж
beta1_power/initial_valueConst*
valueB
 *fff?*&
_class
loc:@biLSTM_layers/W_out*
dtype0*
_output_shapes
: 
Ч
beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *&
_class
loc:@biLSTM_layers/W_out*
	container *
shape: 
╢
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*&
_class
loc:@biLSTM_layers/W_out*
validate_shape(*
_output_shapes
: 
r
beta1_power/readIdentitybeta1_power*
T0*&
_class
loc:@biLSTM_layers/W_out*
_output_shapes
: 
Ж
beta2_power/initial_valueConst*
valueB
 *w╛?*&
_class
loc:@biLSTM_layers/W_out*
dtype0*
_output_shapes
: 
Ч
beta2_power
VariableV2*
shared_name *&
_class
loc:@biLSTM_layers/W_out*
	container *
shape: *
dtype0*
_output_shapes
: 
╢
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*&
_class
loc:@biLSTM_layers/W_out*
validate_shape(*
_output_shapes
: 
r
beta2_power/readIdentitybeta2_power*
T0*&
_class
loc:@biLSTM_layers/W_out*
_output_shapes
: 
═
Gembedding_layer/embedding_matrix/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"Д  А   *3
_class)
'%loc:@embedding_layer/embedding_matrix
╖
=embedding_layer/embedding_matrix/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *3
_class)
'%loc:@embedding_layer/embedding_matrix
╣
7embedding_layer/embedding_matrix/Adam/Initializer/zerosFillGembedding_layer/embedding_matrix/Adam/Initializer/zeros/shape_as_tensor=embedding_layer/embedding_matrix/Adam/Initializer/zeros/Const* 
_output_shapes
:
Д/А*
T0*

index_type0*3
_class)
'%loc:@embedding_layer/embedding_matrix
╥
%embedding_layer/embedding_matrix/Adam
VariableV2*
	container *
shape:
Д/А*
dtype0* 
_output_shapes
:
Д/А*
shared_name *3
_class)
'%loc:@embedding_layer/embedding_matrix
Я
,embedding_layer/embedding_matrix/Adam/AssignAssign%embedding_layer/embedding_matrix/Adam7embedding_layer/embedding_matrix/Adam/Initializer/zeros*
use_locking(*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix*
validate_shape(* 
_output_shapes
:
Д/А
╜
*embedding_layer/embedding_matrix/Adam/readIdentity%embedding_layer/embedding_matrix/Adam* 
_output_shapes
:
Д/А*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix
╧
Iembedding_layer/embedding_matrix/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"Д  А   *3
_class)
'%loc:@embedding_layer/embedding_matrix*
dtype0*
_output_shapes
:
╣
?embedding_layer/embedding_matrix/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *3
_class)
'%loc:@embedding_layer/embedding_matrix*
dtype0*
_output_shapes
: 
┐
9embedding_layer/embedding_matrix/Adam_1/Initializer/zerosFillIembedding_layer/embedding_matrix/Adam_1/Initializer/zeros/shape_as_tensor?embedding_layer/embedding_matrix/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
Д/А*
T0*

index_type0*3
_class)
'%loc:@embedding_layer/embedding_matrix
╘
'embedding_layer/embedding_matrix/Adam_1
VariableV2*
shape:
Д/А*
dtype0* 
_output_shapes
:
Д/А*
shared_name *3
_class)
'%loc:@embedding_layer/embedding_matrix*
	container 
е
.embedding_layer/embedding_matrix/Adam_1/AssignAssign'embedding_layer/embedding_matrix/Adam_19embedding_layer/embedding_matrix/Adam_1/Initializer/zeros*3
_class)
'%loc:@embedding_layer/embedding_matrix*
validate_shape(* 
_output_shapes
:
Д/А*
use_locking(*
T0
┴
,embedding_layer/embedding_matrix/Adam_1/readIdentity'embedding_layer/embedding_matrix/Adam_1*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix* 
_output_shapes
:
Д/А
є
ZbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel*
dtype0*
_output_shapes
:
▌
PbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel*
dtype0*
_output_shapes
: 
Е
JbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zerosFillZbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorPbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel* 
_output_shapes
:
АА
°
8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam
VariableV2*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel
ы
?biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam/AssignAssign8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/AdamJbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА
Ў
=biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam/readIdentity8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel* 
_output_shapes
:
АА
ї
\biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"      *F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel
▀
RbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel*
dtype0*
_output_shapes
: 
Л
LbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zerosFill\biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorRbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
АА*
T0*

index_type0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel
·
:biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel*
	container *
shape:
АА
ё
AbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/AssignAssign:biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1LbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА
·
?biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/readIdentity:biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel* 
_output_shapes
:
АА
щ
XbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam/Initializer/zeros/shape_as_tensorConst*
valueB:А*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
dtype0*
_output_shapes
:
┘
NbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam/Initializer/zeros/ConstConst*
valueB
 *    *D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
dtype0*
_output_shapes
: 
°
HbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam/Initializer/zerosFillXbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam/Initializer/zeros/shape_as_tensorNbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam/Initializer/zeros/Const*
T0*

index_type0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
_output_shapes	
:А
ъ
6biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
	container *
shape:А
▐
=biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam/AssignAssign6biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/AdamHbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam/Initializer/zeros*
_output_shapes	
:А*
use_locking(*
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
validate_shape(
ы
;biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam/readIdentity6biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam*
_output_shapes	
:А*
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias
ы
ZbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB:А*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
dtype0*
_output_shapes
:
█
PbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
dtype0*
_output_shapes
: 
■
JbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Initializer/zerosFillZbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Initializer/zeros/shape_as_tensorPbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Initializer/zeros/Const*
T0*

index_type0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
_output_shapes	
:А
ь
8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1
VariableV2*
shared_name *D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
ф
?biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/AssignAssign8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1JbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
я
=biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/readIdentity8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1*
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
_output_shapes	
:А
є
ZbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel*
dtype0*
_output_shapes
:
▌
PbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel
Е
JbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zerosFillZbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorPbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel* 
_output_shapes
:
АА
°
8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam
VariableV2*
shared_name *F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА
ы
?biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam/AssignAssign8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/AdamJbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА
Ў
=biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam/readIdentity8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel* 
_output_shapes
:
АА
ї
\biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel*
dtype0*
_output_shapes
:
▀
RbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel*
dtype0
Л
LbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zerosFill\biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorRbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel* 
_output_shapes
:
АА
·
:biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1
VariableV2*
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel*
	container 
ё
AbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/AssignAssign:biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1LbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА
·
?biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/readIdentity:biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel* 
_output_shapes
:
АА
щ
XbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam/Initializer/zeros/shape_as_tensorConst*
valueB:А*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
dtype0*
_output_shapes
:
┘
NbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam/Initializer/zeros/ConstConst*
valueB
 *    *D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
dtype0*
_output_shapes
: 
°
HbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam/Initializer/zerosFillXbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam/Initializer/zeros/shape_as_tensorNbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam/Initializer/zeros/Const*
T0*

index_type0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
_output_shapes	
:А
ъ
6biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
	container 
▐
=biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam/AssignAssign6biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/AdamHbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ы
;biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam/readIdentity6biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam*
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
_output_shapes	
:А
ы
ZbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:А*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
dtype0
█
PbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias
■
JbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Initializer/zerosFillZbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Initializer/zeros/shape_as_tensorPbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Initializer/zeros/Const*
T0*

index_type0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
_output_shapes	
:А
ь
8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1
VariableV2*
shared_name *D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
ф
?biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/AssignAssign8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1JbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Initializer/zeros*
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
я
=biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/readIdentity8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1*
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
_output_shapes	
:А
│
:biLSTM_layers/W_out/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *&
_class
loc:@biLSTM_layers/W_out*
dtype0*
_output_shapes
:
Э
0biLSTM_layers/W_out/Adam/Initializer/zeros/ConstConst*
valueB
 *    *&
_class
loc:@biLSTM_layers/W_out*
dtype0*
_output_shapes
: 
Д
*biLSTM_layers/W_out/Adam/Initializer/zerosFill:biLSTM_layers/W_out/Adam/Initializer/zeros/shape_as_tensor0biLSTM_layers/W_out/Adam/Initializer/zeros/Const*

index_type0*&
_class
loc:@biLSTM_layers/W_out*
_output_shapes
:	А*
T0
╢
biLSTM_layers/W_out/Adam
VariableV2*
shared_name *&
_class
loc:@biLSTM_layers/W_out*
	container *
shape:	А*
dtype0*
_output_shapes
:	А
ъ
biLSTM_layers/W_out/Adam/AssignAssignbiLSTM_layers/W_out/Adam*biLSTM_layers/W_out/Adam/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@biLSTM_layers/W_out*
validate_shape(*
_output_shapes
:	А
Х
biLSTM_layers/W_out/Adam/readIdentitybiLSTM_layers/W_out/Adam*
_output_shapes
:	А*
T0*&
_class
loc:@biLSTM_layers/W_out
╡
<biLSTM_layers/W_out/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"      *&
_class
loc:@biLSTM_layers/W_out
Я
2biLSTM_layers/W_out/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *&
_class
loc:@biLSTM_layers/W_out
К
,biLSTM_layers/W_out/Adam_1/Initializer/zerosFill<biLSTM_layers/W_out/Adam_1/Initializer/zeros/shape_as_tensor2biLSTM_layers/W_out/Adam_1/Initializer/zeros/Const*
T0*

index_type0*&
_class
loc:@biLSTM_layers/W_out*
_output_shapes
:	А
╕
biLSTM_layers/W_out/Adam_1
VariableV2*
dtype0*
_output_shapes
:	А*
shared_name *&
_class
loc:@biLSTM_layers/W_out*
	container *
shape:	А
Ё
!biLSTM_layers/W_out/Adam_1/AssignAssignbiLSTM_layers/W_out/Adam_1,biLSTM_layers/W_out/Adam_1/Initializer/zeros*
use_locking(*
T0*&
_class
loc:@biLSTM_layers/W_out*
validate_shape(*
_output_shapes
:	А
Щ
biLSTM_layers/W_out/Adam_1/readIdentitybiLSTM_layers/W_out/Adam_1*
T0*&
_class
loc:@biLSTM_layers/W_out*
_output_shapes
:	А
д
6biLSTM_layers/b/Adam/Initializer/zeros/shape_as_tensorConst*
valueB:*"
_class
loc:@biLSTM_layers/b*
dtype0*
_output_shapes
:
Х
,biLSTM_layers/b/Adam/Initializer/zeros/ConstConst*
valueB
 *    *"
_class
loc:@biLSTM_layers/b*
dtype0*
_output_shapes
: 
я
&biLSTM_layers/b/Adam/Initializer/zerosFill6biLSTM_layers/b/Adam/Initializer/zeros/shape_as_tensor,biLSTM_layers/b/Adam/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@biLSTM_layers/b*
_output_shapes
:
д
biLSTM_layers/b/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@biLSTM_layers/b*
	container *
shape:
╒
biLSTM_layers/b/Adam/AssignAssignbiLSTM_layers/b/Adam&biLSTM_layers/b/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@biLSTM_layers/b*
validate_shape(*
_output_shapes
:
Д
biLSTM_layers/b/Adam/readIdentitybiLSTM_layers/b/Adam*
_output_shapes
:*
T0*"
_class
loc:@biLSTM_layers/b
ж
8biLSTM_layers/b/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB:*"
_class
loc:@biLSTM_layers/b*
dtype0*
_output_shapes
:
Ч
.biLSTM_layers/b/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *"
_class
loc:@biLSTM_layers/b*
dtype0*
_output_shapes
: 
ї
(biLSTM_layers/b/Adam_1/Initializer/zerosFill8biLSTM_layers/b/Adam_1/Initializer/zeros/shape_as_tensor.biLSTM_layers/b/Adam_1/Initializer/zeros/Const*
T0*

index_type0*"
_class
loc:@biLSTM_layers/b*
_output_shapes
:
ж
biLSTM_layers/b/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@biLSTM_layers/b*
	container 
█
biLSTM_layers/b/Adam_1/AssignAssignbiLSTM_layers/b/Adam_1(biLSTM_layers/b/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@biLSTM_layers/b*
validate_shape(*
_output_shapes
:
И
biLSTM_layers/b/Adam_1/readIdentitybiLSTM_layers/b/Adam_1*
T0*"
_class
loc:@biLSTM_layers/b*
_output_shapes
:
г
2transitions/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"      *
_class
loc:@transitions*
dtype0
Н
(transitions/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@transitions*
dtype0*
_output_shapes
: 
у
"transitions/Adam/Initializer/zerosFill2transitions/Adam/Initializer/zeros/shape_as_tensor(transitions/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@transitions*
_output_shapes

:
д
transitions/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *
_class
loc:@transitions*
	container 
╔
transitions/Adam/AssignAssigntransitions/Adam"transitions/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@transitions*
validate_shape(*
_output_shapes

:
|
transitions/Adam/readIdentitytransitions/Adam*
_class
loc:@transitions*
_output_shapes

:*
T0
е
4transitions/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *
_class
loc:@transitions*
dtype0*
_output_shapes
:
П
*transitions/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@transitions*
dtype0*
_output_shapes
: 
щ
$transitions/Adam_1/Initializer/zerosFill4transitions/Adam_1/Initializer/zeros/shape_as_tensor*transitions/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@transitions*
_output_shapes

:
ж
transitions/Adam_1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *
_class
loc:@transitions*
	container 
╧
transitions/Adam_1/AssignAssigntransitions/Adam_1$transitions/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@transitions*
validate_shape(*
_output_shapes

:
А
transitions/Adam_1/readIdentitytransitions/Adam_1*
T0*
_class
loc:@transitions*
_output_shapes

:
W
Adam/learning_rateConst*
_output_shapes
: *
valueB
 *
╫#<*
dtype0
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w╛?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
∙
3Adam/update_embedding_layer/embedding_matrix/UniqueUnique9gradients/embedding_layer/embedding_lookup_grad/Reshape_1*
T0*
out_idx0*3
_class)
'%loc:@embedding_layer/embedding_matrix*2
_output_shapes 
:         :         
┌
2Adam/update_embedding_layer/embedding_matrix/ShapeShape3Adam/update_embedding_layer/embedding_matrix/Unique*
T0*
out_type0*3
_class)
'%loc:@embedding_layer/embedding_matrix*
_output_shapes
:
┐
@Adam/update_embedding_layer/embedding_matrix/strided_slice/stackConst*
valueB: *3
_class)
'%loc:@embedding_layer/embedding_matrix*
dtype0*
_output_shapes
:
┴
BAdam/update_embedding_layer/embedding_matrix/strided_slice/stack_1Const*
valueB:*3
_class)
'%loc:@embedding_layer/embedding_matrix*
dtype0*
_output_shapes
:
┴
BAdam/update_embedding_layer/embedding_matrix/strided_slice/stack_2Const*
valueB:*3
_class)
'%loc:@embedding_layer/embedding_matrix*
dtype0*
_output_shapes
:
П
:Adam/update_embedding_layer/embedding_matrix/strided_sliceStridedSlice2Adam/update_embedding_layer/embedding_matrix/Shape@Adam/update_embedding_layer/embedding_matrix/strided_slice/stackBAdam/update_embedding_layer/embedding_matrix/strided_slice/stack_1BAdam/update_embedding_layer/embedding_matrix/strided_slice/stack_2*
Index0*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Н
?Adam/update_embedding_layer/embedding_matrix/UnsortedSegmentSumUnsortedSegmentSum7gradients/embedding_layer/embedding_lookup_grad/Reshape5Adam/update_embedding_layer/embedding_matrix/Unique:1:Adam/update_embedding_layer/embedding_matrix/strided_slice*
Tnumsegments0*
Tindices0*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix*(
_output_shapes
:         А
м
2Adam/update_embedding_layer/embedding_matrix/sub/xConst*
valueB
 *  А?*3
_class)
'%loc:@embedding_layer/embedding_matrix*
dtype0*
_output_shapes
: 
╙
0Adam/update_embedding_layer/embedding_matrix/subSub2Adam/update_embedding_layer/embedding_matrix/sub/xbeta2_power/read*3
_class)
'%loc:@embedding_layer/embedding_matrix*
_output_shapes
: *
T0
┴
1Adam/update_embedding_layer/embedding_matrix/SqrtSqrt0Adam/update_embedding_layer/embedding_matrix/sub*
_output_shapes
: *
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix
╘
0Adam/update_embedding_layer/embedding_matrix/mulMulAdam/learning_rate1Adam/update_embedding_layer/embedding_matrix/Sqrt*3
_class)
'%loc:@embedding_layer/embedding_matrix*
_output_shapes
: *
T0
о
4Adam/update_embedding_layer/embedding_matrix/sub_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?*3
_class)
'%loc:@embedding_layer/embedding_matrix
╫
2Adam/update_embedding_layer/embedding_matrix/sub_1Sub4Adam/update_embedding_layer/embedding_matrix/sub_1/xbeta1_power/read*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix*
_output_shapes
: 
√
4Adam/update_embedding_layer/embedding_matrix/truedivRealDiv0Adam/update_embedding_layer/embedding_matrix/mul2Adam/update_embedding_layer/embedding_matrix/sub_1*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix*
_output_shapes
: 
о
4Adam/update_embedding_layer/embedding_matrix/sub_2/xConst*
valueB
 *  А?*3
_class)
'%loc:@embedding_layer/embedding_matrix*
dtype0*
_output_shapes
: 
╤
2Adam/update_embedding_layer/embedding_matrix/sub_2Sub4Adam/update_embedding_layer/embedding_matrix/sub_2/x
Adam/beta1*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix*
_output_shapes
: 
Ц
2Adam/update_embedding_layer/embedding_matrix/mul_1Mul?Adam/update_embedding_layer/embedding_matrix/UnsortedSegmentSum2Adam/update_embedding_layer/embedding_matrix/sub_2*3
_class)
'%loc:@embedding_layer/embedding_matrix*(
_output_shapes
:         А*
T0
╤
2Adam/update_embedding_layer/embedding_matrix/mul_2Mul*embedding_layer/embedding_matrix/Adam/read
Adam/beta1*3
_class)
'%loc:@embedding_layer/embedding_matrix* 
_output_shapes
:
Д/А*
T0
б
3Adam/update_embedding_layer/embedding_matrix/AssignAssign%embedding_layer/embedding_matrix/Adam2Adam/update_embedding_layer/embedding_matrix/mul_2*
validate_shape(* 
_output_shapes
:
Д/А*
use_locking( *
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix
О
7Adam/update_embedding_layer/embedding_matrix/ScatterAdd
ScatterAdd%embedding_layer/embedding_matrix/Adam3Adam/update_embedding_layer/embedding_matrix/Unique2Adam/update_embedding_layer/embedding_matrix/mul_14^Adam/update_embedding_layer/embedding_matrix/Assign*
use_locking( *
Tindices0*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix* 
_output_shapes
:
Д/А
г
2Adam/update_embedding_layer/embedding_matrix/mul_3Mul?Adam/update_embedding_layer/embedding_matrix/UnsortedSegmentSum?Adam/update_embedding_layer/embedding_matrix/UnsortedSegmentSum*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix*(
_output_shapes
:         А
о
4Adam/update_embedding_layer/embedding_matrix/sub_3/xConst*
valueB
 *  А?*3
_class)
'%loc:@embedding_layer/embedding_matrix*
dtype0*
_output_shapes
: 
╤
2Adam/update_embedding_layer/embedding_matrix/sub_3Sub4Adam/update_embedding_layer/embedding_matrix/sub_3/x
Adam/beta2*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix*
_output_shapes
: 
Й
2Adam/update_embedding_layer/embedding_matrix/mul_4Mul2Adam/update_embedding_layer/embedding_matrix/mul_32Adam/update_embedding_layer/embedding_matrix/sub_3*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix*(
_output_shapes
:         А
╙
2Adam/update_embedding_layer/embedding_matrix/mul_5Mul,embedding_layer/embedding_matrix/Adam_1/read
Adam/beta2*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix* 
_output_shapes
:
Д/А
е
5Adam/update_embedding_layer/embedding_matrix/Assign_1Assign'embedding_layer/embedding_matrix/Adam_12Adam/update_embedding_layer/embedding_matrix/mul_5*3
_class)
'%loc:@embedding_layer/embedding_matrix*
validate_shape(* 
_output_shapes
:
Д/А*
use_locking( *
T0
Ф
9Adam/update_embedding_layer/embedding_matrix/ScatterAdd_1
ScatterAdd'embedding_layer/embedding_matrix/Adam_13Adam/update_embedding_layer/embedding_matrix/Unique2Adam/update_embedding_layer/embedding_matrix/mul_46^Adam/update_embedding_layer/embedding_matrix/Assign_1* 
_output_shapes
:
Д/А*
use_locking( *
Tindices0*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix
╓
3Adam/update_embedding_layer/embedding_matrix/Sqrt_1Sqrt9Adam/update_embedding_layer/embedding_matrix/ScatterAdd_1*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix* 
_output_shapes
:
Д/А
И
2Adam/update_embedding_layer/embedding_matrix/mul_6Mul4Adam/update_embedding_layer/embedding_matrix/truediv7Adam/update_embedding_layer/embedding_matrix/ScatterAdd*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix* 
_output_shapes
:
Д/А
┌
0Adam/update_embedding_layer/embedding_matrix/addAdd3Adam/update_embedding_layer/embedding_matrix/Sqrt_1Adam/epsilon*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix* 
_output_shapes
:
Д/А
З
6Adam/update_embedding_layer/embedding_matrix/truediv_1RealDiv2Adam/update_embedding_layer/embedding_matrix/mul_60Adam/update_embedding_layer/embedding_matrix/add* 
_output_shapes
:
Д/А*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix
Р
6Adam/update_embedding_layer/embedding_matrix/AssignSub	AssignSub embedding_layer/embedding_matrix6Adam/update_embedding_layer/embedding_matrix/truediv_1*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix* 
_output_shapes
:
Д/А*
use_locking( 
г
7Adam/update_embedding_layer/embedding_matrix/group_depsNoOp7^Adam/update_embedding_layer/embedding_matrix/AssignSub8^Adam/update_embedding_layer/embedding_matrix/ScatterAdd:^Adam/update_embedding_layer/embedding_matrix/ScatterAdd_1*3
_class)
'%loc:@embedding_layer/embedding_matrix
╘
IAdam/update_biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/ApplyAdam	ApplyAdam3biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam:biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonYgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter_grad/b_acc_3*
use_locking( *
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel*
use_nesterov( * 
_output_shapes
:
АА
╞
GAdam/update_biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/ApplyAdam	ApplyAdam1biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias6biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonZgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
use_locking( *
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
use_nesterov( *
_output_shapes	
:А
╘
IAdam/update_biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/ApplyAdam	ApplyAdam3biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam:biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonYgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel*
use_nesterov( * 
_output_shapes
:
АА*
use_locking( 
╞
GAdam/update_biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/ApplyAdam	ApplyAdam1biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias6biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonZgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
use_locking( *
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
use_nesterov( *
_output_shapes	
:А
Ш
)Adam/update_biLSTM_layers/W_out/ApplyAdam	ApplyAdambiLSTM_layers/W_outbiLSTM_layers/W_out/AdambiLSTM_layers/W_out/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/biLSTM_layers/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*&
_class
loc:@biLSTM_layers/W_out*
use_nesterov( *
_output_shapes
:	А
№
%Adam/update_biLSTM_layers/b/ApplyAdam	ApplyAdambiLSTM_layers/bbiLSTM_layers/b/AdambiLSTM_layers/b/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/biLSTM_layers/add_grad/tuple/control_dependency_1*
T0*"
_class
loc:@biLSTM_layers/b*
use_nesterov( *
_output_shapes
:*
use_locking( 
┴
!Adam/update_transitions/ApplyAdam	ApplyAdamtransitionstransitions/Adamtransitions/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_5*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0*
_class
loc:@transitions
╘
Adam/mulMulbeta1_power/read
Adam/beta18^Adam/update_embedding_layer/embedding_matrix/group_depsJ^Adam/update_biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/ApplyAdamH^Adam/update_biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/ApplyAdamJ^Adam/update_biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/ApplyAdamH^Adam/update_biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/ApplyAdam*^Adam/update_biLSTM_layers/W_out/ApplyAdam&^Adam/update_biLSTM_layers/b/ApplyAdam"^Adam/update_transitions/ApplyAdam*&
_class
loc:@biLSTM_layers/W_out*
_output_shapes
: *
T0
Ю
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*&
_class
loc:@biLSTM_layers/W_out*
validate_shape(*
_output_shapes
: 
╓

Adam/mul_1Mulbeta2_power/read
Adam/beta28^Adam/update_embedding_layer/embedding_matrix/group_depsJ^Adam/update_biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/ApplyAdamH^Adam/update_biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/ApplyAdamJ^Adam/update_biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/ApplyAdamH^Adam/update_biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/ApplyAdam*^Adam/update_biLSTM_layers/W_out/ApplyAdam&^Adam/update_biLSTM_layers/b/ApplyAdam"^Adam/update_transitions/ApplyAdam*&
_class
loc:@biLSTM_layers/W_out*
_output_shapes
: *
T0
в
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*&
_class
loc:@biLSTM_layers/W_out*
validate_shape(*
_output_shapes
: 
И
AdamNoOp8^Adam/update_embedding_layer/embedding_matrix/group_depsJ^Adam/update_biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/ApplyAdamH^Adam/update_biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/ApplyAdamJ^Adam/update_biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/ApplyAdamH^Adam/update_biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/ApplyAdam*^Adam/update_biLSTM_layers/W_out/ApplyAdam&^Adam/update_biLSTM_layers/b/ApplyAdam"^Adam/update_transitions/ApplyAdam^Adam/Assign^Adam/Assign_1
^
Shape_2ShapebiLSTM_layers/Reshape_1*
T0*
out_type0*
_output_shapes
:
_
strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Г
strided_slice_2StridedSliceShape_2strided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
K
	Equal_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
M
Equal_2Equalstrided_slice_2	Equal_2/y*
_output_shapes
: *
T0
L
cond_2/SwitchSwitchEqual_2Equal_2*
_output_shapes
: : *
T0

M
cond_2/switch_tIdentitycond_2/Switch:1*
T0
*
_output_shapes
: 
K
cond_2/switch_fIdentitycond_2/Switch*
T0
*
_output_shapes
: 
D
cond_2/pred_idIdentityEqual_2*
T0
*
_output_shapes
: 
{
cond_2/SqueezeSqueezecond_2/Squeeze/Switch:1*
T0*'
_output_shapes
:         *
squeeze_dims

╙
cond_2/Squeeze/SwitchSwitchbiLSTM_layers/Reshape_1cond_2/pred_id*T
_output_shapesB
@:                  :                  *
T0**
_class 
loc:@biLSTM_layers/Reshape_1
k
cond_2/ArgMax/dimensionConst^cond_2/switch_t*
value	B :*
dtype0*
_output_shapes
: 
Н
cond_2/ArgMaxArgMaxcond_2/Squeezecond_2/ArgMax/dimension*
output_type0	*#
_output_shapes
:         *

Tidx0*
T0
i
cond_2/ExpandDims/dimConst^cond_2/switch_t*
dtype0*
_output_shapes
: *
value	B :
Г
cond_2/ExpandDims
ExpandDimscond_2/ArgMaxcond_2/ExpandDims/dim*

Tdim0*
T0	*'
_output_shapes
:         
p
cond_2/Max/reduction_indicesConst^cond_2/switch_t*
value	B :*
dtype0*
_output_shapes
: 
К

cond_2/MaxMaxcond_2/Squeezecond_2/Max/reduction_indices*#
_output_shapes
:         *

Tidx0*
	keep_dims( *
T0
g
cond_2/CastCastcond_2/ExpandDims*

SrcT0	*'
_output_shapes
:         *

DstT0
k
cond_2/ExpandDims_1/dimConst^cond_2/switch_f*
value	B : *
dtype0*
_output_shapes
: 
П
cond_2/ExpandDims_1
ExpandDimscond_2/ExpandDims_1/Switchcond_2/ExpandDims_1/dim*

Tdim0*
T0*"
_output_shapes
:
Щ
cond_2/ExpandDims_1/SwitchSwitchtransitions/readcond_2/pred_id*
_class
loc:@transitions*(
_output_shapes
::*
T0
y
cond_2/Slice/beginConst^cond_2/switch_f*!
valueB"            *
dtype0*
_output_shapes
:
x
cond_2/Slice/sizeConst^cond_2/switch_f*
_output_shapes
:*!
valueB"           *
dtype0
Ф
cond_2/SliceSlicecond_2/Slice/Switchcond_2/Slice/begincond_2/Slice/size*
T0*
Index0*+
_output_shapes
:         
╤
cond_2/Slice/SwitchSwitchbiLSTM_layers/Reshape_1cond_2/pred_id*
T0**
_class 
loc:@biLSTM_layers/Reshape_1*T
_output_shapesB
@:                  :                  
r
cond_2/Squeeze_1Squeezecond_2/Slice*
T0*'
_output_shapes
:         *
squeeze_dims

{
cond_2/Slice_1/beginConst^cond_2/switch_f*!
valueB"           *
dtype0*
_output_shapes
:
z
cond_2/Slice_1/sizeConst^cond_2/switch_f*!
valueB"            *
dtype0*
_output_shapes
:
г
cond_2/Slice_1Slicecond_2/Slice/Switchcond_2/Slice_1/begincond_2/Slice_1/size*
T0*
Index0*4
_output_shapes"
 :                  
`
cond_2/sub/yConst^cond_2/switch_f*
value	B :*
dtype0*
_output_shapes
: 
`

cond_2/subSubcond_2/sub/Switchcond_2/sub/y*
T0*#
_output_shapes
:         
Е
cond_2/sub/SwitchSwitchSumcond_2/pred_id*2
_output_shapes 
:         :         *
T0*
_class

loc:@Sum
c
cond_2/rnn/RankConst^cond_2/switch_f*
value	B :*
dtype0*
_output_shapes
: 
j
cond_2/rnn/range/startConst^cond_2/switch_f*
_output_shapes
: *
value	B :*
dtype0
j
cond_2/rnn/range/deltaConst^cond_2/switch_f*
value	B :*
dtype0*
_output_shapes
: 
В
cond_2/rnn/rangeRangecond_2/rnn/range/startcond_2/rnn/Rankcond_2/rnn/range/delta*

Tidx0*
_output_shapes
:
}
cond_2/rnn/concat/values_0Const^cond_2/switch_f*
valueB"       *
dtype0*
_output_shapes
:
j
cond_2/rnn/concat/axisConst^cond_2/switch_f*
value	B : *
dtype0*
_output_shapes
: 
Э
cond_2/rnn/concatConcatV2cond_2/rnn/concat/values_0cond_2/rnn/rangecond_2/rnn/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
Р
cond_2/rnn/transpose	Transposecond_2/Slice_1cond_2/rnn/concat*
T0*4
_output_shapes"
 :                  *
Tperm0
`
cond_2/rnn/sequence_lengthIdentity
cond_2/sub*
T0*#
_output_shapes
:         
d
cond_2/rnn/ShapeShapecond_2/rnn/transpose*
T0*
out_type0*
_output_shapes
:
z
cond_2/rnn/strided_slice/stackConst^cond_2/switch_f*
valueB:*
dtype0*
_output_shapes
:
|
 cond_2/rnn/strided_slice/stack_1Const^cond_2/switch_f*
valueB:*
dtype0*
_output_shapes
:
|
 cond_2/rnn/strided_slice/stack_2Const^cond_2/switch_f*
valueB:*
dtype0*
_output_shapes
:
░
cond_2/rnn/strided_sliceStridedSlicecond_2/rnn/Shapecond_2/rnn/strided_slice/stack cond_2/rnn/strided_slice/stack_1 cond_2/rnn/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
l
cond_2/rnn/Shape_1Shapecond_2/rnn/sequence_length*
T0*
out_type0*
_output_shapes
:
l
cond_2/rnn/stackPackcond_2/rnn/strided_slice*
N*
_output_shapes
:*
T0*

axis 
d
cond_2/rnn/EqualEqualcond_2/rnn/Shape_1cond_2/rnn/stack*
_output_shapes
:*
T0
l
cond_2/rnn/ConstConst^cond_2/switch_f*
valueB: *
dtype0*
_output_shapes
:
n
cond_2/rnn/AllAllcond_2/rnn/Equalcond_2/rnn/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
д
cond_2/rnn/Assert/ConstConst^cond_2/switch_f*
_output_shapes
: *K
valueBB@ B:Expected shape for Tensor cond_2/rnn/sequence_length:0 is *
dtype0
|
cond_2/rnn/Assert/Const_1Const^cond_2/switch_f*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
м
cond_2/rnn/Assert/Assert/data_0Const^cond_2/switch_f*
dtype0*
_output_shapes
: *K
valueBB@ B:Expected shape for Tensor cond_2/rnn/sequence_length:0 is 
В
cond_2/rnn/Assert/Assert/data_2Const^cond_2/switch_f*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
║
cond_2/rnn/Assert/AssertAssertcond_2/rnn/Allcond_2/rnn/Assert/Assert/data_0cond_2/rnn/stackcond_2/rnn/Assert/Assert/data_2cond_2/rnn/Shape_1*
T
2*
	summarize
З
cond_2/rnn/CheckSeqLenIdentitycond_2/rnn/sequence_length^cond_2/rnn/Assert/Assert*
T0*#
_output_shapes
:         
f
cond_2/rnn/Shape_2Shapecond_2/rnn/transpose*
T0*
out_type0*
_output_shapes
:
|
 cond_2/rnn/strided_slice_1/stackConst^cond_2/switch_f*
valueB: *
dtype0*
_output_shapes
:
~
"cond_2/rnn/strided_slice_1/stack_1Const^cond_2/switch_f*
dtype0*
_output_shapes
:*
valueB:
~
"cond_2/rnn/strided_slice_1/stack_2Const^cond_2/switch_f*
valueB:*
dtype0*
_output_shapes
:
║
cond_2/rnn/strided_slice_1StridedSlicecond_2/rnn/Shape_2 cond_2/rnn/strided_slice_1/stack"cond_2/rnn/strided_slice_1/stack_1"cond_2/rnn/strided_slice_1/stack_2*
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
f
cond_2/rnn/Shape_3Shapecond_2/rnn/transpose*
T0*
out_type0*
_output_shapes
:
|
 cond_2/rnn/strided_slice_2/stackConst^cond_2/switch_f*
dtype0*
_output_shapes
:*
valueB:
~
"cond_2/rnn/strided_slice_2/stack_1Const^cond_2/switch_f*
valueB:*
dtype0*
_output_shapes
:
~
"cond_2/rnn/strided_slice_2/stack_2Const^cond_2/switch_f*
valueB:*
dtype0*
_output_shapes
:
║
cond_2/rnn/strided_slice_2StridedSlicecond_2/rnn/Shape_3 cond_2/rnn/strided_slice_2/stack"cond_2/rnn/strided_slice_2/stack_1"cond_2/rnn/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
m
cond_2/rnn/ExpandDims/dimConst^cond_2/switch_f*
dtype0*
_output_shapes
: *
value	B : 
Л
cond_2/rnn/ExpandDims
ExpandDimscond_2/rnn/strided_slice_2cond_2/rnn/ExpandDims/dim*
_output_shapes
:*

Tdim0*
T0
n
cond_2/rnn/Const_1Const^cond_2/switch_f*
_output_shapes
:*
valueB:*
dtype0
l
cond_2/rnn/concat_1/axisConst^cond_2/switch_f*
dtype0*
_output_shapes
: *
value	B : 
Ю
cond_2/rnn/concat_1ConcatV2cond_2/rnn/ExpandDimscond_2/rnn/Const_1cond_2/rnn/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
j
cond_2/rnn/zeros/ConstConst^cond_2/switch_f*
value	B : *
dtype0*
_output_shapes
: 
Й
cond_2/rnn/zerosFillcond_2/rnn/concat_1cond_2/rnn/zeros/Const*'
_output_shapes
:         *
T0*

index_type0
n
cond_2/rnn/Const_2Const^cond_2/switch_f*
valueB: *
dtype0*
_output_shapes
:

cond_2/rnn/MinMincond_2/rnn/CheckSeqLencond_2/rnn/Const_2*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
n
cond_2/rnn/Const_3Const^cond_2/switch_f*
valueB: *
dtype0*
_output_shapes
:

cond_2/rnn/MaxMaxcond_2/rnn/CheckSeqLencond_2/rnn/Const_3*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
c
cond_2/rnn/timeConst^cond_2/switch_f*
value	B : *
dtype0*
_output_shapes
: 
Ш
cond_2/rnn/TensorArrayTensorArrayV3cond_2/rnn/strided_slice_1*6
tensor_array_name!cond_2/rnn/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *$
element_shape:         *
dynamic_size( *
clear_after_read(*
identical_element_shapes(
Щ
cond_2/rnn/TensorArray_1TensorArrayV3cond_2/rnn/strided_slice_1*$
element_shape:         *
dynamic_size( *
clear_after_read(*
identical_element_shapes(*5
tensor_array_name cond_2/rnn/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: 
w
#cond_2/rnn/TensorArrayUnstack/ShapeShapecond_2/rnn/transpose*
_output_shapes
:*
T0*
out_type0
Н
1cond_2/rnn/TensorArrayUnstack/strided_slice/stackConst^cond_2/switch_f*
dtype0*
_output_shapes
:*
valueB: 
П
3cond_2/rnn/TensorArrayUnstack/strided_slice/stack_1Const^cond_2/switch_f*
dtype0*
_output_shapes
:*
valueB:
П
3cond_2/rnn/TensorArrayUnstack/strided_slice/stack_2Const^cond_2/switch_f*
valueB:*
dtype0*
_output_shapes
:
П
+cond_2/rnn/TensorArrayUnstack/strided_sliceStridedSlice#cond_2/rnn/TensorArrayUnstack/Shape1cond_2/rnn/TensorArrayUnstack/strided_slice/stack3cond_2/rnn/TensorArrayUnstack/strided_slice/stack_13cond_2/rnn/TensorArrayUnstack/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
}
)cond_2/rnn/TensorArrayUnstack/range/startConst^cond_2/switch_f*
value	B : *
dtype0*
_output_shapes
: 
}
)cond_2/rnn/TensorArrayUnstack/range/deltaConst^cond_2/switch_f*
value	B :*
dtype0*
_output_shapes
: 
р
#cond_2/rnn/TensorArrayUnstack/rangeRange)cond_2/rnn/TensorArrayUnstack/range/start+cond_2/rnn/TensorArrayUnstack/strided_slice)cond_2/rnn/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:         
Ш
Econd_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3cond_2/rnn/TensorArray_1#cond_2/rnn/TensorArrayUnstack/rangecond_2/rnn/transposecond_2/rnn/TensorArray_1:1*
_output_shapes
: *
T0*'
_class
loc:@cond_2/rnn/transpose
h
cond_2/rnn/Maximum/xConst^cond_2/switch_f*
value	B :*
dtype0*
_output_shapes
: 
d
cond_2/rnn/MaximumMaximumcond_2/rnn/Maximum/xcond_2/rnn/Max*
T0*
_output_shapes
: 
n
cond_2/rnn/MinimumMinimumcond_2/rnn/strided_slice_1cond_2/rnn/Maximum*
T0*
_output_shapes
: 
v
"cond_2/rnn/while/iteration_counterConst^cond_2/switch_f*
value	B : *
dtype0*
_output_shapes
: 
┬
cond_2/rnn/while/EnterEnter"cond_2/rnn/while/iteration_counter*
parallel_iterations *
_output_shapes
: *.

frame_name cond_2/rnn/while/while_context*
T0*
is_constant( 
▒
cond_2/rnn/while/Enter_1Entercond_2/rnn/time*
_output_shapes
: *.

frame_name cond_2/rnn/while/while_context*
T0*
is_constant( *
parallel_iterations 
║
cond_2/rnn/while/Enter_2Entercond_2/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *.

frame_name cond_2/rnn/while/while_context
├
cond_2/rnn/while/Enter_3Entercond_2/Squeeze_1*
T0*
is_constant( *
parallel_iterations *'
_output_shapes
:         *.

frame_name cond_2/rnn/while/while_context
Г
cond_2/rnn/while/MergeMergecond_2/rnn/while/Entercond_2/rnn/while/NextIteration*
T0*
N*
_output_shapes
: : 
Й
cond_2/rnn/while/Merge_1Mergecond_2/rnn/while/Enter_1 cond_2/rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
Й
cond_2/rnn/while/Merge_2Mergecond_2/rnn/while/Enter_2 cond_2/rnn/while/NextIteration_2*
N*
_output_shapes
: : *
T0
Ъ
cond_2/rnn/while/Merge_3Mergecond_2/rnn/while/Enter_3 cond_2/rnn/while/NextIteration_3*
N*)
_output_shapes
:         : *
T0
s
cond_2/rnn/while/LessLesscond_2/rnn/while/Mergecond_2/rnn/while/Less/Enter*
_output_shapes
: *
T0
┐
cond_2/rnn/while/Less/EnterEntercond_2/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name cond_2/rnn/while/while_context
y
cond_2/rnn/while/Less_1Lesscond_2/rnn/while/Merge_1cond_2/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
╣
cond_2/rnn/while/Less_1/EnterEntercond_2/rnn/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name cond_2/rnn/while/while_context
q
cond_2/rnn/while/LogicalAnd
LogicalAndcond_2/rnn/while/Lesscond_2/rnn/while/Less_1*
_output_shapes
: 
Z
cond_2/rnn/while/LoopCondLoopCondcond_2/rnn/while/LogicalAnd*
_output_shapes
: 
в
cond_2/rnn/while/SwitchSwitchcond_2/rnn/while/Mergecond_2/rnn/while/LoopCond*
_output_shapes
: : *
T0*)
_class
loc:@cond_2/rnn/while/Merge
и
cond_2/rnn/while/Switch_1Switchcond_2/rnn/while/Merge_1cond_2/rnn/while/LoopCond*
_output_shapes
: : *
T0*+
_class!
loc:@cond_2/rnn/while/Merge_1
и
cond_2/rnn/while/Switch_2Switchcond_2/rnn/while/Merge_2cond_2/rnn/while/LoopCond*
T0*+
_class!
loc:@cond_2/rnn/while/Merge_2*
_output_shapes
: : 
╩
cond_2/rnn/while/Switch_3Switchcond_2/rnn/while/Merge_3cond_2/rnn/while/LoopCond*:
_output_shapes(
&:         :         *
T0*+
_class!
loc:@cond_2/rnn/while/Merge_3
a
cond_2/rnn/while/IdentityIdentitycond_2/rnn/while/Switch:1*
_output_shapes
: *
T0
e
cond_2/rnn/while/Identity_1Identitycond_2/rnn/while/Switch_1:1*
_output_shapes
: *
T0
e
cond_2/rnn/while/Identity_2Identitycond_2/rnn/while/Switch_2:1*
_output_shapes
: *
T0
v
cond_2/rnn/while/Identity_3Identitycond_2/rnn/while/Switch_3:1*
T0*'
_output_shapes
:         
t
cond_2/rnn/while/add/yConst^cond_2/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
o
cond_2/rnn/while/addAddcond_2/rnn/while/Identitycond_2/rnn/while/add/y*
T0*
_output_shapes
: 
р
"cond_2/rnn/while/TensorArrayReadV3TensorArrayReadV3(cond_2/rnn/while/TensorArrayReadV3/Entercond_2/rnn/while/Identity_1*cond_2/rnn/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:         
╬
(cond_2/rnn/while/TensorArrayReadV3/EnterEntercond_2/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*.

frame_name cond_2/rnn/while/while_context
∙
*cond_2/rnn/while/TensorArrayReadV3/Enter_1EnterEcond_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *.

frame_name cond_2/rnn/while/while_context
Э
cond_2/rnn/while/GreaterEqualGreaterEqualcond_2/rnn/while/Identity_1#cond_2/rnn/while/GreaterEqual/Enter*#
_output_shapes
:         *
T0
╨
#cond_2/rnn/while/GreaterEqual/EnterEntercond_2/rnn/CheckSeqLen*
is_constant(*
parallel_iterations *#
_output_shapes
:         *.

frame_name cond_2/rnn/while/while_context*
T0
}
cond_2/rnn/while/ExpandDims/dimConst^cond_2/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
й
cond_2/rnn/while/ExpandDims
ExpandDimscond_2/rnn/while/Identity_3cond_2/rnn/while/ExpandDims/dim*+
_output_shapes
:         *

Tdim0*
T0
О
cond_2/rnn/while/add_1Addcond_2/rnn/while/ExpandDimscond_2/rnn/while/add_1/Enter*
T0*+
_output_shapes
:         
┼
cond_2/rnn/while/add_1/EnterEntercond_2/ExpandDims_1*
T0*
is_constant(*
parallel_iterations *"
_output_shapes
:*.

frame_name cond_2/rnn/while/while_context
М
&cond_2/rnn/while/Max/reduction_indicesConst^cond_2/rnn/while/Identity*
valueB:*
dtype0*
_output_shapes
:
к
cond_2/rnn/while/MaxMaxcond_2/rnn/while/add_1&cond_2/rnn/while/Max/reduction_indices*'
_output_shapes
:         *

Tidx0*
	keep_dims( *
T0
Й
cond_2/rnn/while/add_2Add"cond_2/rnn/while/TensorArrayReadV3cond_2/rnn/while/Max*
T0*'
_output_shapes
:         

!cond_2/rnn/while/ArgMax/dimensionConst^cond_2/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
н
cond_2/rnn/while/ArgMaxArgMaxcond_2/rnn/while/add_1!cond_2/rnn/while/ArgMax/dimension*
T0*
output_type0	*'
_output_shapes
:         *

Tidx0
w
cond_2/rnn/while/CastCastcond_2/rnn/while/ArgMax*'
_output_shapes
:         *

DstT0*

SrcT0	
╥
cond_2/rnn/while/SelectSelectcond_2/rnn/while/GreaterEqualcond_2/rnn/while/Select/Entercond_2/rnn/while/Cast*(
_class
loc:@cond_2/rnn/while/Cast*'
_output_shapes
:         *
T0
Є
cond_2/rnn/while/Select/EnterEntercond_2/rnn/zeros*
T0*(
_class
loc:@cond_2/rnn/while/Cast*
parallel_iterations *
is_constant(*'
_output_shapes
:         *.

frame_name cond_2/rnn/while/while_context
╘
cond_2/rnn/while/Select_1Selectcond_2/rnn/while/GreaterEqualcond_2/rnn/while/Identity_3cond_2/rnn/while/add_2*
T0*)
_class
loc:@cond_2/rnn/while/add_2*'
_output_shapes
:         
д
4cond_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3:cond_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Entercond_2/rnn/while/Identity_1cond_2/rnn/while/Selectcond_2/rnn/while/Identity_2*
T0*(
_class
loc:@cond_2/rnn/while/Cast*
_output_shapes
: 
И
:cond_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntercond_2/rnn/TensorArray*
is_constant(*
_output_shapes
:*.

frame_name cond_2/rnn/while/while_context*
T0*(
_class
loc:@cond_2/rnn/while/Cast*
parallel_iterations 
v
cond_2/rnn/while/add_3/yConst^cond_2/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
u
cond_2/rnn/while/add_3Addcond_2/rnn/while/Identity_1cond_2/rnn/while/add_3/y*
_output_shapes
: *
T0
f
cond_2/rnn/while/NextIterationNextIterationcond_2/rnn/while/add*
T0*
_output_shapes
: 
j
 cond_2/rnn/while/NextIteration_1NextIterationcond_2/rnn/while/add_3*
T0*
_output_shapes
: 
И
 cond_2/rnn/while/NextIteration_2NextIteration4cond_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
~
 cond_2/rnn/while/NextIteration_3NextIterationcond_2/rnn/while/Select_1*
T0*'
_output_shapes
:         
W
cond_2/rnn/while/ExitExitcond_2/rnn/while/Switch*
T0*
_output_shapes
: 
[
cond_2/rnn/while/Exit_1Exitcond_2/rnn/while/Switch_1*
_output_shapes
: *
T0
[
cond_2/rnn/while/Exit_2Exitcond_2/rnn/while/Switch_2*
T0*
_output_shapes
: 
l
cond_2/rnn/while/Exit_3Exitcond_2/rnn/while/Switch_3*'
_output_shapes
:         *
T0
╢
-cond_2/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3cond_2/rnn/TensorArraycond_2/rnn/while/Exit_2*)
_class
loc:@cond_2/rnn/TensorArray*
_output_shapes
: 
ж
'cond_2/rnn/TensorArrayStack/range/startConst^cond_2/switch_f*
value	B : *)
_class
loc:@cond_2/rnn/TensorArray*
dtype0*
_output_shapes
: 
ж
'cond_2/rnn/TensorArrayStack/range/deltaConst^cond_2/switch_f*
_output_shapes
: *
value	B :*)
_class
loc:@cond_2/rnn/TensorArray*
dtype0
З
!cond_2/rnn/TensorArrayStack/rangeRange'cond_2/rnn/TensorArrayStack/range/start-cond_2/rnn/TensorArrayStack/TensorArraySizeV3'cond_2/rnn/TensorArrayStack/range/delta*)
_class
loc:@cond_2/rnn/TensorArray*#
_output_shapes
:         *

Tidx0
о
/cond_2/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3cond_2/rnn/TensorArray!cond_2/rnn/TensorArrayStack/rangecond_2/rnn/while/Exit_2*)
_class
loc:@cond_2/rnn/TensorArray*
dtype0*4
_output_shapes"
 :                  *$
element_shape:         
n
cond_2/rnn/Const_4Const^cond_2/switch_f*
valueB:*
dtype0*
_output_shapes
:
e
cond_2/rnn/Rank_1Const^cond_2/switch_f*
value	B :*
dtype0*
_output_shapes
: 
l
cond_2/rnn/range_1/startConst^cond_2/switch_f*
value	B :*
dtype0*
_output_shapes
: 
l
cond_2/rnn/range_1/deltaConst^cond_2/switch_f*
value	B :*
dtype0*
_output_shapes
: 
К
cond_2/rnn/range_1Rangecond_2/rnn/range_1/startcond_2/rnn/Rank_1cond_2/rnn/range_1/delta*

Tidx0*
_output_shapes
:

cond_2/rnn/concat_2/values_0Const^cond_2/switch_f*
valueB"       *
dtype0*
_output_shapes
:
l
cond_2/rnn/concat_2/axisConst^cond_2/switch_f*
value	B : *
dtype0*
_output_shapes
: 
е
cond_2/rnn/concat_2ConcatV2cond_2/rnn/concat_2/values_0cond_2/rnn/range_1cond_2/rnn/concat_2/axis*

Tidx0*
T0*
N*
_output_shapes
:
╡
cond_2/rnn/transpose_1	Transpose/cond_2/rnn/TensorArrayStack/TensorArrayGatherV3cond_2/rnn/concat_2*
T0*4
_output_shapes"
 :                  *
Tperm0
b
cond_2/sub_1/yConst^cond_2/switch_f*
dtype0*
_output_shapes
: *
value	B :
d
cond_2/sub_1Subcond_2/sub/Switchcond_2/sub_1/y*
T0*#
_output_shapes
:         
║
cond_2/ReverseSequenceReverseSequencecond_2/rnn/transpose_1cond_2/sub_1*
	batch_dim *
T0*
seq_dim*4
_output_shapes"
 :                  *

Tlen0
m
cond_2/ArgMax_1/dimensionConst^cond_2/switch_f*
value	B :*
dtype0*
_output_shapes
: 
Ъ
cond_2/ArgMax_1ArgMaxcond_2/rnn/while/Exit_3cond_2/ArgMax_1/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:         
c
cond_2/Cast_1Castcond_2/ArgMax_1*

SrcT0	*#
_output_shapes
:         *

DstT0
t
cond_2/ExpandDims_2/dimConst^cond_2/switch_f*
dtype0*
_output_shapes
: *
valueB :
         
З
cond_2/ExpandDims_2
ExpandDimscond_2/Cast_1cond_2/ExpandDims_2/dim*'
_output_shapes
:         *

Tdim0*
T0
b
cond_2/sub_2/yConst^cond_2/switch_f*
value	B :*
dtype0*
_output_shapes
: 
d
cond_2/sub_2Subcond_2/sub/Switchcond_2/sub_2/y*#
_output_shapes
:         *
T0
e
cond_2/rnn_1/RankConst^cond_2/switch_f*
value	B :*
dtype0*
_output_shapes
: 
l
cond_2/rnn_1/range/startConst^cond_2/switch_f*
_output_shapes
: *
value	B :*
dtype0
l
cond_2/rnn_1/range/deltaConst^cond_2/switch_f*
value	B :*
dtype0*
_output_shapes
: 
К
cond_2/rnn_1/rangeRangecond_2/rnn_1/range/startcond_2/rnn_1/Rankcond_2/rnn_1/range/delta*
_output_shapes
:*

Tidx0

cond_2/rnn_1/concat/values_0Const^cond_2/switch_f*
valueB"       *
dtype0*
_output_shapes
:
l
cond_2/rnn_1/concat/axisConst^cond_2/switch_f*
value	B : *
dtype0*
_output_shapes
: 
е
cond_2/rnn_1/concatConcatV2cond_2/rnn_1/concat/values_0cond_2/rnn_1/rangecond_2/rnn_1/concat/axis*
_output_shapes
:*

Tidx0*
T0*
N
Ь
cond_2/rnn_1/transpose	Transposecond_2/ReverseSequencecond_2/rnn_1/concat*4
_output_shapes"
 :                  *
Tperm0*
T0
d
cond_2/rnn_1/sequence_lengthIdentitycond_2/sub_2*#
_output_shapes
:         *
T0
h
cond_2/rnn_1/ShapeShapecond_2/rnn_1/transpose*
T0*
out_type0*
_output_shapes
:
|
 cond_2/rnn_1/strided_slice/stackConst^cond_2/switch_f*
valueB:*
dtype0*
_output_shapes
:
~
"cond_2/rnn_1/strided_slice/stack_1Const^cond_2/switch_f*
_output_shapes
:*
valueB:*
dtype0
~
"cond_2/rnn_1/strided_slice/stack_2Const^cond_2/switch_f*
valueB:*
dtype0*
_output_shapes
:
║
cond_2/rnn_1/strided_sliceStridedSlicecond_2/rnn_1/Shape cond_2/rnn_1/strided_slice/stack"cond_2/rnn_1/strided_slice/stack_1"cond_2/rnn_1/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
p
cond_2/rnn_1/Shape_1Shapecond_2/rnn_1/sequence_length*
T0*
out_type0*
_output_shapes
:
p
cond_2/rnn_1/stackPackcond_2/rnn_1/strided_slice*
_output_shapes
:*
T0*

axis *
N
j
cond_2/rnn_1/EqualEqualcond_2/rnn_1/Shape_1cond_2/rnn_1/stack*
T0*
_output_shapes
:
n
cond_2/rnn_1/ConstConst^cond_2/switch_f*
_output_shapes
:*
valueB: *
dtype0
t
cond_2/rnn_1/AllAllcond_2/rnn_1/Equalcond_2/rnn_1/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
и
cond_2/rnn_1/Assert/ConstConst^cond_2/switch_f*
_output_shapes
: *M
valueDBB B<Expected shape for Tensor cond_2/rnn_1/sequence_length:0 is *
dtype0
~
cond_2/rnn_1/Assert/Const_1Const^cond_2/switch_f*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
░
!cond_2/rnn_1/Assert/Assert/data_0Const^cond_2/switch_f*M
valueDBB B<Expected shape for Tensor cond_2/rnn_1/sequence_length:0 is *
dtype0*
_output_shapes
: 
Д
!cond_2/rnn_1/Assert/Assert/data_2Const^cond_2/switch_f*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
╞
cond_2/rnn_1/Assert/AssertAssertcond_2/rnn_1/All!cond_2/rnn_1/Assert/Assert/data_0cond_2/rnn_1/stack!cond_2/rnn_1/Assert/Assert/data_2cond_2/rnn_1/Shape_1*
T
2*
	summarize
Н
cond_2/rnn_1/CheckSeqLenIdentitycond_2/rnn_1/sequence_length^cond_2/rnn_1/Assert/Assert*#
_output_shapes
:         *
T0
j
cond_2/rnn_1/Shape_2Shapecond_2/rnn_1/transpose*
T0*
out_type0*
_output_shapes
:
~
"cond_2/rnn_1/strided_slice_1/stackConst^cond_2/switch_f*
_output_shapes
:*
valueB: *
dtype0
А
$cond_2/rnn_1/strided_slice_1/stack_1Const^cond_2/switch_f*
_output_shapes
:*
valueB:*
dtype0
А
$cond_2/rnn_1/strided_slice_1/stack_2Const^cond_2/switch_f*
valueB:*
dtype0*
_output_shapes
:
─
cond_2/rnn_1/strided_slice_1StridedSlicecond_2/rnn_1/Shape_2"cond_2/rnn_1/strided_slice_1/stack$cond_2/rnn_1/strided_slice_1/stack_1$cond_2/rnn_1/strided_slice_1/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
j
cond_2/rnn_1/Shape_3Shapecond_2/rnn_1/transpose*
_output_shapes
:*
T0*
out_type0
~
"cond_2/rnn_1/strided_slice_2/stackConst^cond_2/switch_f*
valueB:*
dtype0*
_output_shapes
:
А
$cond_2/rnn_1/strided_slice_2/stack_1Const^cond_2/switch_f*
valueB:*
dtype0*
_output_shapes
:
А
$cond_2/rnn_1/strided_slice_2/stack_2Const^cond_2/switch_f*
_output_shapes
:*
valueB:*
dtype0
─
cond_2/rnn_1/strided_slice_2StridedSlicecond_2/rnn_1/Shape_3"cond_2/rnn_1/strided_slice_2/stack$cond_2/rnn_1/strided_slice_2/stack_1$cond_2/rnn_1/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
o
cond_2/rnn_1/ExpandDims/dimConst^cond_2/switch_f*
value	B : *
dtype0*
_output_shapes
: 
С
cond_2/rnn_1/ExpandDims
ExpandDimscond_2/rnn_1/strided_slice_2cond_2/rnn_1/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
p
cond_2/rnn_1/Const_1Const^cond_2/switch_f*
dtype0*
_output_shapes
:*
valueB:
n
cond_2/rnn_1/concat_1/axisConst^cond_2/switch_f*
_output_shapes
: *
value	B : *
dtype0
ж
cond_2/rnn_1/concat_1ConcatV2cond_2/rnn_1/ExpandDimscond_2/rnn_1/Const_1cond_2/rnn_1/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
l
cond_2/rnn_1/zeros/ConstConst^cond_2/switch_f*
value	B : *
dtype0*
_output_shapes
: 
П
cond_2/rnn_1/zerosFillcond_2/rnn_1/concat_1cond_2/rnn_1/zeros/Const*

index_type0*'
_output_shapes
:         *
T0
p
cond_2/rnn_1/Const_2Const^cond_2/switch_f*
valueB: *
dtype0*
_output_shapes
:
Е
cond_2/rnn_1/MinMincond_2/rnn_1/CheckSeqLencond_2/rnn_1/Const_2*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
p
cond_2/rnn_1/Const_3Const^cond_2/switch_f*
dtype0*
_output_shapes
:*
valueB: 
Е
cond_2/rnn_1/MaxMaxcond_2/rnn_1/CheckSeqLencond_2/rnn_1/Const_3*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
e
cond_2/rnn_1/timeConst^cond_2/switch_f*
value	B : *
dtype0*
_output_shapes
: 
Ю
cond_2/rnn_1/TensorArrayTensorArrayV3cond_2/rnn_1/strided_slice_1*$
element_shape:         *
dynamic_size( *
clear_after_read(*
identical_element_shapes(*8
tensor_array_name#!cond_2/rnn_1/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: 
Я
cond_2/rnn_1/TensorArray_1TensorArrayV3cond_2/rnn_1/strided_slice_1*7
tensor_array_name" cond_2/rnn_1/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *$
element_shape:         *
dynamic_size( *
clear_after_read(*
identical_element_shapes(
{
%cond_2/rnn_1/TensorArrayUnstack/ShapeShapecond_2/rnn_1/transpose*
T0*
out_type0*
_output_shapes
:
П
3cond_2/rnn_1/TensorArrayUnstack/strided_slice/stackConst^cond_2/switch_f*
valueB: *
dtype0*
_output_shapes
:
С
5cond_2/rnn_1/TensorArrayUnstack/strided_slice/stack_1Const^cond_2/switch_f*
valueB:*
dtype0*
_output_shapes
:
С
5cond_2/rnn_1/TensorArrayUnstack/strided_slice/stack_2Const^cond_2/switch_f*
valueB:*
dtype0*
_output_shapes
:
Щ
-cond_2/rnn_1/TensorArrayUnstack/strided_sliceStridedSlice%cond_2/rnn_1/TensorArrayUnstack/Shape3cond_2/rnn_1/TensorArrayUnstack/strided_slice/stack5cond_2/rnn_1/TensorArrayUnstack/strided_slice/stack_15cond_2/rnn_1/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 

+cond_2/rnn_1/TensorArrayUnstack/range/startConst^cond_2/switch_f*
value	B : *
dtype0*
_output_shapes
: 

+cond_2/rnn_1/TensorArrayUnstack/range/deltaConst^cond_2/switch_f*
value	B :*
dtype0*
_output_shapes
: 
ш
%cond_2/rnn_1/TensorArrayUnstack/rangeRange+cond_2/rnn_1/TensorArrayUnstack/range/start-cond_2/rnn_1/TensorArrayUnstack/strided_slice+cond_2/rnn_1/TensorArrayUnstack/range/delta*#
_output_shapes
:         *

Tidx0
д
Gcond_2/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3cond_2/rnn_1/TensorArray_1%cond_2/rnn_1/TensorArrayUnstack/rangecond_2/rnn_1/transposecond_2/rnn_1/TensorArray_1:1*)
_class
loc:@cond_2/rnn_1/transpose*
_output_shapes
: *
T0
j
cond_2/rnn_1/Maximum/xConst^cond_2/switch_f*
dtype0*
_output_shapes
: *
value	B :
j
cond_2/rnn_1/MaximumMaximumcond_2/rnn_1/Maximum/xcond_2/rnn_1/Max*
T0*
_output_shapes
: 
t
cond_2/rnn_1/MinimumMinimumcond_2/rnn_1/strided_slice_1cond_2/rnn_1/Maximum*
_output_shapes
: *
T0
x
$cond_2/rnn_1/while/iteration_counterConst^cond_2/switch_f*
value	B : *
dtype0*
_output_shapes
: 
╚
cond_2/rnn_1/while/EnterEnter$cond_2/rnn_1/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *0

frame_name" cond_2/rnn_1/while/while_context
╖
cond_2/rnn_1/while/Enter_1Entercond_2/rnn_1/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *0

frame_name" cond_2/rnn_1/while/while_context
└
cond_2/rnn_1/while/Enter_2Entercond_2/rnn_1/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *0

frame_name" cond_2/rnn_1/while/while_context
╩
cond_2/rnn_1/while/Enter_3Entercond_2/ExpandDims_2*
T0*
is_constant( *
parallel_iterations *'
_output_shapes
:         *0

frame_name" cond_2/rnn_1/while/while_context
Й
cond_2/rnn_1/while/MergeMergecond_2/rnn_1/while/Enter cond_2/rnn_1/while/NextIteration*
T0*
N*
_output_shapes
: : 
П
cond_2/rnn_1/while/Merge_1Mergecond_2/rnn_1/while/Enter_1"cond_2/rnn_1/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
П
cond_2/rnn_1/while/Merge_2Mergecond_2/rnn_1/while/Enter_2"cond_2/rnn_1/while/NextIteration_2*
N*
_output_shapes
: : *
T0
а
cond_2/rnn_1/while/Merge_3Mergecond_2/rnn_1/while/Enter_3"cond_2/rnn_1/while/NextIteration_3*)
_output_shapes
:         : *
T0*
N
y
cond_2/rnn_1/while/LessLesscond_2/rnn_1/while/Mergecond_2/rnn_1/while/Less/Enter*
T0*
_output_shapes
: 
┼
cond_2/rnn_1/while/Less/EnterEntercond_2/rnn_1/strided_slice_1*
parallel_iterations *
_output_shapes
: *0

frame_name" cond_2/rnn_1/while/while_context*
T0*
is_constant(

cond_2/rnn_1/while/Less_1Lesscond_2/rnn_1/while/Merge_1cond_2/rnn_1/while/Less_1/Enter*
_output_shapes
: *
T0
┐
cond_2/rnn_1/while/Less_1/EnterEntercond_2/rnn_1/Minimum*
is_constant(*
parallel_iterations *
_output_shapes
: *0

frame_name" cond_2/rnn_1/while/while_context*
T0
w
cond_2/rnn_1/while/LogicalAnd
LogicalAndcond_2/rnn_1/while/Lesscond_2/rnn_1/while/Less_1*
_output_shapes
: 
^
cond_2/rnn_1/while/LoopCondLoopCondcond_2/rnn_1/while/LogicalAnd*
_output_shapes
: 
к
cond_2/rnn_1/while/SwitchSwitchcond_2/rnn_1/while/Mergecond_2/rnn_1/while/LoopCond*
T0*+
_class!
loc:@cond_2/rnn_1/while/Merge*
_output_shapes
: : 
░
cond_2/rnn_1/while/Switch_1Switchcond_2/rnn_1/while/Merge_1cond_2/rnn_1/while/LoopCond*
_output_shapes
: : *
T0*-
_class#
!loc:@cond_2/rnn_1/while/Merge_1
░
cond_2/rnn_1/while/Switch_2Switchcond_2/rnn_1/while/Merge_2cond_2/rnn_1/while/LoopCond*
T0*-
_class#
!loc:@cond_2/rnn_1/while/Merge_2*
_output_shapes
: : 
╥
cond_2/rnn_1/while/Switch_3Switchcond_2/rnn_1/while/Merge_3cond_2/rnn_1/while/LoopCond*:
_output_shapes(
&:         :         *
T0*-
_class#
!loc:@cond_2/rnn_1/while/Merge_3
e
cond_2/rnn_1/while/IdentityIdentitycond_2/rnn_1/while/Switch:1*
T0*
_output_shapes
: 
i
cond_2/rnn_1/while/Identity_1Identitycond_2/rnn_1/while/Switch_1:1*
_output_shapes
: *
T0
i
cond_2/rnn_1/while/Identity_2Identitycond_2/rnn_1/while/Switch_2:1*
T0*
_output_shapes
: 
z
cond_2/rnn_1/while/Identity_3Identitycond_2/rnn_1/while/Switch_3:1*
T0*'
_output_shapes
:         
x
cond_2/rnn_1/while/add/yConst^cond_2/rnn_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
u
cond_2/rnn_1/while/addAddcond_2/rnn_1/while/Identitycond_2/rnn_1/while/add/y*
T0*
_output_shapes
: 
ш
$cond_2/rnn_1/while/TensorArrayReadV3TensorArrayReadV3*cond_2/rnn_1/while/TensorArrayReadV3/Entercond_2/rnn_1/while/Identity_1,cond_2/rnn_1/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:         
╘
*cond_2/rnn_1/while/TensorArrayReadV3/EnterEntercond_2/rnn_1/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*0

frame_name" cond_2/rnn_1/while/while_context
 
,cond_2/rnn_1/while/TensorArrayReadV3/Enter_1EnterGcond_2/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *0

frame_name" cond_2/rnn_1/while/while_context
г
cond_2/rnn_1/while/GreaterEqualGreaterEqualcond_2/rnn_1/while/Identity_1%cond_2/rnn_1/while/GreaterEqual/Enter*
T0*#
_output_shapes
:         
╓
%cond_2/rnn_1/while/GreaterEqual/EnterEntercond_2/rnn_1/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *#
_output_shapes
:         *0

frame_name" cond_2/rnn_1/while/while_context
Й
cond_2/rnn_1/while/SqueezeSqueezecond_2/rnn_1/while/Identity_3*
T0*#
_output_shapes
:         *
squeeze_dims

|
cond_2/rnn_1/while/ShapeShape$cond_2/rnn_1/while/TensorArrayReadV3*
out_type0*
_output_shapes
:*
T0
О
&cond_2/rnn_1/while/strided_slice/stackConst^cond_2/rnn_1/while/Identity*
valueB: *
dtype0*
_output_shapes
:
Р
(cond_2/rnn_1/while/strided_slice/stack_1Const^cond_2/rnn_1/while/Identity*
valueB:*
dtype0*
_output_shapes
:
Р
(cond_2/rnn_1/while/strided_slice/stack_2Const^cond_2/rnn_1/while/Identity*
dtype0*
_output_shapes
:*
valueB:
╪
 cond_2/rnn_1/while/strided_sliceStridedSlicecond_2/rnn_1/while/Shape&cond_2/rnn_1/while/strided_slice/stack(cond_2/rnn_1/while/strided_slice/stack_1(cond_2/rnn_1/while/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
~
cond_2/rnn_1/while/range/startConst^cond_2/rnn_1/while/Identity*
value	B : *
dtype0*
_output_shapes
: 
~
cond_2/rnn_1/while/range/deltaConst^cond_2/rnn_1/while/Identity*
_output_shapes
: *
value	B :*
dtype0
┤
cond_2/rnn_1/while/rangeRangecond_2/rnn_1/while/range/start cond_2/rnn_1/while/strided_slicecond_2/rnn_1/while/range/delta*#
_output_shapes
:         *

Tidx0
Э
cond_2/rnn_1/while/stackPackcond_2/rnn_1/while/rangecond_2/rnn_1/while/Squeeze*
T0*

axis*
N*'
_output_shapes
:         
л
cond_2/rnn_1/while/GatherNdGatherNd$cond_2/rnn_1/while/TensorArrayReadV3cond_2/rnn_1/while/stack*
Tindices0*
Tparams0*#
_output_shapes
:         
К
!cond_2/rnn_1/while/ExpandDims/dimConst^cond_2/rnn_1/while/Identity*
valueB :
         *
dtype0*
_output_shapes
: 
й
cond_2/rnn_1/while/ExpandDims
ExpandDimscond_2/rnn_1/while/GatherNd!cond_2/rnn_1/while/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
ш
cond_2/rnn_1/while/SelectSelectcond_2/rnn_1/while/GreaterEqualcond_2/rnn_1/while/Select/Entercond_2/rnn_1/while/ExpandDims*
T0*0
_class&
$"loc:@cond_2/rnn_1/while/ExpandDims*'
_output_shapes
:         
А
cond_2/rnn_1/while/Select/EnterEntercond_2/rnn_1/zeros*0
_class&
$"loc:@cond_2/rnn_1/while/ExpandDims*
parallel_iterations *
is_constant(*'
_output_shapes
:         *0

frame_name" cond_2/rnn_1/while/while_context*
T0
ш
cond_2/rnn_1/while/Select_1Selectcond_2/rnn_1/while/GreaterEqualcond_2/rnn_1/while/Identity_3cond_2/rnn_1/while/ExpandDims*
T0*0
_class&
$"loc:@cond_2/rnn_1/while/ExpandDims*'
_output_shapes
:         
╢
6cond_2/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3<cond_2/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/Entercond_2/rnn_1/while/Identity_1cond_2/rnn_1/while/Selectcond_2/rnn_1/while/Identity_2*
_output_shapes
: *
T0*0
_class&
$"loc:@cond_2/rnn_1/while/ExpandDims
Ц
<cond_2/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntercond_2/rnn_1/TensorArray*
is_constant(*
_output_shapes
:*0

frame_name" cond_2/rnn_1/while/while_context*
T0*0
_class&
$"loc:@cond_2/rnn_1/while/ExpandDims*
parallel_iterations 
z
cond_2/rnn_1/while/add_1/yConst^cond_2/rnn_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
{
cond_2/rnn_1/while/add_1Addcond_2/rnn_1/while/Identity_1cond_2/rnn_1/while/add_1/y*
T0*
_output_shapes
: 
j
 cond_2/rnn_1/while/NextIterationNextIterationcond_2/rnn_1/while/add*
T0*
_output_shapes
: 
n
"cond_2/rnn_1/while/NextIteration_1NextIterationcond_2/rnn_1/while/add_1*
_output_shapes
: *
T0
М
"cond_2/rnn_1/while/NextIteration_2NextIteration6cond_2/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
В
"cond_2/rnn_1/while/NextIteration_3NextIterationcond_2/rnn_1/while/Select_1*
T0*'
_output_shapes
:         
[
cond_2/rnn_1/while/ExitExitcond_2/rnn_1/while/Switch*
_output_shapes
: *
T0
_
cond_2/rnn_1/while/Exit_1Exitcond_2/rnn_1/while/Switch_1*
T0*
_output_shapes
: 
_
cond_2/rnn_1/while/Exit_2Exitcond_2/rnn_1/while/Switch_2*
T0*
_output_shapes
: 
p
cond_2/rnn_1/while/Exit_3Exitcond_2/rnn_1/while/Switch_3*'
_output_shapes
:         *
T0
╛
/cond_2/rnn_1/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3cond_2/rnn_1/TensorArraycond_2/rnn_1/while/Exit_2*+
_class!
loc:@cond_2/rnn_1/TensorArray*
_output_shapes
: 
к
)cond_2/rnn_1/TensorArrayStack/range/startConst^cond_2/switch_f*
value	B : *+
_class!
loc:@cond_2/rnn_1/TensorArray*
dtype0*
_output_shapes
: 
к
)cond_2/rnn_1/TensorArrayStack/range/deltaConst^cond_2/switch_f*
_output_shapes
: *
value	B :*+
_class!
loc:@cond_2/rnn_1/TensorArray*
dtype0
С
#cond_2/rnn_1/TensorArrayStack/rangeRange)cond_2/rnn_1/TensorArrayStack/range/start/cond_2/rnn_1/TensorArrayStack/TensorArraySizeV3)cond_2/rnn_1/TensorArrayStack/range/delta*#
_output_shapes
:         *

Tidx0*+
_class!
loc:@cond_2/rnn_1/TensorArray
╕
1cond_2/rnn_1/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3cond_2/rnn_1/TensorArray#cond_2/rnn_1/TensorArrayStack/rangecond_2/rnn_1/while/Exit_2*
dtype0*4
_output_shapes"
 :                  *$
element_shape:         *+
_class!
loc:@cond_2/rnn_1/TensorArray
p
cond_2/rnn_1/Const_4Const^cond_2/switch_f*
valueB:*
dtype0*
_output_shapes
:
g
cond_2/rnn_1/Rank_1Const^cond_2/switch_f*
dtype0*
_output_shapes
: *
value	B :
n
cond_2/rnn_1/range_1/startConst^cond_2/switch_f*
value	B :*
dtype0*
_output_shapes
: 
n
cond_2/rnn_1/range_1/deltaConst^cond_2/switch_f*
_output_shapes
: *
value	B :*
dtype0
Т
cond_2/rnn_1/range_1Rangecond_2/rnn_1/range_1/startcond_2/rnn_1/Rank_1cond_2/rnn_1/range_1/delta*
_output_shapes
:*

Tidx0
Б
cond_2/rnn_1/concat_2/values_0Const^cond_2/switch_f*
dtype0*
_output_shapes
:*
valueB"       
n
cond_2/rnn_1/concat_2/axisConst^cond_2/switch_f*
value	B : *
dtype0*
_output_shapes
: 
н
cond_2/rnn_1/concat_2ConcatV2cond_2/rnn_1/concat_2/values_0cond_2/rnn_1/range_1cond_2/rnn_1/concat_2/axis*
_output_shapes
:*

Tidx0*
T0*
N
╗
cond_2/rnn_1/transpose_1	Transpose1cond_2/rnn_1/TensorArrayStack/TensorArrayGatherV3cond_2/rnn_1/concat_2*4
_output_shapes"
 :                  *
Tperm0*
T0
З
cond_2/Squeeze_2Squeezecond_2/rnn_1/transpose_1*0
_output_shapes
:                  *
squeeze_dims
*
T0
f
cond_2/concat/axisConst^cond_2/switch_f*
dtype0*
_output_shapes
: *
value	B :
д
cond_2/concatConcatV2cond_2/ExpandDims_2cond_2/Squeeze_2cond_2/concat/axis*
T0*
N*0
_output_shapes
:                  *

Tidx0
┤
cond_2/ReverseSequence_1ReverseSequencecond_2/concatcond_2/sub/Switch*
	batch_dim *
T0*
seq_dim*0
_output_shapes
:                  *

Tlen0
r
cond_2/Max_1/reduction_indicesConst^cond_2/switch_f*
value	B :*
dtype0*
_output_shapes
: 
Ч
cond_2/Max_1Maxcond_2/rnn/while/Exit_3cond_2/Max_1/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:         
В
cond_2/MergeMergecond_2/ReverseSequence_1cond_2/Cast*
T0*
N*2
_output_shapes 
:                  : 
j
cond_2/Merge_1Mergecond_2/Max_1
cond_2/Max*
T0*
N*%
_output_shapes
:         : 
┼	
initNoOp(^embedding_layer/embedding_matrix/Assign;^biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Assign9^biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Assign;^biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Assign9^biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Assign^biLSTM_layers/W_out/Assign^biLSTM_layers/b/Assign^transitions/Assign^beta1_power/Assign^beta2_power/Assign-^embedding_layer/embedding_matrix/Adam/Assign/^embedding_layer/embedding_matrix/Adam_1/Assign@^biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam/AssignB^biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Assign>^biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam/Assign@^biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Assign@^biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam/AssignB^biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Assign>^biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam/Assign@^biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Assign ^biLSTM_layers/W_out/Adam/Assign"^biLSTM_layers/W_out/Adam_1/Assign^biLSTM_layers/b/Adam/Assign^biLSTM_layers/b/Adam_1/Assign^transitions/Adam/Assign^transitions/Adam_1/Assign
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_d0ec464144cd4fb9893edb7b3da94921/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
М
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
┌
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*■
valueЇBёBbeta1_powerBbeta2_powerBbiLSTM_layers/W_outBbiLSTM_layers/W_out/AdamBbiLSTM_layers/W_out/Adam_1BbiLSTM_layers/bBbiLSTM_layers/b/AdamBbiLSTM_layers/b/Adam_1B1biLSTM_layers/bidirectional_rnn/bw/lstm_cell/biasB6biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/AdamB8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B3biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernelB8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/AdamB:biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B1biLSTM_layers/bidirectional_rnn/fw/lstm_cell/biasB6biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/AdamB8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B3biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernelB8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/AdamB:biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B embedding_layer/embedding_matrixB%embedding_layer/embedding_matrix/AdamB'embedding_layer/embedding_matrix/Adam_1BtransitionsBtransitions/AdamBtransitions/Adam_1*
dtype0
ж
save/SaveV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Е	
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_powerbiLSTM_layers/W_outbiLSTM_layers/W_out/AdambiLSTM_layers/W_out/Adam_1biLSTM_layers/bbiLSTM_layers/b/AdambiLSTM_layers/b/Adam_11biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias6biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_13biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam:biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_11biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias6biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_13biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam:biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1 embedding_layer/embedding_matrix%embedding_layer/embedding_matrix/Adam'embedding_layer/embedding_matrix/Adam_1transitionstransitions/Adamtransitions/Adam_1"/device:CPU:0*(
dtypes
2
а
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: *
T0
м
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
М
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
Й
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0
▌
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*■
valueЇBёBbeta1_powerBbeta2_powerBbiLSTM_layers/W_outBbiLSTM_layers/W_out/AdamBbiLSTM_layers/W_out/Adam_1BbiLSTM_layers/bBbiLSTM_layers/b/AdamBbiLSTM_layers/b/Adam_1B1biLSTM_layers/bidirectional_rnn/bw/lstm_cell/biasB6biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/AdamB8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1B3biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernelB8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/AdamB:biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1B1biLSTM_layers/bidirectional_rnn/fw/lstm_cell/biasB6biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/AdamB8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1B3biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernelB8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/AdamB:biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1B embedding_layer/embedding_matrixB%embedding_layer/embedding_matrix/AdamB'embedding_layer/embedding_matrix/Adam_1BtransitionsBtransitions/AdamBtransitions/Adam_1
й
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ь
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2
д
save/AssignAssignbeta1_powersave/RestoreV2*
use_locking(*
T0*&
_class
loc:@biLSTM_layers/W_out*
validate_shape(*
_output_shapes
: 
и
save/Assign_1Assignbeta2_powersave/RestoreV2:1*
use_locking(*
T0*&
_class
loc:@biLSTM_layers/W_out*
validate_shape(*
_output_shapes
: 
╣
save/Assign_2AssignbiLSTM_layers/W_outsave/RestoreV2:2*
T0*&
_class
loc:@biLSTM_layers/W_out*
validate_shape(*
_output_shapes
:	А*
use_locking(
╛
save/Assign_3AssignbiLSTM_layers/W_out/Adamsave/RestoreV2:3*
use_locking(*
T0*&
_class
loc:@biLSTM_layers/W_out*
validate_shape(*
_output_shapes
:	А
└
save/Assign_4AssignbiLSTM_layers/W_out/Adam_1save/RestoreV2:4*
use_locking(*
T0*&
_class
loc:@biLSTM_layers/W_out*
validate_shape(*
_output_shapes
:	А
м
save/Assign_5AssignbiLSTM_layers/bsave/RestoreV2:5*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@biLSTM_layers/b
▒
save/Assign_6AssignbiLSTM_layers/b/Adamsave/RestoreV2:6*
use_locking(*
T0*"
_class
loc:@biLSTM_layers/b*
validate_shape(*
_output_shapes
:
│
save/Assign_7AssignbiLSTM_layers/b/Adam_1save/RestoreV2:7*"
_class
loc:@biLSTM_layers/b*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
ё
save/Assign_8Assign1biLSTM_layers/bidirectional_rnn/bw/lstm_cell/biassave/RestoreV2:8*
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
Ў
save/Assign_9Assign6biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adamsave/RestoreV2:9*
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
·
save/Assign_10Assign8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1save/RestoreV2:10*
_output_shapes	
:А*
use_locking(*
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias*
validate_shape(
№
save/Assign_11Assign3biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernelsave/RestoreV2:11* 
_output_shapes
:
АА*
use_locking(*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(
Б
save/Assign_12Assign8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adamsave/RestoreV2:12*
use_locking(*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА
Г
save/Assign_13Assign:biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1save/RestoreV2:13*
use_locking(*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА
є
save/Assign_14Assign1biLSTM_layers/bidirectional_rnn/fw/lstm_cell/biassave/RestoreV2:14*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias
°
save/Assign_15Assign6biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adamsave/RestoreV2:15*
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
·
save/Assign_16Assign8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1save/RestoreV2:16*
use_locking(*
T0*D
_class:
86loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
№
save/Assign_17Assign3biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernelsave/RestoreV2:17*
use_locking(*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА
Б
save/Assign_18Assign8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adamsave/RestoreV2:18*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
Г
save/Assign_19Assign:biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1save/RestoreV2:19*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*F
_class<
:8loc:@biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel
╓
save/Assign_20Assign embedding_layer/embedding_matrixsave/RestoreV2:20*
validate_shape(* 
_output_shapes
:
Д/А*
use_locking(*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix
█
save/Assign_21Assign%embedding_layer/embedding_matrix/Adamsave/RestoreV2:21*3
_class)
'%loc:@embedding_layer/embedding_matrix*
validate_shape(* 
_output_shapes
:
Д/А*
use_locking(*
T0
▌
save/Assign_22Assign'embedding_layer/embedding_matrix/Adam_1save/RestoreV2:22*
use_locking(*
T0*3
_class)
'%loc:@embedding_layer/embedding_matrix*
validate_shape(* 
_output_shapes
:
Д/А
к
save/Assign_23Assigntransitionssave/RestoreV2:23*
use_locking(*
T0*
_class
loc:@transitions*
validate_shape(*
_output_shapes

:
п
save/Assign_24Assigntransitions/Adamsave/RestoreV2:24*
use_locking(*
T0*
_class
loc:@transitions*
validate_shape(*
_output_shapes

:
▒
save/Assign_25Assigntransitions/Adam_1save/RestoreV2:25*
_output_shapes

:*
use_locking(*
T0*
_class
loc:@transitions*
validate_shape(
╚
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"Я
trainable_variablesЗД
╡
"embedding_layer/embedding_matrix:0'embedding_layer/embedding_matrix/Assign'embedding_layer/embedding_matrix/read:02=embedding_layer/embedding_matrix/Initializer/random_uniform:0
Б
5biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel:0:biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Assign:biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/read:02PbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform:0
Ё
3biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias:08biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Assign8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/read:02EbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zeros:0
Б
5biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel:0:biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Assign:biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/read:02PbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform:0
Ё
3biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias:08biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Assign8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/read:02EbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zeros:0
Г
biLSTM_layers/W_out:0biLSTM_layers/W_out/AssignbiLSTM_layers/W_out/read:022biLSTM_layers/W_out/Initializer/truncated_normal:0
s
biLSTM_layers/b:0biLSTM_layers/b/AssignbiLSTM_layers/b/read:02.biLSTM_layers/b/Initializer/truncated_normal:0
a
transitions:0transitions/Assigntransitions/read:02(transitions/Initializer/random_uniform:0"вл
cond_contextРлМл
▀
cond/cond_textcond/pred_id:0cond/switch_t:0 *й
biLSTM_layers/Reshape_1:0
cond/GatherNd:0
cond/Reshape/shape:0
cond/Reshape:0
cond/Shape/Switch:1
cond/Shape:0
cond/Squeeze:0
cond/concat/Switch:1
cond/concat/axis:0
cond/concat:0
cond/pred_id:0
cond/range/delta:0
cond/range/start:0
cond/range:0
cond/strided_slice/stack:0
cond/strided_slice/stack_1:0
cond/strided_slice/stack_2:0
cond/strided_slice:0
cond/switch_t:0
	targets:00
biLSTM_layers/Reshape_1:0cond/Shape/Switch:1 
cond/pred_id:0cond/pred_id:0!
	targets:0cond/concat/Switch:1"
cond/switch_t:0cond/switch_t:0
·
cond/cond_text_1cond/pred_id:0cond/switch_f:0*─
Sum:0
biLSTM_layers/Reshape_1:0
cond/ExpandDims/dim:0
cond/ExpandDims:0
cond/ExpandDims_1/dim:0
cond/ExpandDims_1:0
cond/Gather:0
cond/Gather_1:0
cond/Reshape_1/shape:0
cond/Reshape_1:0
cond/Reshape_2/shape:0
cond/Reshape_2:0
cond/Reshape_3/shape:0
cond/Reshape_3:0
cond/Reshape_4/Switch:0
cond/Reshape_4/shape:0
cond/Reshape_4:0
cond/SequenceMask/Cast:0
cond/SequenceMask/Cast_1:0
cond/SequenceMask/Const:0
cond/SequenceMask/Const_1:0
%cond/SequenceMask/ExpandDims/Switch:0
"cond/SequenceMask/ExpandDims/dim:0
cond/SequenceMask/ExpandDims:0
cond/SequenceMask/Less:0
cond/SequenceMask/Range:0
cond/SequenceMask_1/Cast:0
cond/SequenceMask_1/Cast_1:0
cond/SequenceMask_1/Const:0
cond/SequenceMask_1/Const_1:0
$cond/SequenceMask_1/ExpandDims/dim:0
 cond/SequenceMask_1/ExpandDims:0
cond/SequenceMask_1/Less:0
cond/SequenceMask_1/Range:0
cond/Shape_1/Switch:0
cond/Shape_1:0
cond/Shape_2:0
cond/Shape_3:0
cond/Shape_4:0
cond/Shape_5:0
cond/Shape_6:0
cond/Slice/begin:0
cond/Slice/size/0:0
cond/Slice/size:0
cond/Slice:0
cond/Slice_1/begin:0
cond/Slice_1/size/0:0
cond/Slice_1/size:0
cond/Slice_1:0
cond/Slice_2/begin:0
cond/Slice_2/size:0
cond/Slice_2:0
cond/Sum/reduction_indices:0

cond/Sum:0
cond/Sum_1/reduction_indices:0
cond/Sum_1:0

cond/add:0
cond/add_1/Switch:0
cond/add_1:0
cond/add_2:0
cond/add_3:0

cond/mul:0
cond/mul_1:0
cond/mul_2:0
cond/mul_3:0
cond/mul_4/y:0
cond/mul_4:0
cond/mul_5:0
cond/pred_id:0
cond/range_1/delta:0
cond/range_1/start:0
cond/range_1:0
cond/range_2/delta:0
cond/range_2/start:0
cond/range_2:0
cond/strided_slice_1/stack:0
cond/strided_slice_1/stack_1:0
cond/strided_slice_1/stack_2:0
cond/strided_slice_1:0
cond/strided_slice_2/stack:0
cond/strided_slice_2/stack_1:0
cond/strided_slice_2/stack_2:0
cond/strided_slice_2:0
cond/strided_slice_3/stack:0
cond/strided_slice_3/stack_1:0
cond/strided_slice_3/stack_2:0
cond/strided_slice_3:0
cond/strided_slice_4/stack:0
cond/strided_slice_4/stack_1:0
cond/strided_slice_4/stack_2:0
cond/strided_slice_4:0
cond/strided_slice_5/stack:0
cond/strided_slice_5/stack_1:0
cond/strided_slice_5/stack_2:0
cond/strided_slice_5:0
cond/strided_slice_6/stack:0
cond/strided_slice_6/stack_1:0
cond/strided_slice_6/stack_2:0
cond/strided_slice_6:0
cond/sub/y:0

cond/sub:0
cond/switch_f:0
	targets:0
transitions/read:02
biLSTM_layers/Reshape_1:0cond/Shape_1/Switch:0-
transitions/read:0cond/Reshape_4/Switch:0 
cond/pred_id:0cond/pred_id:0.
Sum:0%cond/SequenceMask/ExpandDims/Switch:0 
	targets:0cond/add_1/Switch:0"
cond/switch_f:0cond/switch_f:0
ў
cond_1/cond_textcond_1/pred_id:0cond_1/switch_t:0 *╗
	Squeeze:0
cond_1/ReduceLogSumExp/Exp:0
!cond_1/ReduceLogSumExp/IsFinite:0
cond_1/ReduceLogSumExp/Log:0
#cond_1/ReduceLogSumExp/Max/Switch:1
.cond_1/ReduceLogSumExp/Max/reduction_indices:0
cond_1/ReduceLogSumExp/Max:0
 cond_1/ReduceLogSumExp/Reshape:0
cond_1/ReduceLogSumExp/Select:0
cond_1/ReduceLogSumExp/Shape:0
%cond_1/ReduceLogSumExp/StopGradient:0
.cond_1/ReduceLogSumExp/Sum/reduction_indices:0
cond_1/ReduceLogSumExp/Sum:0
cond_1/ReduceLogSumExp/add:0
cond_1/ReduceLogSumExp/sub:0
#cond_1/ReduceLogSumExp/zeros_like:0
cond_1/pred_id:0
cond_1/switch_t:0&
cond_1/switch_t:0cond_1/switch_t:00
	Squeeze:0#cond_1/ReduceLogSumExp/Max/Switch:1$
cond_1/pred_id:0cond_1/pred_id:0
╘е
cond_1/cond_text_1cond_1/pred_id:0cond_1/switch_f:0*рA
	Squeeze:0
Sum:0
biLSTM_layers/Reshape_1:0
cond_1/ExpandDims/Switch:0
cond_1/ExpandDims/dim:0
cond_1/ExpandDims:0
cond_1/ReduceLogSumExp_1/Exp:0
#cond_1/ReduceLogSumExp_1/IsFinite:0
cond_1/ReduceLogSumExp_1/Log:0
0cond_1/ReduceLogSumExp_1/Max/reduction_indices:0
cond_1/ReduceLogSumExp_1/Max:0
"cond_1/ReduceLogSumExp_1/Reshape:0
!cond_1/ReduceLogSumExp_1/Select:0
 cond_1/ReduceLogSumExp_1/Shape:0
'cond_1/ReduceLogSumExp_1/StopGradient:0
0cond_1/ReduceLogSumExp_1/Sum/reduction_indices:0
cond_1/ReduceLogSumExp_1/Sum:0
cond_1/ReduceLogSumExp_1/add:0
cond_1/ReduceLogSumExp_1/sub:0
%cond_1/ReduceLogSumExp_1/zeros_like:0
cond_1/Slice/Switch:0
cond_1/Slice/begin:0
cond_1/Slice/size:0
cond_1/Slice:0
cond_1/pred_id:0
cond_1/rnn/All:0
!cond_1/rnn/Assert/Assert/data_0:0
!cond_1/rnn/Assert/Assert/data_2:0
cond_1/rnn/Assert/Const:0
cond_1/rnn/Assert/Const_1:0
cond_1/rnn/CheckSeqLen:0
cond_1/rnn/Const:0
cond_1/rnn/Const_1:0
cond_1/rnn/Const_2:0
cond_1/rnn/Const_3:0
cond_1/rnn/Const_4:0
cond_1/rnn/Equal:0
cond_1/rnn/ExpandDims/dim:0
cond_1/rnn/ExpandDims:0
cond_1/rnn/Max:0
cond_1/rnn/Maximum/x:0
cond_1/rnn/Maximum:0
cond_1/rnn/Min:0
cond_1/rnn/Minimum:0
cond_1/rnn/Rank:0
cond_1/rnn/Rank_1:0
cond_1/rnn/Shape:0
cond_1/rnn/Shape_1:0
cond_1/rnn/Shape_2:0
cond_1/rnn/Shape_3:0
cond_1/rnn/TensorArray:0
cond_1/rnn/TensorArray:1
1cond_1/rnn/TensorArrayStack/TensorArrayGatherV3:0
/cond_1/rnn/TensorArrayStack/TensorArraySizeV3:0
)cond_1/rnn/TensorArrayStack/range/delta:0
)cond_1/rnn/TensorArrayStack/range/start:0
#cond_1/rnn/TensorArrayStack/range:0
%cond_1/rnn/TensorArrayUnstack/Shape:0
Gcond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
+cond_1/rnn/TensorArrayUnstack/range/delta:0
+cond_1/rnn/TensorArrayUnstack/range/start:0
%cond_1/rnn/TensorArrayUnstack/range:0
3cond_1/rnn/TensorArrayUnstack/strided_slice/stack:0
5cond_1/rnn/TensorArrayUnstack/strided_slice/stack_1:0
5cond_1/rnn/TensorArrayUnstack/strided_slice/stack_2:0
-cond_1/rnn/TensorArrayUnstack/strided_slice:0
cond_1/rnn/TensorArray_1:0
cond_1/rnn/TensorArray_1:1
cond_1/rnn/concat/axis:0
cond_1/rnn/concat/values_0:0
cond_1/rnn/concat:0
cond_1/rnn/concat_1/axis:0
cond_1/rnn/concat_1:0
cond_1/rnn/concat_2/axis:0
cond_1/rnn/concat_2/values_0:0
cond_1/rnn/concat_2:0
cond_1/rnn/range/delta:0
cond_1/rnn/range/start:0
cond_1/rnn/range:0
cond_1/rnn/range_1/delta:0
cond_1/rnn/range_1/start:0
cond_1/rnn/range_1:0
cond_1/rnn/sequence_length:0
cond_1/rnn/stack:0
 cond_1/rnn/strided_slice/stack:0
"cond_1/rnn/strided_slice/stack_1:0
"cond_1/rnn/strided_slice/stack_2:0
cond_1/rnn/strided_slice:0
"cond_1/rnn/strided_slice_1/stack:0
$cond_1/rnn/strided_slice_1/stack_1:0
$cond_1/rnn/strided_slice_1/stack_2:0
cond_1/rnn/strided_slice_1:0
"cond_1/rnn/strided_slice_2/stack:0
$cond_1/rnn/strided_slice_2/stack_1:0
$cond_1/rnn/strided_slice_2/stack_2:0
cond_1/rnn/strided_slice_2:0
cond_1/rnn/time:0
cond_1/rnn/transpose:0
cond_1/rnn/transpose_1:0
cond_1/rnn/while/Exit:0
cond_1/rnn/while/Exit_1:0
cond_1/rnn/while/Exit_2:0
cond_1/rnn/while/Exit_3:0
cond_1/rnn/while/Switch:0
$cond_1/rnn/while/iteration_counter:0
cond_1/rnn/zeros/Const:0
cond_1/rnn/zeros:0
cond_1/sub/Switch:0
cond_1/sub/y:0
cond_1/sub:0
cond_1/switch_f:0
gradients/b_count:0
gradients/b_count_3:0
.gradients/cond_1/rnn/while/Enter_3_grad/Exit:0
;gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/Switch:0
:gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/f_acc:0
@gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/Switch:0
?gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/f_acc:0
Ggradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/Switch:0
Fgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/f_acc:0
Hgradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/Switch:0
Ggradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/f_acc:0
Jgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/Switch:0
Igradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/f_acc:0
Rgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch:0
Tgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_2:0
Qgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/f_acc:0
Sgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/f_acc_1:0
Rgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch:0
Tgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_2:0
Qgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/f_acc:0
Sgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/f_acc_1:0
8gradients/cond_1/rnn/while/Select_1_grad/Select/Switch:0
7gradients/cond_1/rnn/while/Select_1_grad/Select/f_acc:0
<gradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch:0
;gradients/cond_1/rnn/while/Select_1_grad/zeros_like/f_acc:0
Agradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc:0
Cgradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3:0
^gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Switch:0
]gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc:0
3gradients/cond_1/rnn/while/add_1/Enter_grad/b_acc:0
5gradients/cond_1/rnn/while/add_1/Enter_grad/b_acc_3:0
Dgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/Switch:0
Cgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/f_acc:0
Dgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch:0
Fgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_2:0
Cgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/f_acc:0
Egradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/f_acc_1:0
gradients/f_count:0
gradients/f_count_2:0
transitions/read:0У
Ggradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/f_acc:0Hgradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/Switch:0л
Sgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/f_acc_1:0Tgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_2:0{
;gradients/cond_1/rnn/while/Select_1_grad/zeros_like/f_acc:0<gradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch:0С
Fgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/f_acc:0Ggradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/Switch:0з
Qgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/f_acc:0Rgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch:0з
Qgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/f_acc:0Rgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch:0$
cond_1/pred_id:0cond_1/pred_id:0л
Sgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/f_acc_1:0Tgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_2:0Ч
Igradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/f_acc:0Jgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/Switch:0Л
Cgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/f_acc:0Dgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch:0
Sum:0cond_1/sub/Switch:0&
cond_1/switch_f:0cond_1/switch_f:0Л
Cgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/f_acc:0Dgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/Switch:0s
7gradients/cond_1/rnn/while/Select_1_grad/Select/f_acc:08gradients/cond_1/rnn/while/Select_1_grad/Select/Switch:02
biLSTM_layers/Reshape_1:0cond_1/Slice/Switch:00
transitions/read:0cond_1/ExpandDims/Switch:0y
:gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/f_acc:0;gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/Switch:0Г
?gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/f_acc:0@gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/Switch:0&
	Squeeze:0cond_1/rnn/while/Switch:0П
Egradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/f_acc_1:0Fgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_2:0┐
]gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc:0^gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Switch:02╤N╬N
cond_1/rnn/while/while_context *cond_1/rnn/while/LoopCond:02cond_1/rnn/while/Merge:0:cond_1/rnn/while/Identity:0Bcond_1/rnn/while/Exit:0Bcond_1/rnn/while/Exit_1:0Bcond_1/rnn/while/Exit_2:0Bcond_1/rnn/while/Exit_3:0Bgradients/f_count_2:0JпK
cond_1/ExpandDims:0
cond_1/rnn/CheckSeqLen:0
cond_1/rnn/Minimum:0
cond_1/rnn/TensorArray:0
Gcond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
cond_1/rnn/TensorArray_1:0
cond_1/rnn/strided_slice_1:0
cond_1/rnn/while/Enter:0
cond_1/rnn/while/Enter_1:0
cond_1/rnn/while/Enter_2:0
cond_1/rnn/while/Enter_3:0
cond_1/rnn/while/Exit:0
cond_1/rnn/while/Exit_1:0
cond_1/rnn/while/Exit_2:0
cond_1/rnn/while/Exit_3:0
!cond_1/rnn/while/ExpandDims/dim:0
cond_1/rnn/while/ExpandDims:0
%cond_1/rnn/while/GreaterEqual/Enter:0
cond_1/rnn/while/GreaterEqual:0
cond_1/rnn/while/Identity:0
cond_1/rnn/while/Identity_1:0
cond_1/rnn/while/Identity_2:0
cond_1/rnn/while/Identity_3:0
cond_1/rnn/while/Less/Enter:0
cond_1/rnn/while/Less:0
cond_1/rnn/while/Less_1/Enter:0
cond_1/rnn/while/Less_1:0
cond_1/rnn/while/LogicalAnd:0
cond_1/rnn/while/LoopCond:0
cond_1/rnn/while/Merge:0
cond_1/rnn/while/Merge:1
cond_1/rnn/while/Merge_1:0
cond_1/rnn/while/Merge_1:1
cond_1/rnn/while/Merge_2:0
cond_1/rnn/while/Merge_2:1
cond_1/rnn/while/Merge_3:0
cond_1/rnn/while/Merge_3:1
 cond_1/rnn/while/NextIteration:0
"cond_1/rnn/while/NextIteration_1:0
"cond_1/rnn/while/NextIteration_2:0
"cond_1/rnn/while/NextIteration_3:0
&cond_1/rnn/while/ReduceLogSumExp/Exp:0
+cond_1/rnn/while/ReduceLogSumExp/IsFinite:0
&cond_1/rnn/while/ReduceLogSumExp/Log:0
8cond_1/rnn/while/ReduceLogSumExp/Max/reduction_indices:0
&cond_1/rnn/while/ReduceLogSumExp/Max:0
*cond_1/rnn/while/ReduceLogSumExp/Reshape:0
)cond_1/rnn/while/ReduceLogSumExp/Select:0
(cond_1/rnn/while/ReduceLogSumExp/Shape:0
/cond_1/rnn/while/ReduceLogSumExp/StopGradient:0
8cond_1/rnn/while/ReduceLogSumExp/Sum/reduction_indices:0
&cond_1/rnn/while/ReduceLogSumExp/Sum:0
&cond_1/rnn/while/ReduceLogSumExp/add:0
&cond_1/rnn/while/ReduceLogSumExp/sub:0
-cond_1/rnn/while/ReduceLogSumExp/zeros_like:0
cond_1/rnn/while/Select/Enter:0
cond_1/rnn/while/Select:0
cond_1/rnn/while/Select_1:0
cond_1/rnn/while/Switch_1:0
cond_1/rnn/while/Switch_1:1
cond_1/rnn/while/Switch_2:0
cond_1/rnn/while/Switch_2:1
cond_1/rnn/while/Switch_3:0
cond_1/rnn/while/Switch_3:1
cond_1/rnn/while/Switch_4:0
cond_1/rnn/while/Switch_4:1
*cond_1/rnn/while/TensorArrayReadV3/Enter:0
,cond_1/rnn/while/TensorArrayReadV3/Enter_1:0
$cond_1/rnn/while/TensorArrayReadV3:0
<cond_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
6cond_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
cond_1/rnn/while/add/y:0
cond_1/rnn/while/add:0
cond_1/rnn/while/add_1/Enter:0
cond_1/rnn/while/add_1:0
cond_1/rnn/while/add_2:0
cond_1/rnn/while/add_3/y:0
cond_1/rnn/while/add_3:0
cond_1/rnn/zeros:0
gradients/Add/y:0
gradients/Add:0
gradients/Merge:0
gradients/Merge:1
gradients/NextIteration:0
gradients/Switch:0
gradients/Switch:1
:gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/Enter:0
@gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/StackPushV2:0
:gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/f_acc:0
2gradients/cond_1/rnn/while/ExpandDims_grad/Shape:0
?gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/Enter:0
Egradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/StackPushV2:0
?gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/f_acc:0
Fgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/Enter:0
Lgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/StackPushV2:0
Fgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/f_acc:0
Ggradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/Enter:0
Mgradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/StackPushV2:0
Ggradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/f_acc:0
?gradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Shape:0
Igradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/Enter:0
Ogradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/StackPushV2:0
Igradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/f_acc:0
;gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape:0
Qgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Enter:0
Sgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Enter_1:0
Wgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/StackPushV2:0
Ygradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/StackPushV2_1:0
Qgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/f_acc:0
Sgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/f_acc_1:0
;gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Shape:0
=gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Shape_1:0
Qgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Enter:0
Sgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Enter_1:0
Wgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/StackPushV2:0
Ygradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/StackPushV2_1:0
Qgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/f_acc:0
Sgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/f_acc_1:0
;gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Shape:0
=gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Shape_1:0
7gradients/cond_1/rnn/while/Select_1_grad/Select/Enter:0
=gradients/cond_1/rnn/while/Select_1_grad/Select/StackPushV2:0
7gradients/cond_1/rnn/while/Select_1_grad/Select/f_acc:0
;gradients/cond_1/rnn/while/Select_1_grad/zeros_like/Enter:0
Agradients/cond_1/rnn/while/Select_1_grad/zeros_like/StackPushV2:0
;gradients/cond_1/rnn/while/Select_1_grad/zeros_like/f_acc:0
]gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Enter:0
cgradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPushV2:0
]gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc:0
Cgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/Enter:0
Igradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/StackPushV2:0
Cgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/f_acc:0
-gradients/cond_1/rnn/while/add_1_grad/Shape:0
Cgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Enter:0
Egradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Enter_1:0
Igradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/StackPushV2:0
Kgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/StackPushV2_1:0
Cgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/f_acc:0
Egradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/f_acc_1:0
-gradients/cond_1/rnn/while/add_2_grad/Shape:0
/gradients/cond_1/rnn/while/add_2_grad/Shape_1:0
gradients/f_count:0
gradients/f_count_1:0
gradients/f_count_2:0x
:gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/f_acc:0:gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/Enter:0В
?gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/f_acc:0?gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/Enter:0=
cond_1/rnn/strided_slice_1:0cond_1/rnn/while/Less/Enter:0О
Egradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/f_acc_1:0Egradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Enter_1:0╛
]gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc:0]gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Enter:0Т
Ggradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/f_acc:0Ggradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/Enter:0z
;gradients/cond_1/rnn/while/Select_1_grad/zeros_like/f_acc:0;gradients/cond_1/rnn/while/Select_1_grad/zeros_like/Enter:0к
Sgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/f_acc_1:0Sgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Enter_1:0Р
Fgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/f_acc:0Fgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/Enter:0ж
Qgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/f_acc:0Qgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Enter:0ж
Qgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/f_acc:0Qgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Enter:05
cond_1/rnn/zeros:0cond_1/rnn/while/Select/Enter:0к
Sgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/f_acc_1:0Sgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Enter_1:0A
cond_1/rnn/CheckSeqLen:0%cond_1/rnn/while/GreaterEqual/Enter:0Ц
Igradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/f_acc:0Igradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/Enter:0К
Cgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/f_acc:0Cgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Enter:05
cond_1/ExpandDims:0cond_1/rnn/while/add_1/Enter:0К
Cgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/f_acc:0Cgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/Enter:0H
cond_1/rnn/TensorArray_1:0*cond_1/rnn/while/TensorArrayReadV3/Enter:0w
Gcond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0,cond_1/rnn/while/TensorArrayReadV3/Enter_1:0X
cond_1/rnn/TensorArray:0<cond_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0r
7gradients/cond_1/rnn/while/Select_1_grad/Select/f_acc:07gradients/cond_1/rnn/while/Select_1_grad/Select/Enter:07
cond_1/rnn/Minimum:0cond_1/rnn/while/Less_1/Enter:0Rcond_1/rnn/while/Enter:0Rcond_1/rnn/while/Enter_1:0Rcond_1/rnn/while/Enter_2:0Rcond_1/rnn/while/Enter_3:0Rgradients/f_count_1:0Zcond_1/rnn/strided_slice_1:02рФ▄Ф
(gradients/cond_1/rnn/while/while_context *gradients/b_count_2:02gradients/Merge_1:0:gradients/Sub:0Bgradients/b_count_3:0B.gradients/cond_1/rnn/while/Enter_3_grad/Exit:0BCgradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3:0B5gradients/cond_1/rnn/while/add_1/Enter_grad/b_acc_3:0JЧР
cond_1/pred_id:0
Gcond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
cond_1/rnn/TensorArray_1:0
cond_1/rnn/while/GreaterEqual:0
cond_1/rnn/while/Identity_1:0
cond_1/rnn/while/Identity_3:0
&cond_1/rnn/while/ReduceLogSumExp/Exp:0
8cond_1/rnn/while/ReduceLogSumExp/Sum/reduction_indices:0
&cond_1/rnn/while/ReduceLogSumExp/Sum:0
*cond_1/rnn/while/TensorArrayReadV3/Enter:0
,cond_1/rnn/while/TensorArrayReadV3/Enter_1:0
gradients/AddN_3:0
gradients/AddN_4:0
gradients/GreaterEqual/Enter:0
gradients/GreaterEqual:0
gradients/Merge_1:0
gradients/Merge_1:1
gradients/NextIteration_1:0
gradients/Sub:0
gradients/Switch_1:0
gradients/Switch_1:1
gradients/b_count:0
gradients/b_count_1:0
gradients/b_count_2:0
gradients/b_count_3:0
Fgradients/cond_1/ReduceLogSumExp_1/sub_grad/tuple/control_dependency:0
.gradients/cond_1/rnn/while/Enter_3_grad/Exit:0
/gradients/cond_1/rnn/while/Exit_2_grad/b_exit:0
/gradients/cond_1/rnn/while/Exit_3_grad/b_exit:0
?gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/StackPopV2:0
Cgradients/cond_1/rnn/while/ExpandDims_grad/Reshape/Switch_1/Enter:0
=gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/Switch_1:0
=gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/Switch_1:1
:gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/f_acc:0
4gradients/cond_1/rnn/while/ExpandDims_grad/Reshape:0
2gradients/cond_1/rnn/while/ExpandDims_grad/Shape:0
0gradients/cond_1/rnn/while/Merge_3_grad/Switch:0
0gradients/cond_1/rnn/while/Merge_3_grad/Switch:1
Bgradients/cond_1/rnn/while/Merge_3_grad/tuple/control_dependency:0
Dgradients/cond_1/rnn/while/Merge_3_grad/tuple/control_dependency_1:0
Dgradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/StackPopV2:0
Hgradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/Switch_1/Enter:0
Bgradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/Switch_1:0
Bgradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/Switch_1:1
?gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/f_acc:0
9gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul:0
Kgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/StackPopV2:0
Ogradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/Switch_1/Enter:0
Igradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/Switch_1:0
Igradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/Switch_1:1
Fgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/f_acc:0
@gradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal:0
9gradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/mul:0
Lgradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/StackPopV2:0
Pgradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/Switch_1/Enter:0
Jgradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/Switch_1:0
Jgradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/Switch_1:1
Ggradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/f_acc:0
Agradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape:0
?gradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Shape:0
Ngradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/StackPopV2:0
Rgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/Switch_1/Enter:0
Lgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/Switch_1:0
Lgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/Switch_1:1
Igradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/f_acc:0
Cgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch:0
@gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Fill/value:0
:gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Fill:0
?gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Maximum/y:0
=gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Maximum:0
=gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Reshape:0
;gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape:0
=gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape_1:0
:gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Size:0
:gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Tile:0
?gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/add/Const:0
9gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/add:0
>gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/floordiv:0
9gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/mod:0
Agradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/range/delta:0
Agradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/range/start:0
;gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/range:0
Vgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/StackPopV2:0
Xgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/StackPopV2_1:0
Zgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_1/Enter:0
Tgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_1:0
Tgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_1:1
Zgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_3/Enter:0
Tgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_3:0
Tgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_3:1
Qgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/f_acc:0
Sgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/f_acc_1:0
Kgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs:0
Kgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs:1
=gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Reshape:0
?gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Reshape_1:0
;gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Shape:0
=gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Shape_1:0
9gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Sum:0
;gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Sum_1:0
Ngradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/tuple/control_dependency:0
Pgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/tuple/control_dependency_1:0
Vgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/StackPopV2:0
Xgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/StackPopV2_1:0
Zgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_1/Enter:0
Tgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_1:0
Tgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_1:1
Zgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_3/Enter:0
Tgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_3:0
Tgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_3:1
Qgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/f_acc:0
Sgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/f_acc_1:0
Kgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs:0
Kgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs:1
9gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Neg:0
=gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Reshape:0
?gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Reshape_1:0
;gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Shape:0
=gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Shape_1:0
9gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Sum:0
;gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Sum_1:0
Ngradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/tuple/control_dependency:0
Pgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/tuple/control_dependency_1:0
<gradients/cond_1/rnn/while/Select_1_grad/Select/StackPopV2:0
@gradients/cond_1/rnn/while/Select_1_grad/Select/Switch_1/Enter:0
:gradients/cond_1/rnn/while/Select_1_grad/Select/Switch_1:0
:gradients/cond_1/rnn/while/Select_1_grad/Select/Switch_1:1
7gradients/cond_1/rnn/while/Select_1_grad/Select/f_acc:0
1gradients/cond_1/rnn/while/Select_1_grad/Select:0
3gradients/cond_1/rnn/while/Select_1_grad/Select_1:0
Cgradients/cond_1/rnn/while/Select_1_grad/tuple/control_dependency:0
Egradients/cond_1/rnn/while/Select_1_grad/tuple/control_dependency_1:0
@gradients/cond_1/rnn/while/Select_1_grad/zeros_like/StackPopV2:0
Dgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter:0
Fgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter_1:0
>gradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1:0
>gradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1:1
;gradients/cond_1/rnn/while/Select_1_grad/zeros_like/f_acc:0
5gradients/cond_1/rnn/while/Select_1_grad/zeros_like:0
3gradients/cond_1/rnn/while/Switch_4_grad/b_switch:0
3gradients/cond_1/rnn/while/Switch_4_grad/b_switch:1
:gradients/cond_1/rnn/while/Switch_4_grad_1/NextIteration:0
?gradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Add:0
Igradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration:0
Bgradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:0
Bgradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1
Agradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc:0
Cgradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1:0
Cgradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2:0
Cgradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2:1
Cgradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3:0
[gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter:0
]gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1:0
Ugradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3:0
Ugradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3:1
Qgradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow:0
bgradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopV2:0
fgradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Switch_1/Enter:0
`gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Switch_1:0
`gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Switch_1:1
]gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc:0
Wgradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3:0
1gradients/cond_1/rnn/while/add_1/Enter_grad/Add:0
;gradients/cond_1/rnn/while/add_1/Enter_grad/NextIteration:0
4gradients/cond_1/rnn/while/add_1/Enter_grad/Switch:0
4gradients/cond_1/rnn/while/add_1/Enter_grad/Switch:1
3gradients/cond_1/rnn/while/add_1/Enter_grad/b_acc:0
5gradients/cond_1/rnn/while/add_1/Enter_grad/b_acc_1:0
5gradients/cond_1/rnn/while/add_1/Enter_grad/b_acc_2:0
5gradients/cond_1/rnn/while/add_1/Enter_grad/b_acc_2:1
5gradients/cond_1/rnn/while/add_1/Enter_grad/b_acc_3:0
Hgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/StackPopV2:0
Lgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/Switch_1/Enter:0
Fgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/Switch_1:0
Fgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/Switch_1:1
Cgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/f_acc:0
=gradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs:0
=gradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs:1
/gradients/cond_1/rnn/while/add_1_grad/Reshape:0
1gradients/cond_1/rnn/while/add_1_grad/Reshape_1:0
-gradients/cond_1/rnn/while/add_1_grad/Shape:0
/gradients/cond_1/rnn/while/add_1_grad/Shape_1:0
+gradients/cond_1/rnn/while/add_1_grad/Sum:0
-gradients/cond_1/rnn/while/add_1_grad/Sum_1:0
@gradients/cond_1/rnn/while/add_1_grad/tuple/control_dependency:0
Bgradients/cond_1/rnn/while/add_1_grad/tuple/control_dependency_1:0
Hgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/StackPopV2:0
Jgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/StackPopV2_1:0
Lgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_1/Enter:0
Fgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_1:0
Fgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_1:1
Lgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_3/Enter:0
Fgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_3:0
Fgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_3:1
Cgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/f_acc:0
Egradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/f_acc_1:0
=gradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs:0
=gradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs:1
/gradients/cond_1/rnn/while/add_2_grad/Reshape:0
1gradients/cond_1/rnn/while/add_2_grad/Reshape_1:0
-gradients/cond_1/rnn/while/add_2_grad/Shape:0
/gradients/cond_1/rnn/while/add_2_grad/Shape_1:0
+gradients/cond_1/rnn/while/add_2_grad/Sum:0
-gradients/cond_1/rnn/while/add_2_grad/Sum_1:0
@gradients/cond_1/rnn/while/add_2_grad/tuple/control_dependency:0
Bgradients/cond_1/rnn/while/add_2_grad/tuple/control_dependency_1:0
gradients/f_count_2:0
gradients/zeros_2:0u
2gradients/cond_1/rnn/while/ExpandDims_grad/Shape:0?gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/StackPopV2:0{
7gradients/cond_1/rnn/while/Select_1_grad/Select/f_acc:0@gradients/cond_1/rnn/while/Select_1_grad/Select/Switch_1/Enter:0Г
cond_1/rnn/while/Identity_1:0bgradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/StackPopV2:0П
?gradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Shape:0Lgradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/StackPopV2:0_
cond_1/rnn/while/GreaterEqual:0<gradients/cond_1/rnn/while/Select_1_grad/Select/StackPopV2:0y
-gradients/cond_1/rnn/while/add_1_grad/Shape:0Hgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/StackPopV2:0Б
:gradients/cond_1/rnn/while/ExpandDims_grad/Reshape/f_acc:0Cgradients/cond_1/rnn/while/ExpandDims_grad/Reshape/Switch_1/Enter:0Щ
=gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Shape_1:0Xgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/StackPopV2_1:0Л
?gradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/f_acc:0Hgradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/Switch_1/Enter:0Х
Egradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/f_acc_1:0Lgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_3/Enter:0Н
,cond_1/rnn/while/TensorArrayReadV3/Enter_1:0]gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1:0u
&cond_1/rnn/while/ReduceLogSumExp/Sum:0Kgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/StackPopV2:0╟
]gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/f_acc:0fgradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3/Switch_1/Enter:0Ы
Ggradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/f_acc:0Pgradients/cond_1/rnn/while/ReduceLogSumExp/Reshape_grad/Reshape/Switch_1/Enter:0}
/gradients/cond_1/rnn/while/add_2_grad/Shape_1:0Jgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/StackPopV2_1:0Щ
Fgradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/f_acc:0Ogradients/cond_1/rnn/while/ReduceLogSumExp/Log_grad/Reciprocal/Switch_1/Enter:0п
Qgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/f_acc:0Zgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_1/Enter:0Z
cond_1/pred_id:0Fgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter_1:0{
8cond_1/rnn/while/ReduceLogSumExp/Sum/reduction_indices:0?gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/add/Const:0п
Qgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/f_acc:0Zgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_1/Enter:0▒
Sgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/f_acc_1:0Zgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/Switch_3/Enter:0Я
Igradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/f_acc:0Rgradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/Switch_1/Enter:0Щ
=gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Shape_1:0Xgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/StackPopV2_1:05
gradients/b_count:0gradients/GreaterEqual/Enter:0y
cond_1/rnn/TensorArray_1:0[gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter:0y
-gradients/cond_1/rnn/while/add_2_grad/Shape:0Hgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/StackPopV2:0и
Gcond_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0]gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1:0Х
;gradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/Shape:0Vgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/StackPopV2:0Й
*cond_1/rnn/while/TensorArrayReadV3/Enter:0[gradients/cond_1/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter:0n
&cond_1/rnn/while/ReduceLogSumExp/Exp:0Dgradients/cond_1/rnn/while/ReduceLogSumExp/Exp_grad/mul/StackPopV2:0Х
;gradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/Shape:0Vgradients/cond_1/rnn/while/ReduceLogSumExp/add_grad/BroadcastGradientArgs/StackPopV2:0Н
;gradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/Shape:0Ngradients/cond_1/rnn/while/ReduceLogSumExp/Sum_grad/DynamicStitch/StackPopV2:0Г
;gradients/cond_1/rnn/while/Select_1_grad/zeros_like/f_acc:0Dgradients/cond_1/rnn/while/Select_1_grad/zeros_like/Switch_1/Enter:0▒
Sgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/f_acc_1:0Zgradients/cond_1/rnn/while/ReduceLogSumExp/sub_grad/BroadcastGradientArgs/Switch_3/Enter:0a
cond_1/rnn/while/Identity_3:0@gradients/cond_1/rnn/while/Select_1_grad/zeros_like/StackPopV2:0У
Cgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/f_acc:0Lgradients/cond_1/rnn/while/add_2_grad/BroadcastGradientArgs/Switch_1/Enter:0У
Cgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/f_acc:0Lgradients/cond_1/rnn/while/add_1_grad/BroadcastGradientArgs/Switch_1/Enter:0Rgradients/b_count_1:0R/gradients/cond_1/rnn/while/Exit_3_grad/b_exit:0R/gradients/cond_1/rnn/while/Exit_2_grad/b_exit:0RCgradients/cond_1/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1:0R5gradients/cond_1/rnn/while/add_1/Enter_grad/b_acc_1:0Zcond_1/rnn/strided_slice_1:0
┬
cond_2/cond_textcond_2/pred_id:0cond_2/switch_t:0 *Ж
biLSTM_layers/Reshape_1:0
cond_2/ArgMax/dimension:0
cond_2/ArgMax:0
cond_2/Cast:0
cond_2/ExpandDims/dim:0
cond_2/ExpandDims:0
cond_2/Max/reduction_indices:0
cond_2/Max:0
cond_2/Squeeze/Switch:1
cond_2/Squeeze:0
cond_2/pred_id:0
cond_2/switch_t:0$
cond_2/pred_id:0cond_2/pred_id:0&
cond_2/switch_t:0cond_2/switch_t:04
biLSTM_layers/Reshape_1:0cond_2/Squeeze/Switch:1
Уb
cond_2/cond_text_1cond_2/pred_id:0cond_2/switch_f:0*¤.
Sum:0
biLSTM_layers/Reshape_1:0
cond_2/ArgMax_1/dimension:0
cond_2/ArgMax_1:0
cond_2/Cast_1:0
cond_2/ExpandDims_1/Switch:0
cond_2/ExpandDims_1/dim:0
cond_2/ExpandDims_1:0
cond_2/ExpandDims_2/dim:0
cond_2/ExpandDims_2:0
 cond_2/Max_1/reduction_indices:0
cond_2/Max_1:0
cond_2/ReverseSequence:0
cond_2/ReverseSequence_1:0
cond_2/Slice/Switch:0
cond_2/Slice/begin:0
cond_2/Slice/size:0
cond_2/Slice:0
cond_2/Slice_1/begin:0
cond_2/Slice_1/size:0
cond_2/Slice_1:0
cond_2/Squeeze_1:0
cond_2/Squeeze_2:0
cond_2/concat/axis:0
cond_2/concat:0
cond_2/pred_id:0
cond_2/rnn/All:0
!cond_2/rnn/Assert/Assert/data_0:0
!cond_2/rnn/Assert/Assert/data_2:0
cond_2/rnn/Assert/Const:0
cond_2/rnn/Assert/Const_1:0
cond_2/rnn/CheckSeqLen:0
cond_2/rnn/Const:0
cond_2/rnn/Const_1:0
cond_2/rnn/Const_2:0
cond_2/rnn/Const_3:0
cond_2/rnn/Const_4:0
cond_2/rnn/Equal:0
cond_2/rnn/ExpandDims/dim:0
cond_2/rnn/ExpandDims:0
cond_2/rnn/Max:0
cond_2/rnn/Maximum/x:0
cond_2/rnn/Maximum:0
cond_2/rnn/Min:0
cond_2/rnn/Minimum:0
cond_2/rnn/Rank:0
cond_2/rnn/Rank_1:0
cond_2/rnn/Shape:0
cond_2/rnn/Shape_1:0
cond_2/rnn/Shape_2:0
cond_2/rnn/Shape_3:0
cond_2/rnn/TensorArray:0
cond_2/rnn/TensorArray:1
1cond_2/rnn/TensorArrayStack/TensorArrayGatherV3:0
/cond_2/rnn/TensorArrayStack/TensorArraySizeV3:0
)cond_2/rnn/TensorArrayStack/range/delta:0
)cond_2/rnn/TensorArrayStack/range/start:0
#cond_2/rnn/TensorArrayStack/range:0
%cond_2/rnn/TensorArrayUnstack/Shape:0
Gcond_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
+cond_2/rnn/TensorArrayUnstack/range/delta:0
+cond_2/rnn/TensorArrayUnstack/range/start:0
%cond_2/rnn/TensorArrayUnstack/range:0
3cond_2/rnn/TensorArrayUnstack/strided_slice/stack:0
5cond_2/rnn/TensorArrayUnstack/strided_slice/stack_1:0
5cond_2/rnn/TensorArrayUnstack/strided_slice/stack_2:0
-cond_2/rnn/TensorArrayUnstack/strided_slice:0
cond_2/rnn/TensorArray_1:0
cond_2/rnn/TensorArray_1:1
cond_2/rnn/concat/axis:0
cond_2/rnn/concat/values_0:0
cond_2/rnn/concat:0
cond_2/rnn/concat_1/axis:0
cond_2/rnn/concat_1:0
cond_2/rnn/concat_2/axis:0
cond_2/rnn/concat_2/values_0:0
cond_2/rnn/concat_2:0
cond_2/rnn/range/delta:0
cond_2/rnn/range/start:0
cond_2/rnn/range:0
cond_2/rnn/range_1/delta:0
cond_2/rnn/range_1/start:0
cond_2/rnn/range_1:0
cond_2/rnn/sequence_length:0
cond_2/rnn/stack:0
 cond_2/rnn/strided_slice/stack:0
"cond_2/rnn/strided_slice/stack_1:0
"cond_2/rnn/strided_slice/stack_2:0
cond_2/rnn/strided_slice:0
"cond_2/rnn/strided_slice_1/stack:0
$cond_2/rnn/strided_slice_1/stack_1:0
$cond_2/rnn/strided_slice_1/stack_2:0
cond_2/rnn/strided_slice_1:0
"cond_2/rnn/strided_slice_2/stack:0
$cond_2/rnn/strided_slice_2/stack_1:0
$cond_2/rnn/strided_slice_2/stack_2:0
cond_2/rnn/strided_slice_2:0
cond_2/rnn/time:0
cond_2/rnn/transpose:0
cond_2/rnn/transpose_1:0
cond_2/rnn/while/Exit:0
cond_2/rnn/while/Exit_1:0
cond_2/rnn/while/Exit_2:0
cond_2/rnn/while/Exit_3:0
$cond_2/rnn/while/iteration_counter:0
cond_2/rnn/zeros/Const:0
cond_2/rnn/zeros:0
cond_2/rnn_1/All:0
#cond_2/rnn_1/Assert/Assert/data_0:0
#cond_2/rnn_1/Assert/Assert/data_2:0
cond_2/rnn_1/Assert/Const:0
cond_2/rnn_1/Assert/Const_1:0
cond_2/rnn_1/CheckSeqLen:0
cond_2/rnn_1/Const:0
cond_2/rnn_1/Const_1:0
cond_2/rnn_1/Const_2:0
cond_2/rnn_1/Const_3:0
cond_2/rnn_1/Const_4:0
cond_2/rnn_1/Equal:0
cond_2/rnn_1/ExpandDims/dim:0
cond_2/rnn_1/ExpandDims:0
cond_2/rnn_1/Max:0
cond_2/rnn_1/Maximum/x:0
cond_2/rnn_1/Maximum:0
cond_2/rnn_1/Min:0
cond_2/rnn_1/Minimum:0
cond_2/rnn_1/Rank:0
cond_2/rnn_1/Rank_1:0
cond_2/rnn_1/Shape:0
cond_2/rnn_1/Shape_1:0
cond_2/rnn_1/Shape_2:0
cond_2/rnn_1/Shape_3:0
cond_2/rnn_1/TensorArray:0
cond_2/rnn_1/TensorArray:1
3cond_2/rnn_1/TensorArrayStack/TensorArrayGatherV3:0
1cond_2/rnn_1/TensorArrayStack/TensorArraySizeV3:0
+cond_2/rnn_1/TensorArrayStack/range/delta:0
+cond_2/rnn_1/TensorArrayStack/range/start:0
%cond_2/rnn_1/TensorArrayStack/range:0
'cond_2/rnn_1/TensorArrayUnstack/Shape:0
Icond_2/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
-cond_2/rnn_1/TensorArrayUnstack/range/delta:0
-cond_2/rnn_1/TensorArrayUnstack/range/start:0
'cond_2/rnn_1/TensorArrayUnstack/range:0
5cond_2/rnn_1/TensorArrayUnstack/strided_slice/stack:0
7cond_2/rnn_1/TensorArrayUnstack/strided_slice/stack_1:0
7cond_2/rnn_1/TensorArrayUnstack/strided_slice/stack_2:0
/cond_2/rnn_1/TensorArrayUnstack/strided_slice:0
cond_2/rnn_1/TensorArray_1:0
cond_2/rnn_1/TensorArray_1:1
cond_2/rnn_1/concat/axis:0
cond_2/rnn_1/concat/values_0:0
cond_2/rnn_1/concat:0
cond_2/rnn_1/concat_1/axis:0
cond_2/rnn_1/concat_1:0
cond_2/rnn_1/concat_2/axis:0
 cond_2/rnn_1/concat_2/values_0:0
cond_2/rnn_1/concat_2:0
cond_2/rnn_1/range/delta:0
cond_2/rnn_1/range/start:0
cond_2/rnn_1/range:0
cond_2/rnn_1/range_1/delta:0
cond_2/rnn_1/range_1/start:0
cond_2/rnn_1/range_1:0
cond_2/rnn_1/sequence_length:0
cond_2/rnn_1/stack:0
"cond_2/rnn_1/strided_slice/stack:0
$cond_2/rnn_1/strided_slice/stack_1:0
$cond_2/rnn_1/strided_slice/stack_2:0
cond_2/rnn_1/strided_slice:0
$cond_2/rnn_1/strided_slice_1/stack:0
&cond_2/rnn_1/strided_slice_1/stack_1:0
&cond_2/rnn_1/strided_slice_1/stack_2:0
cond_2/rnn_1/strided_slice_1:0
$cond_2/rnn_1/strided_slice_2/stack:0
&cond_2/rnn_1/strided_slice_2/stack_1:0
&cond_2/rnn_1/strided_slice_2/stack_2:0
cond_2/rnn_1/strided_slice_2:0
cond_2/rnn_1/time:0
cond_2/rnn_1/transpose:0
cond_2/rnn_1/transpose_1:0
cond_2/rnn_1/while/Exit:0
cond_2/rnn_1/while/Exit_1:0
cond_2/rnn_1/while/Exit_2:0
cond_2/rnn_1/while/Exit_3:0
&cond_2/rnn_1/while/iteration_counter:0
cond_2/rnn_1/zeros/Const:0
cond_2/rnn_1/zeros:0
cond_2/sub/Switch:0
cond_2/sub/y:0
cond_2/sub:0
cond_2/sub_1/y:0
cond_2/sub_1:0
cond_2/sub_2/y:0
cond_2/sub_2:0
cond_2/switch_f:0
transitions/read:0&
cond_2/switch_f:0cond_2/switch_f:0
Sum:0cond_2/sub/Switch:0$
cond_2/pred_id:0cond_2/pred_id:02
transitions/read:0cond_2/ExpandDims_1/Switch:02
biLSTM_layers/Reshape_1:0cond_2/Slice/Switch:02┤▒
cond_2/rnn/while/while_context *cond_2/rnn/while/LoopCond:02cond_2/rnn/while/Merge:0:cond_2/rnn/while/Identity:0Bcond_2/rnn/while/Exit:0Bcond_2/rnn/while/Exit_1:0Bcond_2/rnn/while/Exit_2:0Bcond_2/rnn/while/Exit_3:0J└
cond_2/ExpandDims_1:0
cond_2/rnn/CheckSeqLen:0
cond_2/rnn/Minimum:0
cond_2/rnn/TensorArray:0
Gcond_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
cond_2/rnn/TensorArray_1:0
cond_2/rnn/strided_slice_1:0
#cond_2/rnn/while/ArgMax/dimension:0
cond_2/rnn/while/ArgMax:0
cond_2/rnn/while/Cast:0
cond_2/rnn/while/Enter:0
cond_2/rnn/while/Enter_1:0
cond_2/rnn/while/Enter_2:0
cond_2/rnn/while/Enter_3:0
cond_2/rnn/while/Exit:0
cond_2/rnn/while/Exit_1:0
cond_2/rnn/while/Exit_2:0
cond_2/rnn/while/Exit_3:0
!cond_2/rnn/while/ExpandDims/dim:0
cond_2/rnn/while/ExpandDims:0
%cond_2/rnn/while/GreaterEqual/Enter:0
cond_2/rnn/while/GreaterEqual:0
cond_2/rnn/while/Identity:0
cond_2/rnn/while/Identity_1:0
cond_2/rnn/while/Identity_2:0
cond_2/rnn/while/Identity_3:0
cond_2/rnn/while/Less/Enter:0
cond_2/rnn/while/Less:0
cond_2/rnn/while/Less_1/Enter:0
cond_2/rnn/while/Less_1:0
cond_2/rnn/while/LogicalAnd:0
cond_2/rnn/while/LoopCond:0
(cond_2/rnn/while/Max/reduction_indices:0
cond_2/rnn/while/Max:0
cond_2/rnn/while/Merge:0
cond_2/rnn/while/Merge:1
cond_2/rnn/while/Merge_1:0
cond_2/rnn/while/Merge_1:1
cond_2/rnn/while/Merge_2:0
cond_2/rnn/while/Merge_2:1
cond_2/rnn/while/Merge_3:0
cond_2/rnn/while/Merge_3:1
 cond_2/rnn/while/NextIteration:0
"cond_2/rnn/while/NextIteration_1:0
"cond_2/rnn/while/NextIteration_2:0
"cond_2/rnn/while/NextIteration_3:0
cond_2/rnn/while/Select/Enter:0
cond_2/rnn/while/Select:0
cond_2/rnn/while/Select_1:0
cond_2/rnn/while/Switch:0
cond_2/rnn/while/Switch:1
cond_2/rnn/while/Switch_1:0
cond_2/rnn/while/Switch_1:1
cond_2/rnn/while/Switch_2:0
cond_2/rnn/while/Switch_2:1
cond_2/rnn/while/Switch_3:0
cond_2/rnn/while/Switch_3:1
*cond_2/rnn/while/TensorArrayReadV3/Enter:0
,cond_2/rnn/while/TensorArrayReadV3/Enter_1:0
$cond_2/rnn/while/TensorArrayReadV3:0
<cond_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
6cond_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
cond_2/rnn/while/add/y:0
cond_2/rnn/while/add:0
cond_2/rnn/while/add_1/Enter:0
cond_2/rnn/while/add_1:0
cond_2/rnn/while/add_2:0
cond_2/rnn/while/add_3/y:0
cond_2/rnn/while/add_3:0
cond_2/rnn/zeros:07
cond_2/rnn/Minimum:0cond_2/rnn/while/Less_1/Enter:0H
cond_2/rnn/TensorArray_1:0*cond_2/rnn/while/TensorArrayReadV3/Enter:0A
cond_2/rnn/CheckSeqLen:0%cond_2/rnn/while/GreaterEqual/Enter:07
cond_2/ExpandDims_1:0cond_2/rnn/while/add_1/Enter:0w
Gcond_2/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0,cond_2/rnn/while/TensorArrayReadV3/Enter_1:0=
cond_2/rnn/strided_slice_1:0cond_2/rnn/while/Less/Enter:05
cond_2/rnn/zeros:0cond_2/rnn/while/Select/Enter:0X
cond_2/rnn/TensorArray:0<cond_2/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0Rcond_2/rnn/while/Enter:0Rcond_2/rnn/while/Enter_1:0Rcond_2/rnn/while/Enter_2:0Rcond_2/rnn/while/Enter_3:0Zcond_2/rnn/strided_slice_1:02аЭ
 cond_2/rnn_1/while/while_context *cond_2/rnn_1/while/LoopCond:02cond_2/rnn_1/while/Merge:0:cond_2/rnn_1/while/Identity:0Bcond_2/rnn_1/while/Exit:0Bcond_2/rnn_1/while/Exit_1:0Bcond_2/rnn_1/while/Exit_2:0Bcond_2/rnn_1/while/Exit_3:0JТ
cond_2/rnn_1/CheckSeqLen:0
cond_2/rnn_1/Minimum:0
cond_2/rnn_1/TensorArray:0
Icond_2/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
cond_2/rnn_1/TensorArray_1:0
cond_2/rnn_1/strided_slice_1:0
cond_2/rnn_1/while/Enter:0
cond_2/rnn_1/while/Enter_1:0
cond_2/rnn_1/while/Enter_2:0
cond_2/rnn_1/while/Enter_3:0
cond_2/rnn_1/while/Exit:0
cond_2/rnn_1/while/Exit_1:0
cond_2/rnn_1/while/Exit_2:0
cond_2/rnn_1/while/Exit_3:0
#cond_2/rnn_1/while/ExpandDims/dim:0
cond_2/rnn_1/while/ExpandDims:0
cond_2/rnn_1/while/GatherNd:0
'cond_2/rnn_1/while/GreaterEqual/Enter:0
!cond_2/rnn_1/while/GreaterEqual:0
cond_2/rnn_1/while/Identity:0
cond_2/rnn_1/while/Identity_1:0
cond_2/rnn_1/while/Identity_2:0
cond_2/rnn_1/while/Identity_3:0
cond_2/rnn_1/while/Less/Enter:0
cond_2/rnn_1/while/Less:0
!cond_2/rnn_1/while/Less_1/Enter:0
cond_2/rnn_1/while/Less_1:0
cond_2/rnn_1/while/LogicalAnd:0
cond_2/rnn_1/while/LoopCond:0
cond_2/rnn_1/while/Merge:0
cond_2/rnn_1/while/Merge:1
cond_2/rnn_1/while/Merge_1:0
cond_2/rnn_1/while/Merge_1:1
cond_2/rnn_1/while/Merge_2:0
cond_2/rnn_1/while/Merge_2:1
cond_2/rnn_1/while/Merge_3:0
cond_2/rnn_1/while/Merge_3:1
"cond_2/rnn_1/while/NextIteration:0
$cond_2/rnn_1/while/NextIteration_1:0
$cond_2/rnn_1/while/NextIteration_2:0
$cond_2/rnn_1/while/NextIteration_3:0
!cond_2/rnn_1/while/Select/Enter:0
cond_2/rnn_1/while/Select:0
cond_2/rnn_1/while/Select_1:0
cond_2/rnn_1/while/Shape:0
cond_2/rnn_1/while/Squeeze:0
cond_2/rnn_1/while/Switch:0
cond_2/rnn_1/while/Switch:1
cond_2/rnn_1/while/Switch_1:0
cond_2/rnn_1/while/Switch_1:1
cond_2/rnn_1/while/Switch_2:0
cond_2/rnn_1/while/Switch_2:1
cond_2/rnn_1/while/Switch_3:0
cond_2/rnn_1/while/Switch_3:1
,cond_2/rnn_1/while/TensorArrayReadV3/Enter:0
.cond_2/rnn_1/while/TensorArrayReadV3/Enter_1:0
&cond_2/rnn_1/while/TensorArrayReadV3:0
>cond_2/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
8cond_2/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3:0
cond_2/rnn_1/while/add/y:0
cond_2/rnn_1/while/add:0
cond_2/rnn_1/while/add_1/y:0
cond_2/rnn_1/while/add_1:0
 cond_2/rnn_1/while/range/delta:0
 cond_2/rnn_1/while/range/start:0
cond_2/rnn_1/while/range:0
cond_2/rnn_1/while/stack:0
(cond_2/rnn_1/while/strided_slice/stack:0
*cond_2/rnn_1/while/strided_slice/stack_1:0
*cond_2/rnn_1/while/strided_slice/stack_2:0
"cond_2/rnn_1/while/strided_slice:0
cond_2/rnn_1/zeros:0E
cond_2/rnn_1/CheckSeqLen:0'cond_2/rnn_1/while/GreaterEqual/Enter:0{
Icond_2/rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0.cond_2/rnn_1/while/TensorArrayReadV3/Enter_1:0A
cond_2/rnn_1/strided_slice_1:0cond_2/rnn_1/while/Less/Enter:09
cond_2/rnn_1/zeros:0!cond_2/rnn_1/while/Select/Enter:0\
cond_2/rnn_1/TensorArray:0>cond_2/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0;
cond_2/rnn_1/Minimum:0!cond_2/rnn_1/while/Less_1/Enter:0L
cond_2/rnn_1/TensorArray_1:0,cond_2/rnn_1/while/TensorArrayReadV3/Enter:0Rcond_2/rnn_1/while/Enter:0Rcond_2/rnn_1/while/Enter_1:0Rcond_2/rnn_1/while/Enter_2:0Rcond_2/rnn_1/while/Enter_3:0Zcond_2/rnn_1/strided_slice_1:0"
train_op

Adam"Чд
while_contextДдАд
№С
9biLSTM_layers/bidirectional_rnn/fw/fw/while/while_context *6biLSTM_layers/bidirectional_rnn/fw/fw/while/LoopCond:023biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge:0:6biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity:0B2biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit:0B4biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_1:0B4biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_2:0B4biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_3:0B4biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_4:0Bgradients/f_count_5:0JРЛ
3biLSTM_layers/bidirectional_rnn/fw/fw/CheckSeqLen:0
/biLSTM_layers/bidirectional_rnn/fw/fw/Minimum:0
3biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray:0
bbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
5biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray_1:0
7biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_1:0
3biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter:0
5biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_1:0
5biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_2:0
5biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_3:0
5biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_4:0
2biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit:0
4biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_1:0
4biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_2:0
4biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_3:0
4biLSTM_layers/bidirectional_rnn/fw/fw/while/Exit_4:0
@biLSTM_layers/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter:0
:biLSTM_layers/bidirectional_rnn/fw/fw/while/GreaterEqual:0
6biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity:0
8biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_1:0
8biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_2:0
8biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_3:0
8biLSTM_layers/bidirectional_rnn/fw/fw/while/Identity_4:0
8biLSTM_layers/bidirectional_rnn/fw/fw/while/Less/Enter:0
2biLSTM_layers/bidirectional_rnn/fw/fw/while/Less:0
:biLSTM_layers/bidirectional_rnn/fw/fw/while/Less_1/Enter:0
4biLSTM_layers/bidirectional_rnn/fw/fw/while/Less_1:0
8biLSTM_layers/bidirectional_rnn/fw/fw/while/LogicalAnd:0
6biLSTM_layers/bidirectional_rnn/fw/fw/while/LoopCond:0
3biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge:0
3biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge:1
5biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_1:0
5biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_1:1
5biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_2:0
5biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_2:1
5biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_3:0
5biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_3:1
5biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_4:0
5biLSTM_layers/bidirectional_rnn/fw/fw/while/Merge_4:1
;biLSTM_layers/bidirectional_rnn/fw/fw/while/NextIteration:0
=biLSTM_layers/bidirectional_rnn/fw/fw/while/NextIteration_1:0
=biLSTM_layers/bidirectional_rnn/fw/fw/while/NextIteration_2:0
=biLSTM_layers/bidirectional_rnn/fw/fw/while/NextIteration_3:0
=biLSTM_layers/bidirectional_rnn/fw/fw/while/NextIteration_4:0
:biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter:0
4biLSTM_layers/bidirectional_rnn/fw/fw/while/Select:0
6biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1:0
6biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2:0
4biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch:0
4biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch:1
6biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_1:0
6biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_1:1
6biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_2:0
6biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_2:1
6biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_3:0
6biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_3:1
6biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_4:0
6biLSTM_layers/bidirectional_rnn/fw/fw/while/Switch_4:1
EbiLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter:0
GbiLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1:0
?biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3:0
WbiLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
QbiLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3:0
3biLSTM_layers/bidirectional_rnn/fw/fw/while/add/y:0
1biLSTM_layers/bidirectional_rnn/fw/fw/while/add:0
5biLSTM_layers/bidirectional_rnn/fw/fw/while/add_1/y:0
3biLSTM_layers/bidirectional_rnn/fw/fw/while/add_1:0
EbiLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter:0
?biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd:0
=biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Const:0
DbiLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter:0
>biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul:0
?biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid:0
AbiLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_1:0
AbiLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Sigmoid_2:0
<biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh:0
>biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/Tanh_1:0
=biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add/y:0
;biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add:0
=biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1:0
CbiLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat/axis:0
>biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat:0
;biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul:0
=biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1:0
=biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2:0
GbiLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/split/split_dim:0
=biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/split:0
=biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/split:1
=biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/split:2
=biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/split:3
-biLSTM_layers/bidirectional_rnn/fw/fw/zeros:0
8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/read:0
:biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/read:0
gradients/Add_1/y:0
gradients/Add_1:0
gradients/Merge_2:0
gradients/Merge_2:1
gradients/NextIteration_2:0
gradients/Switch_2:0
gradients/Switch_2:1
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Enter:0
Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/StackPushV2:0
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc:0
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/Enter:0
\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/StackPushV2:0
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_acc:0
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/Enter:0
\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/StackPushV2:0
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_acc:0
xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
~gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0
bgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0
jgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0
ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2:0
pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1:0
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0
jgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape:0
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/Shape_1:0
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0
lgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2:0
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/Shape:0
Sgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Shape:0
Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/Enter:0
`gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/StackPushV2:0
Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_acc:0
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0
jgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0
ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1:0
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0
jgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/Enter:0
\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/StackPushV2:0
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_acc:0
Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/Enter:0
^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2:0
Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape:0
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Shape_1:0
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0
jgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0
ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2:0
pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1:0
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0
jgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/Enter:0
\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/StackPushV2:0
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_acc:0
Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/Enter:0
^gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2:0
Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape:0
Tgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Shape_1:0
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0
lgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2:0
ngradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/Enter:0
\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/StackPushV2:0
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_acc:0
Pgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape:0
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Shape_1:0
gradients/f_count_3:0
gradients/f_count_4:0
gradients/f_count_5:0╕
Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/f_acc:0Zgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/ShapeN/Enter:0╪
jgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0jgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0╘
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0░
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/f_acc:0Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul/Enter:0░
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/f_acc:0Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/Mul_1/Enter:0╝
\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0\gradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0╪
jgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0jgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0░
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/f_acc:0Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul/Enter:0и
Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/f_acc:0Rgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/Select/Enter:0┤
Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/Mul_1/Enter:0┤
Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0Xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/Mul_1/Enter:0О
3biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray:0WbiLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0Ї
xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0xgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0╪
jgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0jgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0╨
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0░
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/f_acc:0Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_1_grad/zeros_like/Enter:0╘
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0s
7biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_1:08biLSTM_layers/bidirectional_rnn/fw/fw/while/Less/Enter:0~
5biLSTM_layers/bidirectional_rnn/fw/fw/TensorArray_1:0EbiLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter:0╘
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0╘
hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0hgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0w
3biLSTM_layers/bidirectional_rnn/fw/fw/CheckSeqLen:0@biLSTM_layers/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter:0╨
fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0fgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0m
/biLSTM_layers/bidirectional_rnn/fw/fw/Minimum:0:biLSTM_layers/bidirectional_rnn/fw/fw/while/Less_1/Enter:0В
:biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/read:0DbiLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/MatMul/Enter:0k
-biLSTM_layers/bidirectional_rnn/fw/fw/zeros:0:biLSTM_layers/bidirectional_rnn/fw/fw/while/Select/Enter:0Б
8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/read:0EbiLSTM_layers/bidirectional_rnn/fw/fw/while/lstm_cell/BiasAdd/Enter:0н
bbiLSTM_layers/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0GbiLSTM_layers/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1:0░
Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/f_acc:0Vgradients/biLSTM_layers/bidirectional_rnn/fw/fw/while/Select_2_grad/zeros_like/Enter:0R3biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter:0R5biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_1:0R5biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_2:0R5biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_3:0R5biLSTM_layers/bidirectional_rnn/fw/fw/while/Enter_4:0Rgradients/f_count_4:0Z7biLSTM_layers/bidirectional_rnn/fw/fw/strided_slice_1:0
№С
9biLSTM_layers/bidirectional_rnn/bw/bw/while/while_context *6biLSTM_layers/bidirectional_rnn/bw/bw/while/LoopCond:023biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge:0:6biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity:0B2biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit:0B4biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_1:0B4biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_2:0B4biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_3:0B4biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_4:0Bgradients/f_count_8:0JРЛ
3biLSTM_layers/bidirectional_rnn/bw/bw/CheckSeqLen:0
/biLSTM_layers/bidirectional_rnn/bw/bw/Minimum:0
3biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray:0
bbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
5biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray_1:0
7biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_1:0
3biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter:0
5biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_1:0
5biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_2:0
5biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_3:0
5biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_4:0
2biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit:0
4biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_1:0
4biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_2:0
4biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_3:0
4biLSTM_layers/bidirectional_rnn/bw/bw/while/Exit_4:0
@biLSTM_layers/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter:0
:biLSTM_layers/bidirectional_rnn/bw/bw/while/GreaterEqual:0
6biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity:0
8biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_1:0
8biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_2:0
8biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_3:0
8biLSTM_layers/bidirectional_rnn/bw/bw/while/Identity_4:0
8biLSTM_layers/bidirectional_rnn/bw/bw/while/Less/Enter:0
2biLSTM_layers/bidirectional_rnn/bw/bw/while/Less:0
:biLSTM_layers/bidirectional_rnn/bw/bw/while/Less_1/Enter:0
4biLSTM_layers/bidirectional_rnn/bw/bw/while/Less_1:0
8biLSTM_layers/bidirectional_rnn/bw/bw/while/LogicalAnd:0
6biLSTM_layers/bidirectional_rnn/bw/bw/while/LoopCond:0
3biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge:0
3biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge:1
5biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_1:0
5biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_1:1
5biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_2:0
5biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_2:1
5biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_3:0
5biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_3:1
5biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_4:0
5biLSTM_layers/bidirectional_rnn/bw/bw/while/Merge_4:1
;biLSTM_layers/bidirectional_rnn/bw/bw/while/NextIteration:0
=biLSTM_layers/bidirectional_rnn/bw/bw/while/NextIteration_1:0
=biLSTM_layers/bidirectional_rnn/bw/bw/while/NextIteration_2:0
=biLSTM_layers/bidirectional_rnn/bw/bw/while/NextIteration_3:0
=biLSTM_layers/bidirectional_rnn/bw/bw/while/NextIteration_4:0
:biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter:0
4biLSTM_layers/bidirectional_rnn/bw/bw/while/Select:0
6biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1:0
6biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2:0
4biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch:0
4biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch:1
6biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_1:0
6biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_1:1
6biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_2:0
6biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_2:1
6biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_3:0
6biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_3:1
6biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_4:0
6biLSTM_layers/bidirectional_rnn/bw/bw/while/Switch_4:1
EbiLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter:0
GbiLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1:0
?biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3:0
WbiLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
QbiLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3:0
3biLSTM_layers/bidirectional_rnn/bw/bw/while/add/y:0
1biLSTM_layers/bidirectional_rnn/bw/bw/while/add:0
5biLSTM_layers/bidirectional_rnn/bw/bw/while/add_1/y:0
3biLSTM_layers/bidirectional_rnn/bw/bw/while/add_1:0
EbiLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter:0
?biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd:0
=biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Const:0
DbiLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter:0
>biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul:0
?biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid:0
AbiLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_1:0
AbiLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Sigmoid_2:0
<biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh:0
>biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/Tanh_1:0
=biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add/y:0
;biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add:0
=biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1:0
CbiLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat/axis:0
>biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat:0
;biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul:0
=biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1:0
=biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2:0
GbiLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/split/split_dim:0
=biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/split:0
=biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/split:1
=biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/split:2
=biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/split:3
-biLSTM_layers/bidirectional_rnn/bw/bw/zeros:0
8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/read:0
:biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/read:0
gradients/Add_2/y:0
gradients/Add_2:0
gradients/Merge_4:0
gradients/Merge_4:1
gradients/NextIteration_4:0
gradients/Switch_4:0
gradients/Switch_4:1
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Enter:0
Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/StackPushV2:0
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc:0
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/Enter:0
\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/StackPushV2:0
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_acc:0
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/Enter:0
\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/StackPushV2:0
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_acc:0
xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
~gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0
bgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0
jgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0
ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2:0
pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1:0
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0
jgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape:0
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/Shape_1:0
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0
lgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2:0
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/Shape:0
Sgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Shape:0
Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/Enter:0
`gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/StackPushV2:0
Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_acc:0
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0
jgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0
ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2:0
pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1:0
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0
jgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/Enter:0
\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/StackPushV2:0
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_acc:0
Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/Enter:0
^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2:0
Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape:0
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Shape_1:0
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0
jgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0
ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2:0
pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1:0
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0
jgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/Enter:0
\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/StackPushV2:0
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_acc:0
Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/Enter:0
^gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2:0
Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape:0
Tgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Shape_1:0
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0
lgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2:0
ngradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1:0
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/Enter:0
\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/StackPushV2:0
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_acc:0
Pgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape:0
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Shape_1:0
gradients/f_count_6:0
gradients/f_count_7:0
gradients/f_count_8:0╘
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc:0hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter:0╘
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1:0hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1:0╨
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc:0fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter:0О
3biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray:0WbiLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0н
bbiLSTM_layers/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0GbiLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1:0░
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/f_acc:0Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_2_grad/zeros_like/Enter:0Б
8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/read:0EbiLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/BiasAdd/Enter:0s
7biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_1:08biLSTM_layers/bidirectional_rnn/bw/bw/while/Less/Enter:0~
5biLSTM_layers/bidirectional_rnn/bw/bw/TensorArray_1:0EbiLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter:0╪
jgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1:0jgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1:0╕
Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/f_acc:0Zgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/ShapeN/Enter:0╘
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc:0hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter:0░
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/f_acc:0Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/Mul_1/Enter:0░
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/f_acc:0Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul/Enter:0╝
\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/f_acc:0\gradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul_grad/MatMul_1/Enter:0╪
jgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1:0jgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1:0░
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/f_acc:0Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul/Enter:0w
3biLSTM_layers/bidirectional_rnn/bw/bw/CheckSeqLen:0@biLSTM_layers/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter:0и
Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/f_acc:0Rgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/Select/Enter:0┤
Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/f_acc:0Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/Mul_1/Enter:0┤
Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/f_acc:0Xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/Mul_1/Enter:0m
/biLSTM_layers/bidirectional_rnn/bw/bw/Minimum:0:biLSTM_layers/bidirectional_rnn/bw/bw/while/Less_1/Enter:0Ї
xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0xgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0╨
fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc:0fgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter:0╪
jgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1:0jgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1:0k
-biLSTM_layers/bidirectional_rnn/bw/bw/zeros:0:biLSTM_layers/bidirectional_rnn/bw/bw/while/Select/Enter:0В
:biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/read:0DbiLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/MatMul/Enter:0░
Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/f_acc:0Vgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/Select_1_grad/zeros_like/Enter:0╘
hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc:0hgradients/biLSTM_layers/bidirectional_rnn/bw/bw/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter:0R3biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter:0R5biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_1:0R5biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_2:0R5biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_3:0R5biLSTM_layers/bidirectional_rnn/bw/bw/while/Enter_4:0Rgradients/f_count_7:0Z7biLSTM_layers/bidirectional_rnn/bw/bw/strided_slice_1:0"╓'
	variables╚'┼'
╡
"embedding_layer/embedding_matrix:0'embedding_layer/embedding_matrix/Assign'embedding_layer/embedding_matrix/read:02=embedding_layer/embedding_matrix/Initializer/random_uniform:0
Б
5biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel:0:biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Assign:biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/read:02PbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Initializer/random_uniform:0
Ё
3biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias:08biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Assign8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/read:02EbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Initializer/zeros:0
Б
5biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel:0:biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Assign:biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/read:02PbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Initializer/random_uniform:0
Ё
3biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias:08biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Assign8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/read:02EbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Initializer/zeros:0
Г
biLSTM_layers/W_out:0biLSTM_layers/W_out/AssignbiLSTM_layers/W_out/read:022biLSTM_layers/W_out/Initializer/truncated_normal:0
s
biLSTM_layers/b:0biLSTM_layers/b/AssignbiLSTM_layers/b/read:02.biLSTM_layers/b/Initializer/truncated_normal:0
a
transitions:0transitions/Assigntransitions/read:02(transitions/Initializer/random_uniform:0
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
└
'embedding_layer/embedding_matrix/Adam:0,embedding_layer/embedding_matrix/Adam/Assign,embedding_layer/embedding_matrix/Adam/read:029embedding_layer/embedding_matrix/Adam/Initializer/zeros:0
╚
)embedding_layer/embedding_matrix/Adam_1:0.embedding_layer/embedding_matrix/Adam_1/Assign.embedding_layer/embedding_matrix/Adam_1/read:02;embedding_layer/embedding_matrix/Adam_1/Initializer/zeros:0
М
:biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam:0?biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Assign?biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam/read:02LbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam/Initializer/zeros:0
Ф
<biLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1:0AbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/AssignAbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/read:02NbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/Initializer/zeros:0
Д
8biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam:0=biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam/Assign=biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam/read:02JbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam/Initializer/zeros:0
М
:biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1:0?biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Assign?biLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/read:02LbiLSTM_layers/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/Initializer/zeros:0
М
:biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam:0?biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Assign?biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam/read:02LbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam/Initializer/zeros:0
Ф
<biLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1:0AbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/AssignAbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/read:02NbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/Initializer/zeros:0
Д
8biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam:0=biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam/Assign=biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam/read:02JbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam/Initializer/zeros:0
М
:biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1:0?biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Assign?biLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/read:02LbiLSTM_layers/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/Initializer/zeros:0
М
biLSTM_layers/W_out/Adam:0biLSTM_layers/W_out/Adam/AssignbiLSTM_layers/W_out/Adam/read:02,biLSTM_layers/W_out/Adam/Initializer/zeros:0
Ф
biLSTM_layers/W_out/Adam_1:0!biLSTM_layers/W_out/Adam_1/Assign!biLSTM_layers/W_out/Adam_1/read:02.biLSTM_layers/W_out/Adam_1/Initializer/zeros:0
|
biLSTM_layers/b/Adam:0biLSTM_layers/b/Adam/AssignbiLSTM_layers/b/Adam/read:02(biLSTM_layers/b/Adam/Initializer/zeros:0
Д
biLSTM_layers/b/Adam_1:0biLSTM_layers/b/Adam_1/AssignbiLSTM_layers/b/Adam_1/read:02*biLSTM_layers/b/Adam_1/Initializer/zeros:0
l
transitions/Adam:0transitions/Adam/Assigntransitions/Adam/read:02$transitions/Adam/Initializer/zeros:0
t
transitions/Adam_1:0transitions/Adam_1/Assigntransitions/Adam_1/read:02&transitions/Adam_1/Initializer/zeros:0*о
ner_nameб
4
inputs_x(
inputs:0                  
 
	keep_prob
keep_prob:0=
decode_tags.
cond_2/Merge:0                  ner_name