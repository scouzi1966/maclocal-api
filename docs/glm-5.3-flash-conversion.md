# GLM-5.3 Flash checkpoint conversion

`afm mlx-convert` can convert a complete, already-local copy of the official
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)
raw FP8 multimodal checkpoint into the self-contained MLX layout used by AFM.
AFM does not download this checkpoint as part of conversion.

The source revision validated during implementation is
`04c4e9e95c5da8862dced7e5056455116f83a7e0`. Its SafeTensor index declares
62 shards, 76,108 tensors, and 328,326,771,576 bytes of tensor payload. Pin the
same full revision on the command line unless the local directory is already a
Hugging Face `snapshots/<revision>` directory:

```sh
afm mlx-convert \
  --source /Volumes/edata2/models/zai-org/GLM-5.3-Flash \
  --output /Volumes/edata2/models/GLM-5.3-Flash-AFM-MLX-4bit \
  --source-revision 04c4e9e95c5da8862dced7e5056455116f83a7e0 \
  --profile mlx-affine-4
```

The command fails before conversion unless the source is a local directory,
the destination is not a filesystem or mounted-volume root, the source and
destination are disjoint after resolving symlinks, every
indexed shard and required processor/tokenizer/template asset is present, and
the destination volume reports at least 600 GB free. The output-size estimate
is calculated from the actual source headers; the conservative 600 GB floor
also allows atomic partial output and resume state. On a verified resume, AFM
asks AFMKit to revalidate the complete private manifest, current conversion
plan, source revision/config/index/assets/full shard hashes, and SafeTensor
outputs before crediting completed bytes. It retains a 64 GB atomic-unit margin, so
the initial floor does not incorrectly block a job merely because its own
partial output consumed space. Provider-verified physical output bytes include
SafeTensor headers and may therefore exceed the tensor-payload estimate; the
remaining payload estimate is clamped to zero in that case. This is a storage
floor, not an assertion that the final checkpoint will occupy 600 GB.

Conversion behavior:

- validates SafeTensor header dtypes and accepts FP8 only when the header says
  `F8_E4M3`; ordinary `U8` is never inferred to be FP8;
- decodes E4M3 and applies the declared 128 x 128 inverse scales in FP32 before
  bounded affine 4-bit/group-64 quantization;
- reconstructs numerically ordered routed experts across source shards;
- converts the complete text and vision namespaces, including the required
  PyTorch-to-MLX Conv3d patch-embedding and Conv2d downsample layouts, and
  atomically copies the processor, tokenizer, and chat-template assets;
- writes atomic per-unit SafeTensor outputs and a checksummed resume manifest
  tied to consistent local revision evidence, config, index, full shard
  content hashes, and support-asset hashes;
- explicitly omits the one MTP layer, sets `num_nextn_predict_layers` to zero,
  and records that omission in `config.json` provenance.

The implementation bounds working data to one source shard and one routed
expert projection unit rather than materializing the full 328 GB checkpoint.
Based on the 62-shard layout and published model dimensions, expected peak
working memory is in the low tens of gigabytes, but that estimate has not yet
been measured against the complete source. Conversion should be run on a
machine with ample unified memory and without another large model loaded.

## Validation limit

The FP8 decoder, block scaling, quantization error bound, cross-shard expert
ordering, complete vision-name and convolution-layout mapping, multimodal assets, MTP omission,
dispatcher, storage preflight, and resumability are covered with synthetic
SafeTensor fixtures. The full 328 GB checkpoint was not present locally during
implementation, so no claim is made yet for full-checkpoint conversion time,
measured peak memory, final output size, or end-to-end generated text/image
parity. The GLM runtime architecture remains a separate AFMKit change; a
converted checkpoint requires that runtime support before inference.
