# fungt-mesher

FunGT Mesher is a C++ and SYCL application for generating 3D data from a
single image. It implements the inference pipeline on the GPU using pretrained
TripoSR model weights, encoding the input image, generating triplane features,
and producing a density grid that can be converted into a 3D mesh.

Set the model weights path in the environment:

```bash
export MODEL_PATH=/path/to/model_weights
```

The committed `fungt-mesher.config` reads this variable:

```ini
model_path=${MODEL_PATH}
```

Run the generator with an input image and output density file:

```bash
./FunGTMesher image.png --out density.bin
```

## Model weights

FunGT Mesher does not include or distribute TripoSR model weights.

Users must obtain their own copy of the TripoSR model weights and comply with
the applicable license. After downloading the weights, set their directory
through the `MODEL_PATH` environment variable as shown above.

The model directory must contain `weights.json` and all binary tensor files
referenced by that manifest:

```text
triposr_weights/
├── weights.json
├── image_tokenizer_model_embeddings_cls_token.bin
├── image_tokenizer_model_embeddings_position_embeddings.bin
└── ...
```

`weights.json` records each tensor name, binary filename, shape, and data type.
FunGT Mesher reads this manifest to locate and load the model tensors.

FunGT Mesher is an independent C++ and SYCL implementation of the TripoSR
inference pipeline. It is not affiliated with or endorsed by Stability AI or
Tripo AI.
