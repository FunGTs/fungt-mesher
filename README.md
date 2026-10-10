# FunGT-Mesher

FunGT-Mesher is a GPU accelerated engine written in C++ and SYCL that generates
3D models from images. Its goal is to transform a single image into a usable 3D mesh
through image encoding, a learned triplane representation, neural density
evaluation, and surface extraction. The public interface will hide model and
compute details behind a simple generation workflow.

The intended command-line usage is:

```bash
./fungt-mesher image.png -out model.obj
```
