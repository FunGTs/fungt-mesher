#include "obj_exporter.hpp"

#include <assimp/Exporter.hpp>
#include <assimp/material.h>
#include <assimp/scene.h>
#include <stdexcept>
namespace fgtm {

void ObjExporter::export_mesh(
    const flib::graphics::marching_cubes::Mesh &mesh,
    const std::string &path,
    sycl::queue &queue) const {
  const auto positions = mesh.positions.to_host(queue);
  const auto normals = mesh.normals.to_host(queue);
  const auto colors = mesh.colors.to_host(queue);
  const std::size_t vertex_count = mesh.positions.getShape()[0];

  if (positions.size() != vertex_count * 3 ||
      normals.size() != vertex_count * 3 ||
      colors.size() != vertex_count * 3 || vertex_count % 3 != 0) {
    throw std::invalid_argument("Invalid marching-cubes mesh layout");
  }

  aiScene scene;
  scene.mNumMeshes = 1;
  scene.mMeshes = new aiMesh *[1];
  scene.mMeshes[0] = new aiMesh();
  aiMesh *output = scene.mMeshes[0];
  output->mPrimitiveTypes = aiPrimitiveType_TRIANGLE;
  output->mNumVertices = static_cast<unsigned int>(vertex_count);
  output->mVertices = new aiVector3D[vertex_count];
  output->mNormals = new aiVector3D[vertex_count];
  output->mColors[0] = new aiColor4D[vertex_count];
  output->mNumFaces = static_cast<unsigned int>(vertex_count / 3);
  output->mFaces = new aiFace[output->mNumFaces];

  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    output->mVertices[vertex] = aiVector3D(
        positions[vertex * 3], positions[vertex * 3 + 1],
        positions[vertex * 3 + 2]);
    output->mNormals[vertex] = aiVector3D(
        normals[vertex * 3], normals[vertex * 3 + 1],
        normals[vertex * 3 + 2]);
    output->mColors[0][vertex] = aiColor4D(
        colors[vertex * 3], colors[vertex * 3 + 1],
        colors[vertex * 3 + 2], 1.0f);
  }

  for (unsigned int face = 0; face < output->mNumFaces; ++face) {
    output->mFaces[face].mNumIndices = 3;
    output->mFaces[face].mIndices = new unsigned int[3]{
        face * 3, face * 3 + 1, face * 3 + 2};
  }

  scene.mNumMaterials = 1;
  scene.mMaterials = new aiMaterial *[1];
  scene.mMaterials[0] = new aiMaterial();
  aiColor4D diffuse(1.0f, 1.0f, 1.0f, 1.0f);
  scene.mMaterials[0]->AddProperty(
      &diffuse, 1, AI_MATKEY_COLOR_DIFFUSE);
  output->mMaterialIndex = 0;
  scene.mRootNode = new aiNode();
  scene.mRootNode->mNumMeshes = 1;
  scene.mRootNode->mMeshes = new unsigned int[1]{0};

  Assimp::Exporter exporter;
  const aiReturn result = exporter.Export(&scene, "obj", path);
  if (result != aiReturn_SUCCESS) {
    throw std::runtime_error(exporter.GetErrorString());
  }
}

} // namespace fgtm
