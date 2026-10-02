#if !defined(_FUNGT_MESHER_HPP_H)
#define _FUNGT_MESHER_HPP_H
#include "ComputeBackends/compute_backends.hpp"
#include "ImageEncoder/image_encoder.hpp"
#include "ImageTokenizer/image_tokenizer.hpp"
#include "TriplaneDecoder/triplane_decoder.hpp"
#include "WeightLoader/weight_loader.hpp"
#include <funlib/funlib.hpp>

class FunGTMesher {
private:
  sycl::queue m_queue;
  fgtm_tools::WeightLoader m_weight_loader;
  std::shared_ptr<fgtm_tools::ImageTokenizer> m_image_tokenizer;
  std::shared_ptr<fgtm_tools::ImageEncoder> m_image_encoder;
  std::shared_ptr<fgtm_tools::TriplaneDecoder> m_triplane_decoder;
  flib::vendor m_vendor;
  flib::backend m_backend;
  std::string m_backend_name;
  bool m_model_set = false;

  FunGTMesher();
  void export_image_features(const flib::ftensor &features,
                             const std::string &path_output);
  flib::ftensor drop_cls_token(const flib::ftensor &tokens);
  DecoderWeights load_decoder_weights();
  flib::ftensor decode_triplane(const flib::ftensor &image_tokens,
                                const std::string &diagnostic_directory);

public:
  ~FunGTMesher() = default;
  void generate(const std::string &path_image, const std::string &path_output);
  flib::ftensor image_tokenizer(const std::string &path_image);
  void image_encoder(flib::ftensor &tokens,
                     const std::string &diagnostic_directory);

  void set_model(const std::string &path_to_model = "") {
    if (!path_to_model.empty()) {
      m_weight_loader.set_model_path(path_to_model);
      m_model_set = true;
    } else {
      // later we use the default path for the fungt-model
    }
  }
  static std::unique_ptr<FunGTMesher> create_mesher() {

    return std::unique_ptr<FunGTMesher>(new FunGTMesher());
  }
  // if user wants to set more features to tokenizer
  std::shared_ptr<fgtm_tools::ImageTokenizer> getImageTokenizer() {
    return m_image_tokenizer;
  }
};

#endif // _FUNGT_MESHER_HPP_H
